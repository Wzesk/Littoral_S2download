# Standard library imports
import os
import pathlib
from pathlib import Path
from typing import Collection, Dict, Tuple, Union
import traceback

# Third-party imports
from tqdm import tqdm
import numpy as np
import pandas as pd
import pyproj
import pyTMD.io
import pyTMD.io.model
import pyTMD.predict
import pyTMD.time


######################################################################################################
# Updated Tidal correction functions
######################################################################################################

def calculate_tide_corrections(
    df: pd.DataFrame, reference_elevation: float, beach_slope: float
):
    """
    Applies tidal correction to the timeseries data.

    Args:
    - df (DataFrame): Input data with tide predictions and timeseries data.
    - reference_elevation (float): Reference elevation value.
    - beach_slope (float): Beach slope value.

    Returns:
    - DataFrame: Tidally corrected data.
    """
    correction = (df["tide"] - reference_elevation) / beach_slope
    if "cross_distance" not in df.columns:
        df["cross_distance"] = 0
    df["cross_distance"] = df["cross_distance"] + correction
    return df.drop(columns=["correction"], errors="ignore")

def model_tides(
    x,
    y,
    time,
    model="FES2022",
    directory=None,
    epsg=4326,
    method="bilinear",
    extrapolate=True,
    cutoff=10.0,
):
    """
    Compute tides at points and times using tidal harmonics.
    If multiple x, y points are provided, tides will be
    computed for all timesteps at each point.

    This function supports any tidal model supported by
    `pyTMD`, including the FES2014 Finite Element Solution
    tide model, and FES2022 Finite Element Solution
    tide model.

    This function is a modification of the `pyTMD`
    package's `compute_tide_corrections` function, adapted
    to process multiple timesteps for multiple input point
    locations. For more info:
    https://pytmd.readthedocs.io/en/stable/user_guide/compute_tide_corrections.html

    Parameters:
    -----------
    x, y : float or list of floats
        One or more x and y coordinates used to define
        the location at which to model tides. By default these
        coordinates should be lat/lon; use `epsg` if they
        are in a custom coordinate reference system.
    time : A datetime array or pandas.DatetimeIndex
        An array containing 'datetime64[ns]' values or a
        'pandas.DatetimeIndex' providing the times at which to
        model tides in UTC time.
    model : string
        The tide model used to model tides. Options include:
        - "fes2022b" (only pre-configured option on DEA Sandbox)
        - "TPXO8-atlas"
        - "TPXO9-atlas-v5"
    directory : string
        The directory containing tide model data files. These
        data files should be stored in sub-folders for each
        model that match the structure provided by `pyTMD`:
        https://pytmd.readthedocs.io/en/latest/getting_started/Getting-Started.html#directories
        For example:
        - {directory}/fes2014/ocean_tide/
          {directory}/fes2014/load_tide/
    epsg : int
        Input coordinate system for 'x' and 'y' coordinates.
        Defaults to 4326 (WGS84).
    method : string
        Method used to interpolate tidal contsituents
        from model files. Options include:
        - bilinear: quick bilinear interpolation
        - spline: scipy bivariate spline interpolation
        - linear, nearest: scipy regular grid interpolations
    extrapolate : bool
        Whether to extrapolate tides for locations outside of
        the tide modelling domain using nearest-neighbor
    cutoff : int or float
        Extrapolation cutoff in kilometers. Set to `np.inf`
        to extrapolate for all points.

    Returns
    -------
    A pandas.DataFrame containing tide heights for all the xy points and their corresponding time
    """
    # Check tide directory is accessible
    if directory is not None:
        directory = pathlib.Path(directory).expanduser()
        if not directory.exists():
            raise FileNotFoundError("Invalid tide directory")
    # Validate input arguments
    assert method in ("bilinear", "spline", "linear", "nearest")

    if "fes2022" in model.lower():
        model = 'FES2022'
    # Get parameters for tide model; use custom definition file for
    model = pyTMD.io.model(directory, format="netcdf", compressed=False).elevation(
        model
    )


    # If time passed as a single Timestamp, convert to datetime64
    if isinstance(time, pd.Timestamp):
        time = time.to_datetime64()

    # Handle numeric or array inputs
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    time = np.atleast_1d(time)

    # Determine point and time counts
    assert len(x) == len(y), "x and y must be the same length"
    n_points = len(x)
    n_times = len(time)

    # Converting x,y from EPSG to latitude/longitude
    try:
        # EPSG projection code string or int
        crs1 = pyproj.CRS.from_epsg(int(epsg))
    except (ValueError, pyproj.exceptions.CRSError):
        # Projection SRS string
        crs1 = pyproj.CRS.from_string(epsg)

    # Output coordinate reference system
    crs2 = pyproj.CRS.from_epsg(4326)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    lon, lat = transformer.transform(x.flatten(), y.flatten())

    
    # Convert datetime
    timescale = pyTMD.time.timescale().from_datetime(time.flatten())
    
    n_points = len(x)
    # number of time points
    n_times = len(time)

    
    amp, ph = pyTMD.io.FES.extract_constants(
        lon,
        lat,
        model.model_file,
        type=model.type,
        version=model.version,
        method=method,
        extrapolate=extrapolate,
        cutoff=cutoff,
        scale=model.scale,
        compressed=model.compressed,
    )
    # Available model constituents
    c = model.constituents
    # Delta time (TT - UT1)
    # calculating the difference between Terrestrial Time (TT) and UT1 (Universal Time 1),
    deltat = timescale.tt_ut1

    # Calculate complex phase in radians for Euler's
    cph = -1j * ph * np.pi / 180.0

    # Calculate constituent oscillation
    hc = amp * np.exp(cph)

    # Repeat constituents to length of time and number of input
    # coords before passing to `predict_tide_drift`

    # deltat likely represents the time interval between successive data points or time instances.
    # t =  replicating the timescale.tide array n_points times
    # hc = creates an array with the tidal constituents repeated for each time instance
    # Repeat constituents to length of time and number of input
    # coords before passing to `predict_tide_drift`
    t, hc, deltat = (
        np.tile(timescale.tide, n_points),
        hc.repeat(n_times, axis=0),
        np.tile(deltat, n_points),
    )

    # Predict tidal elevations at time and infer minor corrections
    npts = len(t)
    tide = np.ma.zeros((npts), fill_value=np.nan)
    tide.mask = np.any(hc.mask, axis=1)

    # Predict tides
    tide.data[:] = pyTMD.predict.drift(
        t, hc, c, deltat=deltat, corrections=model.format
    )
    minor = pyTMD.predict.infer_minor(t, hc, c, deltat=deltat, corrections=model.format)
    tide.data[:] += minor.data[:]

    # Replace invalid values with fill value
    tide.data[tide.mask] = tide.fill_value

    df = pd.DataFrame(
        {
            "dates": np.tile(time, n_points),
            "x": np.repeat(x, n_times),
            "y": np.repeat(y, n_times),
            "tide": tide,
        }
    )
    df["dates"] = pd.to_datetime(df["dates"], utc=True)
    df.set_index("dates")
    return df

def apply_tidal_corrections_to_shorelines(filtered_shoreline_files, shoreline_path, tidal_corrections_path, output_folder="TIDAL_CORRECTED"):
    """
    Apply tidal corrections to filtered shoreline files using spline fitting and normal vectors.
    
    Parameters:
    -----------
    filtered_shoreline_files : list
        List of filtered shoreline CSV file names
    shoreline_path : str
        Path to the directory containing shoreline files
    tidal_corrections_path : str
        Path to the CSV file containing tidal corrections data
    output_folder : str
        Name of the output folder for tidal corrected shorelines
        
    Returns:
    --------
    list : Paths to the tidal corrected shoreline files
    """
    import numpy as np
    import pandas as pd
    import os
    from geomdl import BSpline
    from geomdl import utilities
    from datetime import datetime
    
    # Load tidal corrections data
    tidal_df = pd.read_csv(tidal_corrections_path)
    print(f"Loaded tidal corrections with {len(tidal_df)} entries")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(shoreline_path), output_folder)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    corrected_files = []
    
    for i, shoreline_file in enumerate(filtered_shoreline_files):
        try:
            print(f"Processing {i+1}/{len(filtered_shoreline_files)}: {shoreline_file}")
            
            # Load shoreline data
            csv_path = os.path.join(shoreline_path, shoreline_file)
            df = pd.read_csv(csv_path)
            
            # Extract coordinates (assuming xm, ym columns from geotransform)
            if 'xm' in df.columns and 'ym' in df.columns:
                x_coords = df['xm'].values
                y_coords = df['ym'].values
            else:
                x_coords = df.iloc[:, 0].values
                y_coords = df.iloc[:, 1].values
            
            # Skip if insufficient points
            if len(x_coords) < 4:
                print(f"  Skipping {shoreline_file}: insufficient points ({len(x_coords)})")
                continue
                
            # Extract date from filename for tidal correction lookup
            # Assuming filename format contains date like: YYYYMMDD...
            try:
                date_str = shoreline_file[:8]  # First 8 characters as date
                file_date = datetime.strptime(date_str, '%Y%m%d')
            except:
                print(f"  Warning: Could not extract date from {shoreline_file}, using zero correction")
                tidal_correction_m = 0.0
            else:
                # Find the closest date
                # Make a copy of tidal_df to avoid modifying the original
                tidal_df_work = tidal_df.copy()
                tidal_df_work['dates'] = pd.to_datetime(tidal_df_work['dates'], errors='coerce')
                tidal_df_work = tidal_df_work.dropna(subset=['dates'])
                
                # Ensure both dates are timezone-naive for comparison
                if tidal_df_work['dates'].dt.tz is not None:
                    tidal_df_work['dates'] = tidal_df_work['dates'].dt.tz_localize(None)
                
                # Convert file_date to pandas timestamp (timezone-naive)
                file_date_ts = pd.Timestamp(file_date)
                
                tidal_df_work['date_diff'] = (tidal_df_work['dates'] - file_date_ts).abs()
                tidal_match = tidal_df_work.loc[tidal_df_work['date_diff'] == tidal_df_work['date_diff'].min()]
                if len(tidal_match) > 0:
                    tidal_correction_m = tidal_match['cross_distance'].iloc[0]
                else:
                    print(f"  Warning: No tidal correction found for {date_str}, using zero")
                    tidal_correction_m = 0.0
            
            print(f"  Tidal correction: {tidal_correction_m:.3f} m")
            
            # Create B-spline curve
            # Prepare points for spline (ensure closed curve)
            points = list(zip(x_coords, y_coords))
            if points[0] != points[-1]:
                points.append(points[0])  # Close the curve
            
            # Create B-spline curve
            curve = BSpline.Curve()
            curve.degree = min(3, len(points) - 1)  # Cubic or lower if insufficient points
            curve.ctrlpts = points
            
            # Generate knot vector
            curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
            
            # Extract 200 evenly spaced points along the curve
            curve.delta = 1.0 / 199  # For 200 points (0 to 1 inclusive)
            spline_points = curve.evalpts
            spline_points = np.array(spline_points)
            
            # Calculate normal vectors at each point
            corrected_points = []
            
            for j in range(len(spline_points)):
                current_point = spline_points[j]
                
                # Calculate tangent vector using neighboring points
                if j == 0:
                    # First point: use forward difference
                    tangent = spline_points[j+1] - current_point
                elif j == len(spline_points) - 1:
                    # Last point: use backward difference
                    tangent = current_point - spline_points[j-1]
                else:
                    # Middle points: use central difference
                    tangent = spline_points[j+1] - spline_points[j-1]
                
                # Calculate normal vector (rotate tangent 90 degrees)
                # For outward normal, we need to determine which direction is "out"
                normal = np.array([-tangent[1], tangent[0]])  # Rotate 90 degrees counterclockwise
                
                # Normalize the normal vector
                normal_length = np.linalg.norm(normal)
                if normal_length > 0:
                    normal = normal / normal_length
                
                # Apply tidal correction by moving point along normal vector
                corrected_point = current_point + normal * tidal_correction_m
                corrected_points.append(corrected_point)
            
            # Save corrected shoreline
            corrected_points = np.array(corrected_points)
            corrected_df = pd.DataFrame({
                'xm': corrected_points[:, 0],
                'ym': corrected_points[:, 1]
            })
            
            # Generate output filename
            output_filename = shoreline_file.replace('.csv', '_tidal_corrected.csv')
            output_path = os.path.join(output_dir, output_filename)
            
            corrected_df.to_csv(output_path, index=False)
            corrected_files.append(output_path)
            
            print(f"  Saved corrected shoreline: {output_filename}")
            
        except Exception as e:
            print(f"  Error processing {shoreline_file}: {e}")
            continue
    
    print(f"\nSuccessfully processed {len(corrected_files)} out of {len(filtered_shoreline_files)} shorelines")
    return corrected_files

######################################################################################################
# Previous Tidal correction functions
######################################################################################################



# def apply_tide_correction_to_set(
#     df: pd.DataFrame, reference_elevation=0, beach_slopes:np.Array
# ):
#     List set_corrections = []
#     for beach_slope in beach_slopes:
#         tc = apply_tide_correction(df, reference_elevation, beach_slope)
#         set_corrections.append(tc)
        
#     return set_corrections
        
        
# def apply_tide_correction(
#     df: pd.DataFrame, reference_elevation=0, beach_slope=0.02
# ):
#     """
#     Applies tidal correction to the timeseries data.

#     Args:
#     - df (DataFrame): Input data with tide predictions and timeseries data.
#     - reference_elevation (float): Reference elevation value.
#     - beach_slope (float): Beach slope value.

#     Returns:
#     - DataFrame: Tidally corrected data.
#     """
#     correction = (df["tide"] - reference_elevation) / beach_slope
#     if "cross_distance" not in df.columns:
#         df["cross_distance"] = 0
#     df["cross_distance"] = df["cross_distance"] + correction
#     return df.drop(columns=["correction"], errors="ignore")


# # using simple time modeling.  set 1 point for littoral system and calculate tide range for relevant times.  
# def model_tides(
#     x,
#     y,
#     time,
#     model="FES2022",
#     directory=None,
#     epsg=4326,
#     method="bilinear",
#     extrapolate=True,
#     cutoff=10.0,
# ):
#     """
#     Compute tides at points and times using tidal harmonics.
#     If multiple x, y points are provided, tides will be
#     computed for all timesteps at each point.

#     This function supports any tidal model supported by
#     `pyTMD`, including the FES2014 Finite Element Solution
#     tide model, and FES2022 Finite Element Solution
#     tide model.

#     This function is a modification of the `pyTMD`
#     package's `compute_tide_corrections` function, adapted
#     to process multiple timesteps for multiple input point
#     locations. For more info:
#     https://pytmd.readthedocs.io/en/stable/user_guide/compute_tide_corrections.html

#     Parameters:
#     -----------
#     x, y : float or list of floats
#         One or more x and y coordinates used to define
#         the location at which to model tides. By default these
#         coordinates should be lat/lon; use `epsg` if they
#         are in a custom coordinate reference system.
#     time : A datetime array or pandas.DatetimeIndex
#         An array containing 'datetime64[ns]' values or a
#         'pandas.DatetimeIndex' providing the times at which to
#         model tides in UTC time.
#     model : string
#         The tide model used to model tides. Options include:
#         - "fes2022b" (only pre-configured option on DEA Sandbox)
#         - "TPXO8-atlas"
#         - "TPXO9-atlas-v5"
#     directory : string
#         The directory containing tide model data files. These
#         data files should be stored in sub-folders for each
#         model that match the structure provided by `pyTMD`:
#         https://pytmd.readthedocs.io/en/latest/getting_started/Getting-Started.html#directories
#         For example:
#         - {directory}/fes2014/ocean_tide/
#           {directory}/fes2014/load_tide/
#     epsg : int
#         Input coordinate system for 'x' and 'y' coordinates.
#         Defaults to 4326 (WGS84).
#     method : string
#         Method used to interpolate tidal contsituents
#         from model files. Options include:
#         - bilinear: quick bilinear interpolation
#         - spline: scipy bivariate spline interpolation
#         - linear, nearest: scipy regular grid interpolations
#     extrapolate : bool
#         Whether to extrapolate tides for locations outside of
#         the tide modelling domain using nearest-neighbor
#     cutoff : int or float
#         Extrapolation cutoff in kilometers. Set to `np.inf`
#         to extrapolate for all points.

#     Returns
#     -------
#     A pandas.DataFrame containing tide heights for all the xy points and their corresponding time
#     """
#     # Check tide directory is accessible
#     if directory is not None:
#         directory = pathlib.Path(directory).expanduser()
#         if not directory.exists():
#             raise FileNotFoundError("Invalid tide directory")
#     # Validate input arguments
#     assert method in ("bilinear", "spline", "linear", "nearest")

#     if "fes2022" in model.lower():
#         model = 'FES2022'
#     # Get parameters for tide model; use custom definition file for
#     model = pyTMD.io.model(directory, format="netcdf", compressed=False).elevation(
#         model
#     )


#     # If time passed as a single Timestamp, convert to datetime64
#     if isinstance(time, pd.Timestamp):
#         time = time.to_datetime64()

#     # Handle numeric or array inputs
#     x = np.atleast_1d(x)
#     y = np.atleast_1d(y)
#     time = np.atleast_1d(time)

#     # Determine point and time counts
#     assert len(x) == len(y), "x and y must be the same length"
#     n_points = len(x)
#     n_times = len(time)

#     # Converting x,y from EPSG to latitude/longitude
#     try:
#         # EPSG projection code string or int
#         crs1 = pyproj.CRS.from_epsg(int(epsg))
#     except (ValueError, pyproj.exceptions.CRSError):
#         # Projection SRS string
#         crs1 = pyproj.CRS.from_string(epsg)

#     # Output coordinate reference system
#     crs2 = pyproj.CRS.from_epsg(4326)
#     transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
#     lon, lat = transformer.transform(x.flatten(), y.flatten())

#     # Convert datetime
#     timescale = pyTMD.time.timescale().from_datetime(time.flatten())
#     n_points = len(x)
#     # number of time points
#     n_times = len(time)

    
#     amp, ph = pyTMD.io.FES.extract_constants(
#         lon,
#         lat,
#         model.model_file,
#         type=model.type,
#         version=model.version,
#         method=method,
#         extrapolate=extrapolate,
#         cutoff=cutoff,
#         scale=model.scale,
#         compressed=model.compressed,
#     )
#     # Available model constituents
#     c = model.constituents
#     # Delta time (TT - UT1)
#     # calculating the difference between Terrestrial Time (TT) and UT1 (Universal Time 1),
#     deltat = timescale.tt_ut1

#     # Calculate complex phase in radians for Euler's
#     cph = -1j * ph * np.pi / 180.0

#     # Calculate constituent oscillation
#     hc = amp * np.exp(cph)

#     # Repeat constituents to length of time and number of input
#     # coords before passing to `predict_tide_drift`

#     # deltat likely represents the time interval between successive data points or time instances.
#     # t =  replicating the timescale.tide array n_points times
#     # hc = creates an array with the tidal constituents repeated for each time instance
#     # Repeat constituents to length of time and number of input
#     # coords before passing to `predict_tide_drift`
#     t, hc, deltat = (
#         np.tile(timescale.tide, n_points),
#         hc.repeat(n_times, axis=0),
#         np.tile(deltat, n_points),
#     )

#     # Predict tidal elevations at time and infer minor corrections
#     npts = len(t)
#     tide = np.ma.zeros((npts), fill_value=np.nan)
#     tide.mask = np.any(hc.mask, axis=1)

#     # Predict tides
#     tide.data[:] = pyTMD.predict.drift(
#         t, hc, c, deltat=deltat, corrections=model.format
#     )
#     minor = pyTMD.predict.infer_minor(t, hc, c, deltat=deltat, corrections=model.format)
#     tide.data[:] += minor.data[:]

#     # Replace invalid values with fill value
#     tide.data[tide.mask] = tide.fill_value

#     df = pd.DataFrame(
#         {
#             "dates": np.tile(time, n_points),
#             "x": np.repeat(x, n_times),
#             "y": np.repeat(y, n_times),
#             "tide": tide,
#         }
#     )
#     df["dates"] = pd.to_datetime(df["dates"], utc=True)
#     df.set_index("dates")
#     return df