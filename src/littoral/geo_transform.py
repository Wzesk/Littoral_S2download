"""
Geospatial Transformation Module for Littoral Pipeline

This module converts pixel coordinates from shoreline extraction to geographic coordinates,
with proper handling of superresolution scaling and coregistration corrections.

Key Features:
- Accounts for 4x superresolution scaling from Real-ESRGAN
- Applies coregistration corrections for improved accuracy
- Converts between pixel coordinates and geographic coordinate systems
- Supports batch processing of multiple shoreline files

Based on coastal project functions by Kilian Vos, Water Research Laboratory, 
University of New South Wales, with modifications for littoral pipeline.
"""

# load modules
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import skimage.transform as transform
import pyproj 
import os


def batch_geotransform(shoreline_path, cloudless_report_path, coreg_path, debug=False, scaling=4.0, flip_xy=False):
    """
    Transform pixel coordinates to geographic coordinates for multiple shoreline files.
    
    This function processes all refined shoreline CSV files (*_rl.csv) in a directory,
    applying geospatial transformation with superresolution scaling correction and 
    coregistration adjustments.
    
    Args:
        shoreline_path (str): Directory containing refined shoreline CSV files (*_rl.csv)
        cloudless_report_path (str): Path to cloudless report CSV with geotransform data
        coreg_path (str): Path to coregistration CSV file with shift corrections
        debug (bool, optional): Enable verbose output. Defaults to False.
        scaling (float, optional): Superresolution scaling factor to correct. 
                                 Defaults to 4.0 for Real-ESRGAN 4x upscaling.
        flip_xy (bool, optional): Whether to flip x,y coordinates before transformation.
                                Defaults to True based on testing validation.
    
    Returns:
        None: Creates *_geo.csv files alongside input *_rl.csv files
    """

    #get the base transform
    cldreport_df = pd.read_csv(cloudless_report_path)
    t_string,img_offsets = get_first_transform(cldreport_df,shoreline_path)

    if debug: print("using tranformation: " + t_string)
    t_values = parse_transform_string(t_string)

    #load the refined shoreline csvs all files in the shoreline_path ending with rl.csv
    shoreline_csv_paths = [os.path.join(shoreline_path, f) for f in os.listdir(shoreline_path) if f.endswith('_rl.csv')]
    if debug: print(f"Found {len(shoreline_csv_paths)} shoreline files to process.")

    # if debug randomly pick 5 shorelines to process
    if debug and len(shoreline_csv_paths) > 5:
        np.random.seed(42)
        shoreline_csv_paths = list(np.random.choice(shoreline_csv_paths, size=5, replace=False))
        if debug: print(f"Debug mode: processing a random sample of 5 shorelines.")

    # for each shoreline, get the name from the filename, get the transform, apply it to the shoreline points, save new csv with _geo.csv suffix
    for csv_path in shoreline_csv_paths:
        #get name from filename
        base_name = os.path.basename(csv_path).replace('_nir_rl.csv','')
        if debug: print(f"Processing shoreline: {base_name}")

        #get transform for this image
        coreg_correction = get_coreg_correction(base_name,coreg_path)
        
        # Handle NaN values and None properly
        if coreg_correction is not None:
            coreg_correction_meters = coreg_correction[2:4]
            # Replace any NaN values with 0
            coreg_correction_meters = [0 if pd.isna(val) else val for val in coreg_correction_meters]
        else:
            coreg_correction_meters = [0, 0]
            
        if debug: print(f"Coregistration correction (meters): {coreg_correction_meters}")

        # Create a copy of t_values to avoid cumulative corrections
        t_values_corrected = t_values.copy()
        
        #apply coreg correction to the copied transform
        t_values_corrected[0] += coreg_correction_meters[0]
        t_values_corrected[3] += coreg_correction_meters[1]   
        if debug: print(f"Using transform: {t_values_corrected}")

        #load shoreline points
        df = pd.read_csv(csv_path)
        #first column is x, second is y
        points = df.to_numpy()
        
        if debug: print(f"Loaded {points.shape[0]} shoreline points from {csv_path}")

        if scaling != 1.0:
            if debug: print(f"Applying scaling correction of {scaling}x to geotransform.")
            points = points / scaling
            if debug: print(f"Scaled geotransform: {t_values_corrected}")

        # apply image offset (png to tif)
        points[:,0] += img_offsets[0]
        points[:,1] += img_offsets[1]
        if debug: print(f"Applied image offsets: {img_offsets}")
        
        #transform points
        try:
            # The shoreline CSV files from extract_boundary contain coordinates in (row,col) format
            geo_points = convert_pix2world(points, t_values_corrected, flip_xy=flip_xy)
            if debug: print(f"Transformed {geo_points.shape[0]} shoreline points to geo coordinates.")
            
            # Check if transformation resulted in valid values
            if np.any(np.isnan(geo_points)) or np.any(np.isinf(geo_points)):
                if debug: print(f"Warning: Invalid values in transformed points for {base_name}")
                # Skip saving this file
                continue
                
        except Exception as e:
            if debug: print(f"Error transforming points for {base_name}: {e}")
            continue

        #save new csv with _geo.csv suffix
        geo_df = pd.DataFrame(geo_points,columns=['xm','ym'])
        geo_csv_path = csv_path.replace('_rl.csv','_geo.csv')
        geo_df.to_csv(geo_csv_path,index=False)
        if debug: print(f"Saved georeferenced shoreline to {geo_csv_path}")

def test_transform_tif_bounds(cloudless_report_path):
    """
    input the data from the cloudless report dataframe.  Get the first transform and the size of the first tif image.
    create a rectangle from the imamge bounds in pixel coordinates, transform to world coordinates and return as an np array.
    """
    df = pd.read_csv(cloudless_report_path)
    transforms = df['image_Transform'].dropna().tolist()
    transform = transforms[0]
    #get image dimensions from the dataframe
    image_sizes = df['original_image_size'].dropna().tolist()
    image_dimensions_string = image_sizes[0]
    if image_dimensions_string:
        tif_dimensions = [tuple(map(int, shape.split('x'))) for shape in image_dimensions_string]
        #create an array with the 4 corners of the image in pixel coordinates
        width = tif_dimensions[0][1]
        height = tif_dimensions[0][0]
        pixel_corners = np.array([[0,0],[width,0],[width,height],[0,height],[0,0]])
        print(f"Pixel corners: {pixel_corners}")
        #parse the transform string
        t_values = parse_transform_string(transform)
        #transform the pixel corners to world coordinates
        world_corners = convert_pix2world(pixel_corners, t_values, flip_xy=False)
        return world_corners
    else:
        return None

def get_first_transform(df,shoreline_path):
    """
    Extract the first valid geotransform from cloudless report dataframe.
    
    Args:
        df (pandas.DataFrame): Cloudless report dataframe with image_Transform column
        
    Returns:
        str or None: First non-NaN transform string, or None if none found
    """

    transforms = df['image_Transform'].dropna().tolist()

    #create default x and y image offsets of 0
    img_offsets = [0,0]

    image_dimensions_string = df['original_image_size'].dropna().tolist()
    # covert image string
    if image_dimensions_string:
        tif_dimensions = [tuple(map(int, shape.split('x'))) for shape in image_dimensions_string]

    #get image dimensions from the corresponding RAWRGB folder
    raw_rgb_folder = shoreline_path.replace("SHORELINE","RAWRGB")
    image_files = glob.glob(os.path.join(raw_rgb_folder, '*.png'))
    png_dimensions = []
    if image_files:
        with rasterio.open(image_files[0]) as src:
            png_dimensions = src.shape
    # calculate the image offsets as half the difference between tif and png dimensions
    if tif_dimensions and png_dimensions:
        tif_shape = tif_dimensions[0]  # (height, width)
        png_shape = png_dimensions  # (height, width)
        offset_x = (tif_shape[0] - png_shape[0]) // 2
        offset_y = (tif_shape[1] - png_shape[1]) // 2
        img_offsets = [offset_x, offset_y]

    return transforms[0],img_offsets

def parse_transform_string(xform_string):
    """
    Parse geotransform string into numerical array for affine transformation.
    
    Args:
        xform_string (str): Transform string with format like:
                           "| 10.00, 0.00, 264460.00|\n| 0.00,-10.00, 598380.00|"
                           
    Returns:
        numpy.ndarray: Flattened array of transform values [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
    """
    # Split by newlines and remove pipe characters
    xform_array_strings = xform_string.split('\n')
    xform_array_strings = np.array([arr.replace('|','') for arr in xform_array_strings])
    
    # Convert to float arrays and reorder for affine transformation
    xform_values = [np.array([float(val) for val in arr.split(',')]) for arr in xform_array_strings[0:2]]
    xform_values = [np.roll(arr, 1) for arr in xform_values]  # Reorder for [translation, scale, shear]
    xform_values = np.concatenate(xform_values).flatten()
    
    return xform_values

def get_coreg_correction(name, coreg_table_path):
    """
    Retrieve coregistration correction values for a specific image.
    
    Args:
        name (str): Base filename (without .tif extension)
        coreg_table_path (str): Path to CSV file with coregistration results
        
    Returns:
        list or None: [shift_x, shift_y, shift_x_meters, shift_y_meters] or None if not found
    """
    df = pd.read_csv(coreg_table_path)
    row = df[df['filename'] == name + ".tif"]
    if row.empty:
        return None
    else:
        return row[['shift_x', 'shift_y', 'shift_x_meters', 'shift_y_meters']].values.flatten().tolist()

def transform_gdf_to_crs(gdf, crs=4326):
    """Convert the GeoDataFrame to the specified CRS."""
    return gdf.to_crs(crs)

def convert_pix2world(points, georef, flip_xy=False):
    """
    Converts pixel coordinates to world projected coordinates performing an affine transformation.
    
    KV WRL 2018

    Arguments:
    -----------
    points: np.array or list of np.array
        array with 2 columns. Format depends on flip_xy parameter:
        - If flip_xy=False: expects (x, y) format 
        - If flip_xy=True: expects (row, col) format and will flip to (x, y)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
    flip_xy: bool, optional
        If True, flips input coordinates from (row, col) to (x, y) before transformation.
        If False, assumes input is already in (x, y) format. Default: False.
                
    Returns:    
    -----------
    points_converted: np.array or list of np.array 
        converted coordinates, first column with X and second column with Y
        
    """
    
    # make affine transformation matrix
    aff_mat = np.array([[georef[1], georef[2], georef[0]],
                       [georef[4], georef[5], georef[3]],
                       [0, 0, 1]])
    # create affine transformation
    tform = transform.AffineTransform(aff_mat)

    # if list of arrays
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            if flip_xy:
                tmp = arr[:,[1,0]]  # Flip from (row, col) to (x, y)
            else:
                tmp = arr  # Use coordinates as-is (already in x, y format)
            points_converted.append(tform(tmp))
          
    # if single array
    elif type(points) is np.ndarray:
        if flip_xy:
            tmp = points[:,[1,0]]  # Flip from (row, col) to (x, y)  
        else:
            tmp = points  # Use coordinates as-is (already in x, y format)
        points_converted = tform(tmp)
        
    else:
        raise Exception('invalid input type')
        
    return points_converted

def convert_world2pix(points, georef):
    """
    Converts world projected coordinates (X,Y) to image coordinates 
    (pixel row and column) performing an affine transformation.
    
    KV WRL 2018

    Arguments:
    -----------
    points: np.array or list of np.array
        array with 2 columns (X,Y)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
                
    Returns:    
    -----------
    points_converted: np.array or list of np.array 
        converted coordinates (pixel row and column)
    
    """
    
    # make affine transformation matrix
    aff_mat = np.array([[georef[1], georef[2], georef[0]],
                       [georef[4], georef[5], georef[3]],
                       [0, 0, 1]])
    # create affine transformation
    tform = transform.AffineTransform(aff_mat)
    
    # if list of arrays
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            points_converted.append(tform.inverse(points))
            
    # if single array    
    elif type(points) is np.ndarray:
        points_converted = tform.inverse(points)
        
    else:
        print('invalid input type')
        raise
        
    return points_converted

def convert_epsg(points, epsg_in, epsg_out):
    """
    Converts from one spatial reference to another using the epsg codes
    
    KV WRL 2018

    Arguments:
    -----------
    points: np.array or list of np.ndarray
        array with 2 columns (rows first and columns second)
    epsg_in: int
        epsg code of the spatial reference in which the input is
    epsg_out: int
        epsg code of the spatial reference in which the output will be            
                
    Returns:    
    -----------
    points_converted: np.array or list of np.array 
        converted coordinates from epsg_in to epsg_out
        
    """
    
    # define transformer
    proj = pyproj.Transformer.from_crs(epsg_in, epsg_out, always_xy=True)
    
    # transform points
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            x,y = proj.transform(arr[:,0], arr[:,1])
            arr_converted = np.transpose(np.array([x,y]))
            points_converted.append(arr_converted)
    elif type(points) is np.ndarray:
        x,y = proj.transform(points[:,0], points[:,1])
        points_converted = np.transpose(np.array([x,y]))
    else:
        raise Exception('invalid input type')

    return points_converted