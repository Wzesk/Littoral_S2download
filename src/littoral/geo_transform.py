"""
This module is based on similar functions from the coastal project,
authored by Kilian Vos, Water Research Laboratory, University of New South Wales
"""

# load modules
import numpy as np
import pandas as pd
import geopandas as gpd
import skimage.transform as transform
import pyproj 
import os


def batch_geotransform(shoreline_path,cloudless_report_path,coreg_path,debug=False):

    #get the base transform
    cldreport_df = pd.read_csv(cloudless_report_path)
    t_string = get_first_transform(cldreport_df)
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
        
        #transform points using the corrected transform
        try:
            geo_points = convert_pix2world(points,t_values_corrected)
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

def get_first_transform(df):
    #get the whole image_Transform column and then the the first element that is not NaN
    transforms = df['image_Transform'].dropna().tolist()
    if transforms:
        return transforms[0]
    else:
        return None

def parse_transform_string(xform_string):
    # parse the string into arrays of floats separated by \n
    xform_array_strings =  xform_string.split('\n')
    xform_array_strings = np.array([arr.replace('|','') for arr in xform_array_strings])
    xform_values = [np.array([float(val) for val in arr.split(',')]) for arr in xform_array_strings[0:2]]
    xform_values = [np.roll(arr,1) for arr in xform_values]
    xform_values = np.concatenate(xform_values).flatten()
    return xform_values

def get_coreg_correction(name,coreg_table_path):
    df = pd.read_csv(coreg_table_path)
    row = df[df['filename'] == name+".tif"]
    if row.empty:
        return None
    else:
        # return the following columns from that row as a list: shift_x,shift_y,shift_x_meters,shift_y_meters
        return row[['shift_x', 'shift_y', 'shift_x_meters', 'shift_y_meters']].values.flatten().tolist()

def transform_gdf_to_crs(gdf, crs=4326):
    """Convert the GeoDataFrame to the specified CRS."""
    return gdf.to_crs(crs)

def convert_pix2world(points, georef):
    """
    Converts pixel coordinates (pixel row and column) to world projected 
    coordinates performing an affine transformation.
    
    KV WRL 2018

    Arguments:
    -----------
    points: np.array or list of np.array
        array with 2 columns (row first and column second)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
                
    Returns:    
    -----------
    points_converted: np.array or list of np.array 
        converted coordinates, first columns with X and second column with Y
        
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
            tmp = arr[:,[1,0]]
            points_converted.append(tform(tmp))
          
    # if single array
    elif type(points) is np.ndarray:
        tmp = points[:,[1,0]]
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