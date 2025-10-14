"""
Sentinel-2 Data Extraction Module
================================

This module handles the extraction and processing of Sentinel-2 satellite imagery
using Google Earth Engine. It provides functionality for downloading imagery,
cloud masking, and initial processing for Littoral's shoreline analysis pipeline.

Functions
---------
connect : Initialize Earth Engine connection
get_image_collection : Retrieve filtered Sentinel-2 collection
retrieve_rgb_nir_from_collection : Extract RGB+NIR bands with cloud masking
process_collection_images : Batch process multiple images

Notes
-----
This module is part of Littoral's modular processing pipeline, specifically
handling the initial data extraction and preparation stage.
"""

import datetime
import io
import os
from typing import Dict, Tuple

import ee
import numpy as np
import pandas as pd
import requests
from omnicloudmask import predict_from_array
from PIL import Image
from skimage.morphology import binary_dilation, disk

from . import littoral_sites, tario


def connect(project_id: str = "useful-theory-442820-q8") -> None:
    """Initialize connection to Google Earth Engine.

    Parameters
    ----------
    project_id : str, optional
        Google Cloud project ID for Earth Engine authentication.
        Default is 'useful-theory-442820-q8'.

    Raises
    ------
    Exception
        If initial authentication fails, will attempt to re-authenticate.

    Examples
    --------
    >>> connect()  # Use default project
    >>> connect('my-project-id')  # Use specific project
    """
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)


def get_filtered_image_collection(
    data: Dict,
    valid_pixel_threshold: float = 75.0
) -> ee.ImageCollection:
    """Retrieve filtered Sentinel-2 image collection.

    This function filters for date, location, and cloud cover, and also
    removes images that are mostly black (zero values) by calculating the
    percentage of valid pixels within the AOI.

    Parameters
    ----------
    data : dict
        Configuration dictionary containing:
        - aoi : list
            Area of interest coordinates [x1, y1, x2, y2]
        - start_date : str
            Start date for image collection
        - end_date : str
            End date for image collection
        - cloudy_pixel_percentage : int
            Maximum cloud coverage percentage
    valid_pixel_threshold : float, optional
        The minimum percentage of valid (non-zero) pixels an image must have
        within the AOI to be included, by default 95.0.

    Returns
    -------
    ee.ImageCollection
        Filtered collection of Sentinel-2 images

    Examples
    --------
    >>> ee.Initialize() # Make sure to initialize the library.
    >>> data = {
    ...     "aoi": [-122.5, 37.7, -122.4, 37.8],
    ...     "start_date": "2023-01-01",
    ...     "end_date": "2023-12-31",
    ...     "max_cloudy_pixel_percentage": 10
    ... }
    >>> collection = get_image_collection(data, valid_pixel_threshold=90)
    """
    aoi_rec = ee.Geometry.Rectangle(data["aoi"])

    # Initial filtering based on metadata
    se2_col = (
        ee.ImageCollection("COPERNICUS/S2")
        .filterDate(data["start_date"], data["end_date"])
        .filterBounds(aoi_rec)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", data["max_cloudy_pixel_percentage"]))
    )

    # Function to calculate the percentage of valid pixels for an image
    def calculate_valid_pixels(image):
        # Select a band (e.g., B2) and create a mask of non-zero pixels.
        # unmask(0) replaces masked pixels with 0, then gt(0) creates a
        # binary mask where valid pixels are 1 and zero/masked pixels are 0.
        valid_mask = image.select('B2').unmask(0).gt(0)

        # Use reduceRegion to calculate the percentage of valid pixels.
        stats = valid_mask.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi_rec,
            scale=100,  # Use a coarser scale for efficiency
            maxPixels=1e9
        )
        # The mean of a binary mask is the proportion of 1s.
        # Multiply by 100 to get a percentage.
        valid_percentage = ee.Number(stats.get('B2')).multiply(100)
        return image.set('valid_percentage', valid_percentage)

    # Map the function over the collection and filter by the new property
    filtered_col = se2_col.map(calculate_valid_pixels).filter(
        ee.Filter.gte('valid_percentage', valid_pixel_threshold)
    )

    return filtered_col


def get_image_collection(
    data: Dict
) -> ee.ImageCollection:
    """Retrieve filtered Sentinel-2 image collection.

    Parameters
    ----------
    data : dict
        Configuration dictionary containing:
        - aoi : list
            Area of interest coordinates [x1, y1, x2, y2]
        - start_date : str
            Start date for image collection
        - end_date : str
            End date for image collection
        - cloudy_pixel_percentage : int
            Maximum cloud coverage percentage

    Returns
    -------
    ee.ImageCollection
        Filtered collection of Sentinel-2 images

    Examples
    --------
    >>> data = {
    ...     "aoi": [-122.5, 37.7, -122.4, 37.8],
    ...     "start_date": "2023-01-01",
    ...     "end_date": "2023-12-31",
    ...     "max_cloudy_pixel_percentage": "10"
    ... }
    >>> collection = get_image_collection(data)
    """
    aoi_rec = ee.Geometry.Rectangle(data["aoi"])
    se2_col = (
        ee.ImageCollection("COPERNICUS/S2")
        .filterDate(data["start_date"], data["end_date"])
        .filterBounds(aoi_rec)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", data["max_cloudy_pixel_percentage"]))
    )
    return se2_col


def visualize_nir_from_collection(
    data: Dict, se2_col: ee.ImageCollection, index: int
) -> ee.Image:
    """Visualize NIR band from image collection.

    Parameters
    ----------
    data : dict
        Configuration dictionary with AOI
    se2_col : ee.ImageCollection
        Sentinel-2 image collection
    index : int
        Index of image in collection to visualize

    Returns
    -------
    ee.Image
        Image with NIR bands selected and scaled

    Notes
    -----
    Uses bands B4 (Red), B3 (Green), and B8 (NIR)
    """
    aoi = ee.Geometry.Rectangle(data["aoi"])
    rg_nir = ["B8"]
    rg_nir_img = (
        ee.Image(se2_col.toList(se2_col.size()).get(index)).select(rg_nir).clip(aoi)
    )
    rg_nir_img_disp = rg_nir_img.divide(10000)
    return rg_nir_img_disp


def visualize_rgb_from_collection(
    data: Dict, se2_col: ee.ImageCollection, index: int
) -> ee.Image:
    """Visualize rgb bands from image collection.

    Parameters
    ----------
    data : dict
        Configuration dictionary with AOI
    se2_col : ee.ImageCollection
        Sentinel-2 image collection
    index : int
        Index of image in collection to visualize

    Returns
    -------
    ee.Image
        Image with rgb bands selected and scaled

    Notes
    -----
    Uses bands B4 (Red), B3 (Green), and B2 (blue)
    """
    aoi = ee.Geometry.Rectangle(data["aoi"])
    rg_nir = ["B4", "B3", "B2"] 
    rg_nir_img = (
        ee.Image(se2_col.toList(se2_col.size()).get(index)).select(rg_nir).clip(aoi)
    )
    rg_nir_img_disp = rg_nir_img.divide(10000)
    return rg_nir_img_disp


def rgnir_cldmask_from_collection(
    data: Dict, se2_col: ee.ImageCollection, index: int
) -> Tuple[np.ndarray, np.ndarray, float, Dict]:
    """Retrieve RG+NIR bands with cloud masking.

    Parameters
    ----------
    data : dict
        Configuration dictionary with AOI
    se2_col : ee.ImageCollection
        Sentinel-2 image collection
    index : int
        Index of image in collection

    Returns
    -------
    tuple
        - np.ndarray : NIR band data
        - np.ndarray : Cloud mask
        - float : Percentage of usable pixels
        - dict : Image projection information

    Notes
    -----
    Uses omnicloudmask for cloud detection and masking
    """
    aoi = ee.Geometry.Rectangle(data["aoi"])
    rg_nir = ["B4", "B3", "B8"] 
    rg_nir_img = ee.Image(se2_col.toList(se2_col.size()).get(index))
    url = rg_nir_img.getDownloadUrl(
        {"bands": rg_nir, "region": aoi, "scale": 10, "format": "NPY"}
    )
    projection = rg_nir_img.select("B8").projection().getInfo()
    response = requests.get(url)
    img_data = np.load(io.BytesIO(response.content))

    # stack 3 2d arrays into 3d array
    img_arr = np.dstack([img_data["B4"], img_data["B3"], img_data["B8"]])
    # move 3 dimension to first position
    img_arr = np.moveaxis(img_arr, -1, 0)
    # Predict cloud and cloud shadow masks
    pred_mask = predict_from_array(img_arr)

    # merge cloud mask with cloud shadow mask
    pred_array = np.where(pred_mask == 0, 1, 0) + np.where(pred_mask == 2, 1, 0)
    pred_array = np.where(pred_array > 0, 1, 0)
    cld_pred = np.squeeze(pred_array)

    usable_pixels = np.sum(cld_pred) / cld_pred.size

    return img_data["B8"], cld_pred, usable_pixels, projection


def retrieve_rgb_nir_from_collection(data, se2_col,index):
    """Retrieve RGB+NIR bands as two images

    Parameters
    ----------
    data : dict
        Configuration dictionary with AOI
    se2_col : ee.ImageCollection
        Sentinel-2 image collection
    index : int
        Index of image in collection

    Returns
    -------
    tuple
        - np.ndarray : rgb bands
        - np.ndarray : nir bands
    """

    aoi = ee.Geometry.Rectangle(data["aoi"])    

    rgb_nir = ['B4','B3','B2','B8'] # blue is 'B2'
    rg_nir_img = ee.Image(se2_col.toList(se2_col.size()).get(index))#.select(rg_nir)
    url = rg_nir_img.getDownloadUrl({
        'bands': rgb_nir,
        'region': aoi,
        'scale': 10,
        'format': 'NPY'
    })

    response = requests.get(url)
    data = np.load(io.BytesIO(response.content))

    # stack 3 2d arrays into 3d array
    rgb_img_arr = np.dstack([data['B4'], data['B3'], data['B2']])
    nir_img_arr  = np.dstack([data['B8'], data['B8'], data['B8']])

    return rgb_img_arr , nir_img_arr 


def dilate_mask(mask: np.ndarray, size_disk: int = 9) -> Image.Image:
    """Dilate binary mask using disk structuring element.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask to dilate
    size_disk : int, optional
        Radius of disk structuring element, by default 9

    Returns
    -------
    PIL.Image
        Dilated mask as PIL Image
    """
    mask = binary_dilation(mask, disk(size_disk))
    mask_image = Image.fromarray(mask)
    return mask_image


def process_collection_images_tofiles(data, se2_col, max_images=-1):
    """Batch process multiple images from collection.

    Parameters
    ----------
    data : dict
        Configuration dictionary containing:
        - aoi : list
            Area of interest coordinates
        - project_name : str
            Name of the project
        - path : str
            Output directory path
        - usable_pixel_percentage : float
            Minimum percentage of usable pixels
    se2_col : ee.ImageCollection
        Sentinel-2 image collection
    max_images : optional cap on the number of imagers to process (default is -1 which means no cap)

    Returns
    -------
    pd.DataFrame
        Processing results and status tracking

    Notes
    -----
    Saves processed images to files.
    """
    
    #first get the ae mask
    ae_path,cluster_data = detect_islands_from_embeddings(data["aoi"], year=2024, n_samples=1000, n_clusters=2, folder_path=data["path"])

    name = data["project_name"]
    path = data["path"]
    threshold = 1

    #set a max of images per site
    length = se2_col.size().getInfo()
    if max_images > 0:
        if length > max_images:
            length = max_images

    #if path does not exist, creat it
    if not os.path.exists(path):
        os.makedirs(path)

    #create a new pandas dataframe with the columns: Index, name, status,usable_percentage
    proj_track = pd.DataFrame(columns=['Index', 'name','rgb path','nir path'])

    #get landsat for coregistration reference
    l9_col = get_landsat_coreg(data)
    print(f"landsat count = {l9_col.size().getInfo()}")

    # if subfolders do not exist, create them
    if not os.path.exists(path + "/REFERENCE"):
        os.makedirs(path + "/REFERENCE")
    # if target folder does not exist, create it
    if not os.path.exists(path + "/TARGETS"):
        os.makedirs(path + "/TARGETS")
    # if rawrgb folder does not exist, create it
    if not os.path.exists(path + "/RAWRGB"):
        os.makedirs(path + "/RAWRGB")
    # if rawnir folder does not exist, create it
    if not os.path.exists(path + "/RAWNIR"):
        os.makedirs(path + "/RAWNIR")

    retrieve_tiff_from_collection(data, l9_col, 0, path + "/REFERENCE")
    
    for i in range(length):
        #check if image is all zeros
        img = ee.Image(se2_col.toList(se2_col.size()).get(i))
        stats = img.reduceRegion(reducer=ee.Reducer.anyNonZero(), geometry=ee.Geometry.Rectangle(data['aoi']), scale=10)
        has_data = stats.get('B4')
        if has_data == 0:
            print(f"Image {i} is all zeros.")
            continue

        #get two arrays from each s2 image, one for rgb and one for nir
        rgb,nir = retrieve_rgb_nir_from_collection(data, se2_col, i)
        #get a tiff with all bands to calculate coregistration offsets
        retrieve_tiff_from_collection(data, se2_col,i, path + "/TARGETS")

        #get the name of the i image file from the collection
        img_name = ee.Image(se2_col.toList(se2_col.size()).get(i)).get('system:index').getInfo()

        #normalize
        n_rgb = rgb/np.max(rgb)
        
        #convert into an image
        rgb_img = Image.fromarray((n_rgb * 255).astype(np.uint8))

        #normalize
        n_nir = nir/np.max(nir)
        #convert into an image
        nir_img = Image.fromarray((n_nir * 255).astype(np.uint8))

        #save images to png
        rgb_path = path + "/RAWRGB/" + img_name + "_rgb.png"
        rgb_img.save(rgb_path)
        nir_path = path + "/RAWNIR/" + img_name + "_nir.png"
        nir_img.save(nir_path)

        # get cloud cover percentages for collection images
        cloudy_pixel_percentage = ee.Image(se2_col.toList(se2_col.size()).get(i)).get('CLOUDY_PIXEL_PERCENTAGE').getInfo()

        # add row to table
        new_row = pd.DataFrame({'Index': [i], 'name': [img_name], 'rgb path': [rgb_path], 'nir path': [nir_path], 'reference_path': [path + "/REFERENCE"], 'target_path': [path + "/TARGETS"], 'cloudy_pixel_percentage': [cloudy_pixel_percentage]})
        proj_track = pd.concat([proj_track, new_row], ignore_index=True)
        
        
        #add coregistration settings
        coreg_settings = {
          "coregister_settings": {
            "ws": [256, 256],
            "nodata": [0, 0],
            "max_shift": 100,
            "binary_ws": False,
            "progress": False,
            "v": False,
            "ignore_errors": True,
            "fmt_out": "GTiff"
          },
          "filtering_settings": {
            "shift_reliability": 40,
            "window_size": 50,
            "max_shift_meters": 250,
            "filter_z_score": True,
            "filter_z_score_filter_passed_only": False,
            "z_score_threshold": 2
          }
        }
        import json
        with open(path + "/coreg_settings.json", "w") as f:
            json.dump(coreg_settings, f, indent=2)

    return proj_track


def get_cloud_cover_percentages_fom_collection(data, se2_col):
    """Get cloud cover percentages from image collection.

    Parameters
    ----------
    data : dict
        Configuration dictionary containing:
        - aoi : list
            Area of interest coordinates
        - project_name : str
            Name of the project
        - path : str
            Output directory path
        - usable_pixel_percentage : float
            Minimum percentage of usable pixels
    se2_col : ee.ImageCollection
        Sentinel-2 image collection

    Returns
    -------
    pd.DataFrame
        DataFrame with image names and their cloud cover percentages

    Notes
    -----
    Saves results to CSV file.
    """
    name = data["project_name"]
    path = data["path"]
    length = se2_col.size().getInfo()

    # if path does not exist, create it
    if not os.path.exists(path):
        os.makedirs(path)
    
    # create tracking dataframe
    columns = [
        "Index",
        "name",
        "cloudy_pixel_percentage"
    ]
    cloud_cover = pd.DataFrame(columns=columns)

    for i in range(length):
        # get image name from collection
        img_name = (
            ee.Image(se2_col.toList(se2_col.size()).get(i))
            .get("system:index")
            .getInfo()
        )
        cloudy_pixel_percentage = (
            ee.Image(se2_col.toList(se2_col.size()).get(i))
            .get("CLOUDY_PIXEL_PERCENTAGE")
            .getInfo()
        )
        new_row = pd.DataFrame(
            {
                "Index": [i],
                "name": [img_name],
                "cloudy_pixel_percentage": [cloudy_pixel_percentage]
            }
        )

        # add row to table
        cloud_cover = pd.concat([cloud_cover, new_row], ignore_index=True)

    return cloud_cover

def process_collection_images_totar(data: Dict, se2_col: ee.ImageCollection,site_path='') -> pd.DataFrame:
    """Batch process multiple images from collection.

    Parameters
    ----------
    data : dict
        Configuration dictionary containing:
        - aoi : list
            Area of interest coordinates
        - project_name : str
            Name of the project
        - path : str
            Output directory path
        - usable_pixel_percentage : float
            Minimum percentage of usable pixels
    se2_col : ee.ImageCollection
        Sentinel-2 image collection

    Returns
    -------
    pd.DataFrame
        Processing results and status tracking

    Notes
    -----
    Saves processed images to TAR archives and updates processing status in CSV.
    """
    name = data["project_name"]
    path = data["path"]
    max_cloudy_pixel_percentage = data["max_cloudy_pixel_percentage"]
    #transform max_cloudy_pixel_percentage to a minimum threshold of usable pixels
    threshold = (100 - max_cloudy_pixel_percentage) / 100
    length = se2_col.size().getInfo()

    # if path does not exist, create it
    if not os.path.exists(path):
        os.makedirs(path)
    rgnir_path = path + "/rgnir.tar"
    cldmsk_path = path + "/cldmsk.tar"

    # create tracking dataframe
    columns = [
        "Index",
        "name",
        "rgnir_download",
        "cld_calculation",
        "percentage_usable",
    ]
    proj_track = pd.DataFrame(columns=columns)

    for i in range(length):
        nir, cld, percentage_usable, projection = rgnir_cldmask_from_collection(
            data, se2_col, i
        )
        print(f"percentage usable pixels = {percentage_usable}")

        # get image name from collection
        img_name = (
            ee.Image(se2_col.toList(se2_col.size()).get(i))
            .get("system:index")
            .getInfo()
        )
        new_row = pd.DataFrame(
            {
                "Index": [i],
                "name": [img_name],
                "rgnir_download": [False],
                "cld_calculation": [False],
                "percentage_usable": [percentage_usable],
            }
        )

        if percentage_usable > threshold:
            # normalize
            n_nir = (nir - np.min(nir)) / (np.max(nir) - np.min(nir))
            # stack
            nir_3d = np.dstack((n_nir, n_nir, n_nir))
            nir_img = Image.fromarray((nir_3d * 255).astype(np.uint8))
            # normalize
            cld = cld / np.max(cld)
            cld_3d = np.dstack((cld, cld, cld))
            cld_img = Image.fromarray((cld_3d * 255).astype(np.uint8))
            rgnirtar = tario.tar_io(rgnir_path)
            rgnirtar.save_to_tar(nir_img, img_name + "_rgnir.png", overwrite=True)
            new_row["rgnir_download"] = True
            cldtar = tario.tar_io(cldmsk_path)
            cldtar.save_to_tar(cld_img, img_name + "_cld.png", overwrite=True)
            new_row["cld_calculation"] = True

            print(f"{img_name} saved")
        else:
            print(f"{i} has too many clouds")

        # add row to table
        proj_track = pd.concat([proj_track, new_row], ignore_index=True)

    # save proj_track to csv
    proj_track.to_csv(path + "/" + "proj_track.csv", index=False)

    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
    if len(site_path) < 3:
        site_path = path.replace(name,"littoral_sites.csv")
        
    littoral_sites.set_last_run(name, date_str,site_path)
    print(f"finished run: {name} {date_str}")

    return proj_track


def retrieve_tiff_from_collection(data, se2_col,index,folder_path):

  aoi = ee.Geometry.Rectangle(data["aoi"])

  # expand the aoi by 2600m to get more pixels to co-register
  aoi_rec = aoi.buffer(2600)

  tiff_collection = ee.Image(se2_col.toList(se2_col.size()).get(index))
 
  # Dynamically get the list of all band names from that image
  all_bands = tiff_collection.bandNames()
  #print('All available bands:', all_bands.getInfo())

  coreg_bands = all_bands.slice(1, 10)
  coreg_band_names = coreg_bands.getInfo()
  #print('Bands selected for download:', coreg_band_names)
    
  url = tiff_collection.getDownloadUrl({
    'bands':coreg_band_names,
    'region': aoi_rec,
    'scale': 10,
    'format': 'GEO_TIFF'
  })

  response = requests.get(url)

  if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
  img_name = ee.Image(se2_col.toList(se2_col.size()).get(index)).get('system:index').getInfo()

  tiff_name = img_name + ".tif"
  tiff_save_path = os.path.join(folder_path, tiff_name)

  with open(tiff_save_path, 'wb') as fd:
    fd.write(response.content)

  return data


def get_landsat_coreg(data):
    aoi = ee.Geometry.Rectangle(data["aoi"])
    
    # expand the aoi by 2600m to get more pixels to co-register
    aoi_rec = aoi.buffer(2600)

    landsat_col = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterDate("2021-01-01",data["end_date"]).filterBounds(aoi_rec)
    landsat_sorted = landsat_col.sort('CLOUD_COVER')

    return landsat_sorted


def process_cloud_imputed_images(existing_nir_folder, existing_rgb_folder, clear_tiff_folder, clear_output_folder):
    """
    Process NIR and RGB images by replacing them with cloud-imputed versions from TIFF files.
    
    Parameters:
    -----------
    existing_nir_folder : str
        Path to folder containing original NIR PNG images
    existing_rgb_folder : str
        Path to folder containing original RGB PNG images
    clear_tiff_folder : str
        Path to folder containing cloudless TIFF files
    clear_output_folder : str
        Path to output folder for processed PNG images
        
    Returns:
    --------
    list : List of successfully processed output file paths
    """
    import glob
    from PIL import Image
    import rasterio
    import numpy as np
    import os
    
    # Create the output folder if it doesn't exist
    os.makedirs(clear_output_folder, exist_ok=True)
    
    # Find all NIR and RGB images in the folder
    nir_images = glob.glob(os.path.join(existing_nir_folder, '*nir.png'))
    rgb_images = glob.glob(os.path.join(existing_rgb_folder, '*rgb.png'))
    
    processed_files = []
    
    for img_path, band_type in [(p, 'nir') for p in nir_images] + [(p, 'rgb') for p in rgb_images]:
        try:
            # 1. Get the image size
            with Image.open(img_path) as img:
                width, height = img.size

            # 2. Find corresponding tiff
            base_name = os.path.basename(img_path).replace(f'_{band_type}.png', '')
            tiff_pattern = os.path.join(clear_tiff_folder, f"{base_name}_pred.tif")
            tiff_files = glob.glob(tiff_pattern)
            if not tiff_files:
                print(f"No TIFF found for {img_path}")
                continue
            tiff_path = tiff_files[0]

            # 3. Get bands and create 3-band PNG
            with rasterio.open(tiff_path) as src:
                if band_type == 'nir':
                    # Sentinel: NIR is band B08 if available, else band 4 if present
                    band_names = src.descriptions if hasattr(src, 'descriptions') else None
                    nir_band = None
                    if band_names and 'B08' in band_names:
                        b08_index = band_names.index('B08') + 1  # rasterio bands are 1-based
                        nir_band = src.read(b08_index)
                        print(f"Using B08 (band {b08_index}) for NIR: {tiff_path}")
                    elif src.count >= 6:
                        nir_band = src.read(6)
                        print(f"Using band 8 for NIR: {tiff_path}")
                    else:
                        print(f"ERROR: No NIR band (B08 or band 4) found in {tiff_path}. Skipping.")
                        continue
                    print(f"NIR band stats for {tiff_path}: min={nir_band.min()}, max={nir_band.max()}, mean={nir_band.mean()}")
                    nir_band = np.nan_to_num(nir_band, nan=0)
                    nir_band = np.clip(nir_band, 0, 65535)
                    if np.ptp(nir_band) == 0:
                        img_arr = np.zeros((nir_band.shape[0], nir_band.shape[1], 3), dtype=np.uint8)
                    else:
                        nir_band = ((nir_band - nir_band.min()) / (np.ptp(nir_band) + 1e-6) * 255).astype(np.uint8)
                        img_arr = np.stack([nir_band]*3, axis=-1)
                else:
                    # RGB is bands 3, 2, 1 (Sentinel/Landsat convention)
                    r = src.read(3) if src.count >= 3 else src.read(1)
                    g = src.read(2) if src.count >= 2 else src.read(1)
                    b = src.read(1)
                    rgb_stack = np.stack([r, g, b], axis=-1)
                    rgb_stack = np.nan_to_num(rgb_stack, nan=0)
                    rgb_stack = np.clip(rgb_stack, 0, 65535)
                    img_arr = np.zeros_like(rgb_stack, dtype=np.uint8)
                    for i in range(3):
                        band = rgb_stack[..., i]
                        if np.ptp(band) == 0:
                            img_arr[..., i] = np.zeros_like(band, dtype=np.uint8)
                        else:
                            img_arr[..., i] = ((band - band.min()) / (np.ptp(band) + 1e-6) * 255).astype(np.uint8)

            # 4. Crop to original size, centered
            h, w = img_arr.shape[:2]
            left = max((w - width) // 2, 0)
            top = max((h - height) // 2, 0)
            cropped = img_arr[top:top+height, left:left+width]

            # 5. Save as PNG
            out_path = os.path.join(clear_output_folder, os.path.basename(img_path))
            Image.fromarray(cropped).save(out_path)
            print(f"Saved: {out_path}")
            processed_files.append(out_path)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print(f"Successfully processed {len(processed_files)} images")
    return processed_files

##### using Alpha Earth embeddings to detect islands #####

def detect_islands_from_embeddings(aoi, year=2024, n_samples=1000, n_clusters=2, folder_path=""):
    """
    Detect islands using Google Earth Engine satellite embeddings.
    
    This function replicates the functionality from the alpha_earth notebook
    to identify islands within an AOI using unsupervised clustering on 
    satellite embedding features.
    
    Parameters
    ----------
    aoi : list
        Area of interest coordinates [x1, y1, x2, y2] in WGS84
    year : int, optional
        Year for embedding data, by default 2024
    n_samples : int, optional
        Number of samples for training the clusterer, by default 1000
    n_clusters : int, optional
        Number of clusters for k-means, by default 2
        
    Returns
    -------
    tuple
        - ee.Image : Binary mask where 1 = island pixels, 0 = water pixels
        - ee.Image : Full clustering result with cluster IDs
        - dict : Cluster information including pixel counts
        
    Examples
    --------
    >>> aoi = [72.9201, 5.628, 72.9259, 5.6325]  # Maldives coordinates
    >>> island_mask, clusters, info = detect_islands_from_embeddings(aoi)
    >>> # Use island_mask for further analysis or visualization
    
    Notes
    -----
    This function assumes the cluster with fewer pixels represents islands,
    while the cluster with more pixels represents water. This heuristic
    works well for small island detection in oceanic environments.
    """
    # Convert AOI to Earth Engine geometry
      # expand the aoi by 1000m to get more pixels to co-register
    aoi_rec = ee.Geometry.Rectangle(aoi)
    geometry = aoi_rec.buffer(1000)

    # Access the Satellite Embedding Dataset
    embeddings = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
    
    # Filter embeddings by date and bounds
    start_date = ee.Date.fromYMD(year, 1, 1)
    end_date = start_date.advance(1, 'year')
    
    filtered_embeddings = embeddings.filter(
        ee.Filter.date(start_date, end_date)
    ).filter(
        ee.Filter.bounds(geometry)
    )
    
    # Create mosaic of embeddings
    embeddings_image = filtered_embeddings.mosaic()
    
    # Create training dataset for unsupervised clustering
    training = embeddings_image.sample(
        region=geometry,
        scale=10,
        numPixels=n_samples,
        seed=100
    )
    
    # Train k-means clusterer
    clusterer = ee.Clusterer.wekaKMeans(n_clusters).train(training)
    
    # Apply clustering to the image
    clustered = embeddings_image.cluster(clusterer)
    
    # Calculate pixel counts for each cluster to determine which is land vs water
    cluster_counts = {}
    cluster_masks = {}
    
    for cluster_id in range(n_clusters):
        # Create mask for this cluster
        cluster_mask = clustered.eq(cluster_id)
        cluster_masks[cluster_id] = cluster_mask
        
        # Count pixels in this cluster
        pixel_count = cluster_mask.selfMask().reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=geometry,
            scale=10,
            maxPixels=1e9
        )
        
        cluster_counts[cluster_id] = pixel_count
    
    # For a 2-cluster scenario, use a simple heuristic:
    # Assume cluster 1 represents islands (smaller areas) and cluster 0 represents water
    # This is a reasonable assumption for oceanic environments with small islands
    island_cluster_id = 1
    water_cluster_id = 0
    
    # Create binary island mask
    island_mask = clustered.eq(island_cluster_id)

    # check for disconnected islands and only keep the one whose centroid is closest to the AOI centroid
    island_vectors = extract_island_vectors(island_mask, geometry)
    island_list = island_vectors.toList(island_vectors.size())
    if island_vectors.size().getInfo() > 1:
        print(f"Found {island_vectors.size().getInfo()} disconnected islands, keeping the closest to AOI centroid")
        aoi_centroid = geometry.centroid(maxError=1).coordinates().getInfo()
        min_dist = float('inf')
        closest_island = None
        for i in range(island_vectors.size().getInfo()):
            island = ee.Feature(island_list.get(i))
            island_centroid = island.geometry().centroid(maxError=1).coordinates().getInfo()
            dist = ((island_centroid[0] - aoi_centroid[0])**2 + (island_centroid[1] - aoi_centroid[1])**2)**0.5
            if dist < min_dist:
                min_dist = dist
                closest_island = island
        # Create a mask from the closest island
        island_mask = ee.Image(0).toByte().paint(closest_island.geometry(), 1)
        print("Kept the closest island to AOI centroid")
    else:
        print("Only one island detected, no need to filter")
    
    # Create cluster information dictionary
    cluster_info = {
        'cluster_counts': cluster_counts,
        'island_cluster_id': island_cluster_id,
        'water_cluster_id': water_cluster_id,
        'geometry': geometry,
        'scale': 10
    }

    #create a AE folder from the folder path
    AE_folder = os.path.join(folder_path, "AE")
    if not os.path.exists(AE_folder):
        os.makedirs(AE_folder)  

    #save the mask as a png. 
    island_mask_vis = island_mask.visualize(min=0, max=1, palette=['blue', 'green'])
    url = island_mask_vis.getDownloadUrl({
        'region': geometry,#aoi_rec, #geometry is the buffered AOI aoi_rec is the original.
        'scale': 10,
        'format': 'png'
    })
    response = requests.get(url)
    island_mask_image = Image.open(io.BytesIO(response.content))
    island_mask_path = os.path.join(AE_folder, "island_mask.png")
    island_mask_image.save(island_mask_path)
    print(f"Island mask saved to {island_mask_path}")
    
    return island_mask_path, cluster_info


def extract_island_vectors(island_mask, geometry, min_area=100, scale=10):
    """
    Convert island mask to vector polygons with area filtering.
    
    Parameters
    ----------
    island_mask : ee.Image
        Binary mask where 1 = island pixels
    geometry : ee.Geometry
        Area of interest geometry
    min_area : float, optional
        Minimum island area in square meters, by default 100
    scale : int, optional
        Processing scale in meters, by default 10
        
    Returns
    -------
    ee.FeatureCollection
        Collection of island polygons with area attributes
    """
    # Convert mask to vectors
    island_vectors = island_mask.selfMask().reduceToVectors(
        geometry=geometry,
        scale=scale,
        geometryType='polygon',
        eightConnected=False,
        maxPixels=1e9
    )
    
    # Add area calculation to each feature
    def add_area(feature):
        area = feature.geometry().area(maxError=1)
        perimeter = feature.geometry().perimeter(maxError=1)
        # Calculate compactness (4π*area/perimeter²)
        compactness = area.multiply(4).multiply(3.14159).divide(
            perimeter.multiply(perimeter)
        )
        
        return feature.set({
            'area_sqm': area,
            'perimeter_m': perimeter,
            'compactness': compactness
        })
    
    islands_with_metrics = island_vectors.map(add_area)
    
    # Filter by minimum area
    filtered_islands = islands_with_metrics.filter(
        ee.Filter.gte('area_sqm', min_area)
    )
    
    return filtered_islands


def analyze_island_detection_results(island_mask, island_vectors, geometry):
    """
    Generate summary statistics for island detection results.
    
    Parameters
    ----------
    island_mask : ee.Image
        Binary island mask
    island_vectors : ee.FeatureCollection
        Vector polygons of detected islands
    geometry : ee.Geometry
        Study area geometry
        
    Returns
    -------
    dict
        Summary statistics including counts, areas, and metrics
    """
    # Calculate total island area from mask
    total_island_pixels = island_mask.selfMask().reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=geometry,
        scale=10,
        maxPixels=1e9
    )
    
    # Calculate study area
    study_area = geometry.area(maxError=1)
    
    # Summary statistics from vectors
    island_count = island_vectors.size()
    total_vector_area = island_vectors.aggregate_sum('area_sqm')
    avg_island_area = island_vectors.aggregate_mean('area_sqm')
    avg_compactness = island_vectors.aggregate_mean('compactness')
    
    summary = {
        'study_area_sqm': study_area,
        'total_island_pixels': total_island_pixels,
        'island_count': island_count,
        'total_island_area_sqm': total_vector_area,
        'average_island_area_sqm': avg_island_area,
        'average_compactness': avg_compactness
    }
    
    return summary