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
    aoi = ee.Geometry.Rectangle(data["aoi"])
    name = data["project_name"]
    path = data["path"]
    threshold = 1

    #get a max of 3 imagers per site
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
    retrieve_tiff_from_collection(data, l9_col, 0, path + "/l9tiffs")


    for i in range(length):
        #get two arrays from each s2 image, one for rgb and one for nir
        rgb,nir = retrieve_rgb_nir_from_collection(data, se2_col, i)
        #get a tiff with all bands to calculate coregistration offsets
        retrieve_tiff_from_collection(data, se2_col,i, path + "/tiffs")

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
        rgb_path = path + "/" + img_name + "_rgb.png"
        rgb_img.save(rgb_path)
        nir_path = path + "/" + img_name + "_nir.png"
        nir_img.save(nir_path)

        # add row to table
        new_row = pd.DataFrame({'Index': [i], 'name': [img_name], 'rgb path': [rgb_path], 'nir path': [nir_path]})
        proj_track = pd.concat([proj_track, new_row], ignore_index=True)

    return proj_track


def process_collection_images_totar(data: Dict, se2_col: ee.ImageCollection) -> pd.DataFrame:
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
    site_path = path.replace(name,"littoral_sites.csv")
    littoral_sites.set_last_run(name, date_str,site_path)
    print(f"finished run: {name} {date_str}")

    return proj_track


def retrieve_tiff_from_collection(data, se2_col,index,folder_path):

  aoi = ee.Geometry.Rectangle(data["aoi"])

  # expand the aoi by 0.1 degrees to get more pixels to co-register
  aoi = aoi.buffer(0.1)

  tiff_collection = ee.Image(se2_col.toList(se2_col.size()).get(index))
  url = tiff_collection.getDownloadUrl({
    'region': aoi,
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
    aoi_rec = ee.Geometry.Rectangle(data["aoi"])
    #expand the aoi by 0.1 degrees
    aoi_rec = aoi_rec.buffer(0.1)

    landsat_col = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterDate("2021-01-01",data["end_date"]).filterBounds(aoi_rec)
    landsat_col.sort('CLOUD_COVER')

    return landsat_col