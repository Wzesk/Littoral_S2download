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

from littoral import littoral_sites, tario


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
    data: Dict, cloudy_pixel_percentage: int = 10
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
    cloudy_pixel_percentage : int, optional
        Maximum cloud coverage percentage, by default 10

    Returns
    -------
    ee.ImageCollection
        Filtered collection of Sentinel-2 images

    Examples
    --------
    >>> data = {
    ...     "aoi": [-122.5, 37.7, -122.4, 37.8],
    ...     "start_date": "2023-01-01",
    ...     "end_date": "2023-12-31"
    ... }
    >>> collection = get_image_collection(data)
    """
    aoi_rec = ee.Geometry.Rectangle(data["aoi"])
    se2_col = (
        ee.ImageCollection("COPERNICUS/S2")
        .filterDate(data["start_date"], data["end_date"])
        .filterBounds(aoi_rec)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloudy_pixel_percentage))
    )
    return se2_col


def visualize_nir_from_collection(
    data: Dict, se2_col: ee.ImageCollection, index: int
) -> ee.Image:
    """Visualize NIR bands from image collection.

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
    rg_nir = ["B4", "B3", "B8"]  # blue is 'B2'
    rg_nir_img = (
        ee.Image(se2_col.toList(se2_col.size()).get(index)).select(rg_nir).clip(aoi)
    )
    rg_nir_img_disp = rg_nir_img.divide(10000)
    return rg_nir_img_disp


def retrieve_rgb_nir_from_collection(
    data: Dict, se2_col: ee.ImageCollection, index: int
) -> Tuple[np.ndarray, np.ndarray, float, Dict]:
    """Retrieve RGB+NIR bands with cloud masking.

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
    rg_nir = ["B4", "B3", "B8"]  # blue is 'B2'
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


def process_collection_images(data: Dict, se2_col: ee.ImageCollection) -> pd.DataFrame:
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
    threshold = data["usable_pixel_percentage"]
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
        nir, cld, percentage_usable, projection = retrieve_rgb_nir_from_collection(
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
    littoral_sites.set_last_run(name, date_str)
    print(f"finished run: {name} {date_str}")

    return proj_track
