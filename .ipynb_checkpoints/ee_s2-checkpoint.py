import omnicloudmask
from omnicloudmask import (
    predict_from_load_func,
    predict_from_array,
    load_s2,
)

from littoral_pipeline import tario,littoral_sites
import ee#,geemap
import os
import pandas as pd
from PIL import Image
import io
from io import BytesIO
import datetime
import os
import json
from skimage.morphology import binary_dilation, disk
import requests
import numpy as np


def connect(project_id='useful-theory-442820-q8'):
    try:
        ee.Initialize(project=project_id)
    except Exception as e:
        ee.Authenticate()
        ee.Initialize(project=project_id)

def get_image_collection(data,cloudy_pixel_percentage=10):
    aoi_rec = ee.Geometry.Rectangle(data["aoi"])
    se2_col = ee.ImageCollection('COPERNICUS/S2').filterDate(data["start_date"],data["end_date"]).filterBounds(aoi_rec).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudy_pixel_percentage))
    return se2_col

def visualize_nir_from_collection(data, se2_col,index):
    aoi = ee.Geometry.Rectangle(data["aoi"])

    rg_nir = ['B4','B3','B8'] # blue is 'B2'
    rg_nir_img = ee.Image(se2_col.toList(se2_col.size()).get(index)).select(rg_nir).clip(aoi)
    rg_nir_img_disp = rg_nir_img.divide(10000)
    return rg_nir_img_disp

def retrieve_rgb_nir_from_collection(data, se2_col,index):
    aoi = ee.Geometry.Rectangle(data["aoi"])

    rg_nir = ['B4','B3','B8'] # blue is 'B2'
    rg_nir_img = ee.Image(se2_col.toList(se2_col.size()).get(index))#.select(rg_nir)
    url = rg_nir_img.getDownloadUrl({
        'bands': rg_nir,
        'region': aoi,
        'scale': 10,
        'format': 'NPY'
    })
    projection = rg_nir_img.select('B8').projection().getInfo()
    response = requests.get(url)
    data = np.load(io.BytesIO(response.content))

    # stack 3 2d arrays into 3d array
    img_arr = np.dstack([data['B4'], data['B3'], data['B8']])
    #move 3 dimension to first position
    img_arr = np.moveaxis(img_arr, -1, 0)
    # Predict cloud and cloud shadow masks
    pred_mask = predict_from_array(img_arr)

    #merge cloud mask with cloud shadow mask
    pred_array = np.where(pred_mask == 0, 1, 0) + np.where(pred_mask == 2, 1, 0)
    pred_array = np.where(pred_array > 0, 1, 0)
    cld_pred = np.squeeze(pred_array)
    # cld_pred = np.ones_like(img_arr[0,:,:]) # to turn off cloud mask

    usable_pixels = np.sum(cld_pred) / cld_pred.size

    return data['B8'], cld_pred,usable_pixels,projection

def dilate_mask(mask, size_disk=9):
    mask = binary_dilation(mask, disk(size_disk))
    ## convert back to PIL format
    mask_image = Image.fromarray(mask)
    return mask_image

def process_collection_images(data, se2_col):
    aoi = ee.Geometry.Rectangle(data["aoi"])
    name = data["project_name"]
    path = data["path"]
    threshold = data["usable_pixel_percentage"]
    length = se2_col.size().getInfo()

    #if path does not exist, creat it
    if not os.path.exists(path):
        os.makedirs(path)
    rgnir_path = path + "/rgnir.tar"
    cldmsk_path = path + "/cldmsk.tar"

    #create a new pandas dataframe with the columns: Index, name, status,usable_percentage
    proj_track = pd.DataFrame(columns=['Index', 'name', 'rgnir_download','cld_calculation','percentage_usable'])


    for i in range(length):
        nir,cld,percentage_usable,projection = retrieve_rgb_nir_from_collection(data, se2_col, i)
        print("percentage usable pixels = " + str(percentage_usable))

        #get the name of the i image file from the collection
        img_name = ee.Image(se2_col.toList(se2_col.size()).get(i)).get('system:index').getInfo()
        new_row = pd.DataFrame({'Index': [i], 'name': [img_name], 'rgnir_download': [False], 'cld_calculation': [False], 'percentage_usable': [percentage_usable]})

        if(percentage_usable > threshold):
            #normalize
            n_nir = (nir - np.min(nir))/(np.max(nir) - np.min(nir))
            #stack
            nir_3d = np.dstack((n_nir,n_nir,n_nir))
            nir_img = Image.fromarray((nir_3d * 255).astype(np.uint8))
            #normalize
            cld = cld/np.max(cld)
            cld_3d = np.dstack((cld,cld,cld))
            cld_img = Image.fromarray((cld_3d * 255).astype(np.uint8))
            rgnirtar = tario.tar_io(rgnir_path)
            rgnirtar.save_to_tar(nir_img,img_name + "_rgnir.png",overwrite=True)
            new_row['rgnir_download'] = True
            cldtar = tario.tar_io(cldmsk_path)
            cldtar.save_to_tar(cld_img,img_name + "_cld.png",overwrite=True)
            new_row['cld_calculation'] = True

            print(img_name + " saved")
        else:
            print(str(i) + " has too many clouds")

        # add row to table
        proj_track = pd.concat([proj_track, new_row], ignore_index=True)

    #save proj_track to csv
    proj_track.to_csv(path + "/" + "proj_track.csv", index=False)

    name = data["project_name"]
    date = datetime.datetime.now()
    date_str = date.strftime("%Y-%m-%d")
    littoral_sites.set_last_run(name, date_str)
    print("finished run:",name,date_str)

    return proj_track


