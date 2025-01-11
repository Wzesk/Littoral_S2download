
from littoral_S2download import littoral_sites
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

def get_image_collection(data,max_cloudy_pixel_percentage=0.5):
    aoi_rec = ee.Geometry.Rectangle(data["aoi"])
    se2_col = ee.ImageCollection('COPERNICUS/S2').filterDate(data["start_date"],data["end_date"]).filterBounds(aoi_rec).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloudy_pixel_percentage))
    return se2_col

def retrieve_rgb_nir_from_collection(data, se2_col,index):
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

def process_collection_images(data, se2_col, max_images=3):
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
    proj_track = pd.DataFrame(columns=['Index', 'name'])


    for i in range(length):
        #get two arrays from each s2 image, one for rgb and one for nir
        rgb,nir = retrieve_rgb_nir_from_collection(data, se2_col, i)

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
        rgb_img.save(path + "/" + img_name + "_rgb.png")
        nir_img.save(path + "/" + img_name + "_nir.png")

        # add row to table
        new_row = pd.DataFrame({'Index': [i], 'name': [img_name]})
        proj_track = pd.concat([proj_track, new_row], ignore_index=True)

    return proj_track


