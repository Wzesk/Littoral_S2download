# import io
# from io import BytesIO
# import tarfile
# import datetime
# import os
# import json
# import pandas as pd
# from PIL import Image
# from skimage.morphology import binary_dilation, disk, remove_small_objects, binary_erosion
# import tarfile
# import requests
# from pathlib import Path
# from functools import partial
# import rasterio as rio
# import numpy as np
# from matplotlib import pyplot as plt

import io
import os
import json
import pandas as pd
import ee

def load_sites(path='littoral_pipeline/littoral_sites.csv'):
  #open the csv file as a dataframe
  sites = pd.read_csv(path)
  return sites

def get_site_by_name(name):
  sites = load_sites()
  site = sites[sites['site_name'] == name]
  return site

def list_site_names():
  sites = load_sites()
  return sites['site_name'].tolist()

def set_last_run(site_name, date):
  sites = load_sites()
  sites.loc[sites['site_name'] == site_name, 'last_run'] = date
  spreadsheet = gc.open('littoral_analysis_sites')
  worksheet = spreadsheet.worksheet('sites')
  worksheet.update([sites.columns.values.tolist()] + sites.values.tolist())
  new_sites = load_sites()
  return new_sites

def load_site_parameters(name,path):
  site_row = get_site_by_name(name)
  aoi_str = site_row['aoi'].values[0]
  aoi_str = "".join(aoi_str.split())
  aoi = json.loads(aoi_str)
  aoi_rec = ee.Geometry.Rectangle(aoi)

  start = site_row['start'].values[0]
  end = site_row['end'].values[0]
  usable_percentage = float(site_row['usable_percentage'].values[0])
  proj_name = name
  path = path

  save_path = path + "/" + proj_name
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  proj_data = save_parameters_to_json(aoi, start, end, usable_percentage, proj_name, save_path)

  return proj_data

def save_parameters_to_json(aoi, start, end, usable_pixel_percentage, project_name, path):
    data = {
        "aoi": aoi,
        "start_date": start,
        "end_date": end,
        "usable_pixel_percentage": usable_pixel_percentage,
        "project_name": project_name,
        "path":path
    }
    save_path = path + "/" + project_name + ".json"
    with open(save_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    return data