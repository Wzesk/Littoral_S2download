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

def load_site_parameters(name,path):
  site_row = get_site_by_name(name)
  aoi_str = site_row['aoi'].values[0]
  aoi_str = "".join(aoi_str.split())
  aoi = json.loads(aoi_str)
  aoi_rec = ee.Geometry.Rectangle(aoi)

  start = '2024-01-01'
  end = '2024-12-31'
  proj_name = name
  path = path

  save_path = path + "/" + proj_name
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  proj_data = save_parameters_to_json(aoi, start, end, proj_name, save_path)

  return proj_data

def save_parameters_to_json(aoi, start, end, project_name, path):
    data = {
        "aoi": aoi,
        "start_date": start,
        "end_date": end,
        "project_name": project_name,
        "path":path
    }
    save_path = path + "/" + project_name + ".json"
    with open(save_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    return data