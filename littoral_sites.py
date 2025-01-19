import io
import os
import json
import pandas as pd
import ee

def load_sites(path='littoral_pipeline/littoral_sites.csv'):
  #open the csv file as a dataframe
  sites = pd.read_csv(path)
  return sites

def get_site_by_name(name,path='littoral_pipeline/littoral_sites.csv'):
  sites = load_sites(path)
  site = sites[sites['site_name'] == name]
  return site

def list_site_names(path='littoral_pipeline/littoral_sites.csv'):
  sites = load_sites(path)
  return sites['site_name'].tolist()


def load_site_parameters_cg(name, save_path, load_path, 
                            start='2024-01-01',end='2024-12-31',
                            max_cloudy_pixel_percentage=10):
  
  site_row = get_site_by_name(name,path=load_path)
  aoi_str = site_row['aoi'].values[0]
  aoi_str = "".join(aoi_str.split())
  aoi = json.loads(aoi_str)
  periodic = site_row['periodic'].values[0]

  proj_name = name
  save_path = save_path + "/" + proj_name

  if not os.path.exists(save_path):
    os.makedirs(save_path)

  proj_data = save_parameters_to_json(
     aoi, start, end, max_cloudy_pixel_percentage, proj_name, save_path, periodic
     )

  return proj_data

def load_site_parameters(name, path):
    """Load and prepare site parameters for processing.

    Parameters
    ----------
    name : str
        Site name to load parameters for
    path : str
        Base path for saving site data

    Returns
    -------
    dict
        Site parameters including:
        - aoi : list
            Area of interest coordinates
        - start_date : str
            Start date for processing
        - end_date : str
            End date for processing
        - project_name : str
            Name of the project
        - path : str
            Output directory path

    Notes
    -----
    Creates output directory if it doesn't exist.
    """
    site_row = get_site_by_name(name)
    aoi_str = site_row["aoi"].values[0]
    aoi_str = "".join(aoi_str.split())
    aoi = json.loads(aoi_str)
    periodic = site_row['periodic'].values[0]

    start = site_row["start"].values[0]
    end = site_row["end"].values[0]
    max_cloudy_pixel_percentage = float(site_row["max_cloudy_pixel_percentage"].values[0])
    proj_name = name
    path = path

    save_path = path + "/" + proj_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    proj_data = save_parameters_to_json(
        aoi, start, end, max_cloudy_pixel_percentage, proj_name, save_path, periodic
    )

    return proj_data

def save_parameters_to_json(aoi, start, end, max_cloudy_pixel_percentage, project_name, path, periodic):
    data = {
        "aoi": aoi,
        "start_date": start,
        "end_date": end,
        "max_cloudy_pixel_percentage": max_cloudy_pixel_percentage,
        "project_name": project_name,
        "path":path,
        "periodic":periodic
    }
    save_path = path + "/" + project_name + ".json"
    with open(save_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    return data