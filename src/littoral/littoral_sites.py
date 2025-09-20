"""
Site Management Module
====================

This module handles the management of site data for Littoral's shoreline analysis pipeline.
It provides functionality for loading site parameters, tracking processing status,
and managing site configurations.

Functions
---------
load_sites : Load site data from CSV
get_site_by_name : Retrieve specific site data
list_site_names : Get list of all site names
set_last_run : Update site processing status
load_site_parameters : Load and prepare site configuration
save_parameters_to_json : Save site parameters to JSON
"""

import io
import os
import json
import pandas as pd
import ee
from datetime import datetime

try:
    from google.colab import auth
    from google.auth import default
    import gspread
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    auth.authenticate_user()
    creds, _ = default()
    gc = gspread.authorize(creds)
except ImportError:
    print("Google Sheets integration not available. Using local CSV only.")
    gc = None


def load_sites(path="littoral_pipeline/littoral_sites.csv"):
    """Load site data from CSV file.

    Parameters
    ----------
    path : str, optional
        Path to sites CSV file, by default 'littoral_pipeline/littoral_sites.csv'

    Returns
    -------
    pd.DataFrame
        DataFrame containing site information
    """
    # open the csv file as a dataframe
    sites = pd.read_csv(path)
    return sites


def get_site_by_name(name,path='littoral_pipeline/littoral_sites.csv'):
    """Retrieve site data by site name.

    Parameters
    ----------
    name : str
        Name of the site to retrieve
    path : str, optional
        Path to sites CSV file, by default 'littoral_pipeline/littoral_sites.csv'

    Returns
    -------
    pd.DataFrame
        Single row DataFrame with site information
    """
    sites = load_sites(path)
    site = sites[sites["site_name"] == name]
    return site


def list_site_names(path='littoral_pipeline/littoral_sites.csv'):
    """Get list of all available site names.

    Parameters
    ----------
    path : str, optional
        Path to sites CSV file, by default 'littoral_pipeline/littoral_sites.csv'

    Returns
    -------
    list
        List of site names from the sites CSV
    """
    sites = load_sites(path)
    return sites["site_name"].tolist()


def set_last_run(site_name, date, path="littoral_pipeline/littoral_sites.csv"):
    """Update the last run date for a site.

    Parameters
    ----------
    site_name : str
        Name of the site to update
    date : str
        Date string to set as last run date
    path : str, optional
        Path to sites CSV file, by default 'littoral_pipeline/littoral_sites.csv'

    Returns
    -------
    pd.DataFrame
        Updated sites DataFrame

    Notes
    -----
    Updates both local CSV and remote spreadsheet if Google Sheets integration
    is available. Always updates local CSV file.
    """
    sites = load_sites(path)
    sites.loc[sites["site_name"] == site_name, "last_run"] = date
    if gc is not None:
        try:
            spreadsheet = gc.open("littoral_analysis_sites")
            worksheet = spreadsheet.worksheet("sites")
            worksheet.update([sites.columns.values.tolist()] + sites.values.tolist())
        except Exception as e:
            print(f"Failed to update Google Sheet: {e}")
    # Save to local CSV regardless of Google Sheets status
    sites.to_csv(path, index=False)
    return sites


def load_site_parameters_cg(name, save_path, load_path):
    """Load and prepare site parameters for processing (cg demo version)

    Parameters
    ----------
    name : str
        Site name to load parameters for
    save_path : str
        Base path for saving site data
    load_path : str
        Path to table of site data

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

    site_row = get_site_by_name(name,path=load_path)
    aoi_str = site_row['aoi'].values[0]
    aoi_str = "".join(aoi_str.split())
    aoi = json.loads(aoi_str)
    
    periodic = site_row["periodic"].values[0]

    start = site_row["start"].values[0]
    end = site_row["end"].values[0]
    
    # If no end date is specified, use current date for ongoing monitoring
    if pd.isna(end) or end == '' or end is None:
        end = datetime.now().strftime('%Y-%m-%d')
        print(f"No end date specified for site '{name}'. Using current date: {end}")
    
    try:
        max_cloudy_pixel_percentage = float(site_row["max_cloudy_pixel_percentage"].values[0])
    except:
        max_cloudy_pixel_percentage = 10

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
    load_path = path + '/littoral_sites.csv'
    return load_site_parameters_cg(name, path, load_path)


def save_parameters_to_json(
    aoi, start, end, max_cloudy_pixel_percentage, project_name, path,periodic
):
    """Save site parameters to JSON file.

    Parameters
    ----------
    aoi : list
        Area of interest coordinates [x1, y1, x2, y2]
    start : str
        Start date for processing
    end : str
        End date for processing
    max_cloudy_pixel_percentage : float 
        max percentage of cloudy pixels allowed in images
    project_name : str
        Name of the project
    path : str
        Directory path for saving JSON

    Returns
    -------
    dict
        Dictionary of saved parameters
    """
    data = {
        "aoi": aoi,
        "start_date": start,
        "end_date": end,
        "max_cloudy_pixel_percentage": max_cloudy_pixel_percentage,
        "project_name": project_name,
        "path": path,
        "periodic": str(periodic),
    }
    save_path = path + "/" + project_name + ".json"
    with open(save_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    return data
