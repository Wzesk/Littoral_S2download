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


def get_site_by_name(name):
    """Retrieve site data by site name.

    Parameters
    ----------
    name : str
        Name of the site to retrieve

    Returns
    -------
    pd.DataFrame
        Single row DataFrame with site information
    """
    sites = load_sites()
    site = sites[sites["site_name"] == name]
    return site


def list_site_names():
    """Get list of all available site names.

    Returns
    -------
    list
        List of site names from the sites CSV
    """
    sites = load_sites()
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
    aoi_rec = ee.Geometry.Rectangle(aoi)

    start = site_row["start"].values[0]
    end = site_row["end"].values[0]
    usable_percentage = float(site_row["usable_percentage"].values[0])
    proj_name = name
    path = path

    save_path = path + "/" + proj_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    proj_data = save_parameters_to_json(
        aoi, start, end, usable_percentage, proj_name, save_path
    )

    return proj_data


def save_parameters_to_json(
    aoi, start, end, usable_pixel_percentage, project_name, path
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
    usable_pixel_percentage : float
        Minimum percentage of usable pixels required
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
        "usable_pixel_percentage": usable_pixel_percentage,
        "project_name": project_name,
        "path": path,
    }
    save_path = path + "/" + project_name + ".json"
    with open(save_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    return data
