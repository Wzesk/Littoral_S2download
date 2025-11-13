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
from typing import Optional

try:
    from google.cloud import bigquery  # BigQuery client for remote site metadata
    import google.auth
except ImportError:
    bigquery = None
    google.auth = None

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


# --- BigQuery site loading utilities -------------------------------------------------
_cached_bq_sites_df: Optional[pd.DataFrame] = None

def load_sites_from_bigquery(table_fqid: str = "useful-theory-442820-q8.shoreline_metadata.islands") -> pd.DataFrame:
    """
    Load site parameters from BigQuery.
    
    Args:
        table_fqid: Fully qualified BigQuery table ID (project.dataset.table)
        
    Returns:
        DataFrame with site parameters
    """
    global _cached_bq_sites_df
    
    if _cached_bq_sites_df is not None:
        return _cached_bq_sites_df
    
    try:
        # Authenticate and create BigQuery client
        credentials, project_id = google.auth.default()
        client = bigquery.Client(credentials=credentials, project=project_id)
        
        # Query the table
        query = f"SELECT * FROM `{table_fqid}`"
        df = client.query(query).to_dataframe()
        
        # Normalize column names to match expected format
        column_mapping = {}
        
        # Handle site_name variations
        if 'name' in df.columns and 'site_name' not in df.columns:
            column_mapping['name'] = 'site_name'
        
        # Handle AOI variations - look for aoi_coordinates first
        if 'aoi_coordinates' in df.columns and 'aoi' not in df.columns:
            column_mapping['aoi_coordinates'] = 'aoi'
        elif 'aoi_coordinates_parsed' in df.columns and 'aoi' not in df.columns:
            column_mapping['aoi_coordinates_parsed'] = 'aoi'
        
        # Handle date field variations
        if 'start_date' in df.columns and 'start' not in df.columns:
            column_mapping['start_date'] = 'start'
        if 'end_date' in df.columns and 'end' not in df.columns:
            column_mapping['end_date'] = 'end'
        
        # Apply column mapping
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_cols = ['site_name', 'aoi', 'start', 'periodic']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in BigQuery table: {missing}")
        
        # Add optional columns with defaults if missing
        if 'end' not in df.columns:
            df['end'] = None
        if 'max_cloudy_pixel_percentage' not in df.columns:
            df['max_cloudy_pixel_percentage'] = 10.0
        
        # Cache the result
        _cached_bq_sites_df = df
        return df
        
    except Exception as e:
        raise RuntimeError(f"Failed to load sites from BigQuery table {table_fqid}: {e}")


def get_site_by_name_bq(name: str, table_fqid: str = "useful-theory-442820-q8.shoreline_metadata.islands") -> pd.DataFrame:
    """Retrieve a single site row from BigQuery by site name.

    Parameters
    ----------
    name : str
        Site name to retrieve.
    table_fqid : str, optional
        BigQuery table FQID, by default islands metadata table.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame or empty DataFrame if not found.
    """
    df = load_sites_from_bigquery(table_fqid)
    site = df[df["site_name"].str.lower() == name.lower()]
    return site


def load_site_parameters_bq(name: str, save_path: str, table_fqid: str = "useful-theory-442820-q8.shoreline_metadata.islands"):
    """Load and prepare site parameters from BigQuery metadata.

    Mirrors functionality of `load_site_parameters_cg` but sources data from BigQuery.
    """
    site_row = get_site_by_name_bq(name, table_fqid)
    if site_row.empty:
        raise ValueError(f"Site '{name}' not found in BigQuery table {table_fqid}.")

    aoi_raw = site_row["aoi"].values[0]
    try:
        aoi = json.loads(aoi_raw) if isinstance(aoi_raw, str) else aoi_raw
    except Exception:
        raise ValueError(f"Invalid AOI JSON for site '{name}': {aoi_raw}")

    # Convert periodic to native Python bool
    periodic_raw = site_row["periodic"].values[0]
    periodic = bool(periodic_raw) if pd.notna(periodic_raw) else False
    
    start_raw = site_row["start"].values[0]
    end_raw = site_row["end"].values[0]

    # Convert dates to strings if they're date/datetime objects
    if hasattr(start_raw, 'strftime'):
        start = start_raw.strftime('%Y-%m-%d')
    else:
        start = str(start_raw) if start_raw else None
    
    if pd.isna(end_raw) or end_raw == "" or end_raw is None:
        end = datetime.now().strftime('%Y-%m-%d')
        print(f"No end date specified for site '{name}'. Using current date: {end}")
    elif hasattr(end_raw, 'strftime'):
        end = end_raw.strftime('%Y-%m-%d')
    else:
        end = str(end_raw)

    try:
        max_cloudy = float(site_row["max_cloudy_pixel_percentage"].values[0])
    except Exception:
        max_cloudy = 10.0

    proj_name = name
    site_save_path = os.path.join(save_path, proj_name)
    os.makedirs(site_save_path, exist_ok=True)

    return save_parameters_to_json(
        aoi,
        start,
        end,
        max_cloudy,
        proj_name,
        site_save_path,
        periodic,
    )


def list_site_names_bq(table_fqid: str = "useful-theory-442820-q8.shoreline_metadata.islands") -> list:
    """Get list of all available site names from BigQuery.

    Parameters
    ----------
    table_fqid : str, optional
        BigQuery table FQID, by default islands metadata table.

    Returns
    -------
    list
        List of site names from BigQuery
    """
    df = load_sites_from_bigquery(table_fqid)
    return df["site_name"].tolist()


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
    # Decide source based on environment variable. Default to BigQuery.
    source = os.getenv("LITTORAL_SITES_SOURCE", "bigquery").lower()
    bq_table = os.getenv("LITTORAL_SITES_TABLE", "littoral-375622.coastal_viewer_static.islands_metadata")

    if source == "bigquery":
        try:
            return load_site_parameters_bq(name, path, bq_table)
        except Exception as e:
            print(f"⚠️ BigQuery site loading failed ({e}); falling back to local CSV.")
            # Fall through to CSV fallback
    # CSV fallback
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
