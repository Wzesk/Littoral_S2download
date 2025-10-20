#!/usr/bin/env python3
"""
Test script for GeoJSON conversion step with Fenfushi data.
"""

import os
import sys
import glob
import logging
from datetime import datetime

# Add the pipeline directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.pipeline_advanced import GeoJSONConversion

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleConfig:
    """Simple configuration class for testing."""
    
    def __init__(self, site_name, site_path):
        self.config = {
            'site_name': site_name
        }
        self._site_path = site_path
    
    def __getitem__(self, key):
        return self.config[key]
    
    def get_site_path(self):
        return self._site_path

def test_geojson_conversion():
    """Test the GeoJSON conversion step."""
    
    # Configuration
    site_name = 'Fenfushi'
    site_path = '/home/walter_littor_al/geotools_sites/Fenfushi'
    
    print(f"Testing GeoJSON conversion for {site_name}")
    print(f"Site path: {site_path}")
    
    # Create config
    config = SimpleConfig(site_name, site_path)
    
    # Find tidal corrected files
    tidal_dir = os.path.join(site_path, 'TIDAL_CORRECTED')
    csv_files = glob.glob(os.path.join(tidal_dir, '*_tidal_corrected.csv'))
    
    print(f"Found {len(csv_files)} tidal corrected files:")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")
    
    if not csv_files:
        print("No tidal corrected files found - checking SHORELINE directory")
        shoreline_dir = os.path.join(site_path, 'SHORELINE')
        csv_files = glob.glob(os.path.join(shoreline_dir, '*_geo.csv'))
        print(f"Found {len(csv_files)} shoreline files:")
        for f in csv_files:
            print(f"  - {os.path.basename(f)}")
    
    if not csv_files:
        print("No suitable CSV files found for conversion")
        return
    
    # Test coordinate conversion first
    print("\n=== Testing Coordinate Conversion ===")
    import pandas as pd
    import pyproj
    
    test_file = csv_files[0]
    print(f"Testing with: {os.path.basename(test_file)}")
    
    try:
        df = pd.read_csv(test_file)
        print(f"Loaded {len(df)} coordinates")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Sample coordinates: {df.head(3).values.tolist()}")
        
        # Test projection
        x_coords = df['xm'].values[:5]
        y_coords = df['ym'].values[:5]
        
        utm_proj = pyproj.Proj(proj='utm', zone=43, ellps='WGS84', datum='WGS84')
        wgs84_proj = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
        transformer = pyproj.Transformer.from_proj(utm_proj, wgs84_proj, always_xy=True)
        
        lon_coords, lat_coords = transformer.transform(x_coords, y_coords)
        print(f"Converted coordinates: {list(zip(lon_coords[:3], lat_coords[:3]))}")
        
    except Exception as e:
        print(f"Error in coordinate test: {e}")
        return
    
    # Test GeoJSON conversion without cloud upload
    print("\n=== Testing GeoJSON Creation ===")
    
    try:
        # Create converter instance
        converter = GeoJSONConversion.__new__(GeoJSONConversion)
        converter.config = config
        converter.logger = logger
        
        # Test CSV to GeoJSON conversion manually
        import json
        
        geojson_data = converter._csv_to_geojson(test_file)
        
        if geojson_data:
            print("✓ GeoJSON conversion successful!")
            feature = geojson_data['features'][0]
            coords = feature['geometry']['coordinates']
            print(f"✓ Created LineString with {len(coords)} points")
            print(f"✓ First coordinate: {coords[0]}")
            print(f"✓ Last coordinate: {coords[-1]}")
            print(f"✓ Properties: {feature['properties']}")
            
            # Save to local file for inspection
            output_file = f"/tmp/{site_name}_test.geojson"
            with open(output_file, 'w') as f:
                json.dump(geojson_data, f, indent=2)
            print(f"✓ Saved test GeoJSON to: {output_file}")
            
        else:
            print("✗ GeoJSON conversion failed")
            
    except Exception as e:
        print(f"✗ Error in GeoJSON conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_geojson_conversion()