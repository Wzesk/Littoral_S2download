#!/usr/bin/env python3
"""
Simple GeoJSON conversion test for Fenfushi data without cloud dependencies.
"""

import os
import sys
import glob
import json
import pandas as pd
import pyproj
from datetime import datetime
from typing import List, Dict, Any

def convert_coordinates_to_wgs84(x_coords, y_coords):
    """Convert UTM coordinates to WGS84 (longitude, latitude)."""
    try:
        # Assume UTM Zone 43N based on the file names (T43NBF, T43NBG)
        utm_proj = pyproj.Proj(proj='utm', zone=43, ellps='WGS84', datum='WGS84')
        wgs84_proj = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
        
        # Convert coordinates
        transformer = pyproj.Transformer.from_proj(utm_proj, wgs84_proj, always_xy=True)
        lon_coords, lat_coords = transformer.transform(x_coords, y_coords)
        
        # Return as [longitude, latitude] pairs for GeoJSON
        coordinates = [[float(lon), float(lat)] for lon, lat in zip(lon_coords, lat_coords)]
        
        return coordinates
        
    except Exception as e:
        print(f"Error converting coordinates: {e}")
        # Fallback: return original coordinates (scaled for rough lat/lon)
        return [[float(x)/100000, float(y)/100000] for x, y in zip(x_coords, y_coords)]

def csv_to_geojson(csv_file: str, site_name: str) -> Dict[str, Any]:
    """Convert CSV coordinates to GeoJSON format."""
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Check columns
        if 'xm' not in df.columns or 'ym' not in df.columns:
            print(f"CSV file {csv_file} missing required columns (xm, ym)")
            return None
        
        print(f"Processing {len(df)} coordinates from {os.path.basename(csv_file)}")
        
        # Convert UTM to WGS84 for web compatibility
        coordinates = convert_coordinates_to_wgs84(df['xm'].values, df['ym'].values)
        
        if len(coordinates) < 2:
            print(f"Insufficient coordinates in {csv_file}")
            return None
        
        # Create GeoJSON LineString
        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {
                    "site_name": site_name,
                    "source_file": os.path.basename(csv_file),
                    "processing_date": datetime.now().isoformat(),
                    "data_source": "littoral_pipeline",
                    "coordinate_system": "WGS84",
                    "total_points": len(coordinates),
                    "bounds": {
                        "min_lon": min(c[0] for c in coordinates),
                        "max_lon": max(c[0] for c in coordinates),
                        "min_lat": min(c[1] for c in coordinates),
                        "max_lat": max(c[1] for c in coordinates)
                    }
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates
                }
            }]
        }
        
        return geojson
        
    except Exception as e:
        print(f"Error converting CSV to GeoJSON: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_line_length(coordinates) -> float:
    """Calculate approximate length of line in meters."""
    try:
        from math import radians, sin, cos, sqrt, atan2
        
        total_length = 0.0
        R = 6371000  # Earth radius in meters
        
        for i in range(1, len(coordinates)):
            lon1, lat1 = radians(coordinates[i-1][0]), radians(coordinates[i-1][1])
            lon2, lat2 = radians(coordinates[i][0]), radians(coordinates[i][1])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            
            total_length += R * c
        
        return total_length
        
    except Exception as e:
        print(f"Error calculating line length: {e}")
        return 0.0

def test_fenfushi_conversion():
    """Test GeoJSON conversion with Fenfushi data."""
    
    site_name = 'Fenfushi'
    site_path = '/home/walter_littor_al/geotools_sites/Fenfushi'
    
    print(f"Testing GeoJSON conversion for {site_name}")
    print("=" * 50)
    
    # Find tidal corrected files
    tidal_dir = os.path.join(site_path, 'TIDAL_CORRECTED')
    csv_files = glob.glob(os.path.join(tidal_dir, '*_tidal_corrected.csv'))
    
    print(f"Found {len(csv_files)} tidal corrected files:")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")
    
    if not csv_files:
        print("No tidal corrected files found")
        return
    
    # Convert each file
    output_dir = '/tmp/fenfushi_geojson'
    os.makedirs(output_dir, exist_ok=True)
    
    successful_conversions = 0
    
    for csv_file in csv_files:
        print(f"\nProcessing: {os.path.basename(csv_file)}")
        print("-" * 40)
        
        # Convert to GeoJSON
        geojson_data = csv_to_geojson(csv_file, site_name)
        
        if geojson_data:
            # Extract info
            feature = geojson_data['features'][0]
            coords = feature['geometry']['coordinates']
            props = feature['properties']
            
            # Calculate metrics
            length_m = calculate_line_length(coords)
            
            print(f"✓ Conversion successful!")
            print(f"  - Points: {len(coords)}")
            print(f"  - Length: {length_m:.1f} meters")
            print(f"  - Bounds: [{props['bounds']['min_lon']:.6f}, {props['bounds']['min_lat']:.6f}] to [{props['bounds']['max_lon']:.6f}, {props['bounds']['max_lat']:.6f}]")
            
            # Save to file
            base_name = os.path.basename(csv_file).replace('_tidal_corrected.csv', '')
            output_file = os.path.join(output_dir, f"{base_name}.geojson")
            
            with open(output_file, 'w') as f:
                json.dump(geojson_data, f, indent=2)
            
            print(f"  - Saved to: {output_file}")
            successful_conversions += 1
            
        else:
            print(f"✗ Conversion failed for {os.path.basename(csv_file)}")
    
    print(f"\n" + "=" * 50)
    print(f"Conversion Summary:")
    print(f"  - Total files: {len(csv_files)}")
    print(f"  - Successful: {successful_conversions}")
    print(f"  - Failed: {len(csv_files) - successful_conversions}")
    print(f"  - Output directory: {output_dir}")
    
    # Test loading one of the GeoJSON files
    if successful_conversions > 0:
        print(f"\nTesting GeoJSON file loading...")
        geojson_files = glob.glob(os.path.join(output_dir, '*.geojson'))
        test_file = geojson_files[0]
        
        try:
            with open(test_file, 'r') as f:
                loaded_data = json.load(f)
            
            print(f"✓ Successfully loaded {os.path.basename(test_file)}")
            print(f"  - Type: {loaded_data['type']}")
            print(f"  - Features: {len(loaded_data['features'])}")
            
            feature = loaded_data['features'][0]
            print(f"  - Geometry type: {feature['geometry']['type']}")
            print(f"  - Coordinates: {len(feature['geometry']['coordinates'])}")
            
        except Exception as e:
            print(f"✗ Error loading GeoJSON: {e}")

if __name__ == '__main__':
    test_fenfushi_conversion()