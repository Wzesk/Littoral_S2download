#!/usr/bin/env python3
"""
Test-friendly GeoJSON conversion class that works without cloud credentials.
"""

import os
import sys
import glob
import json
import pandas as pd
import pyproj
from datetime import datetime
from typing import List, Dict, Any
import logging

# Add pipeline to path
sys.path.insert(0, os.path.dirname(__file__))
from pipeline.pipeline_functions import PipelineStep

class TestableGeoJSONConversion(PipelineStep):
    """Test version of GeoJSON conversion that works without cloud credentials."""
    
    def __init__(self, config):
        super().__init__(config, 'step_14_geojson_convert')
        self.logger = logging.getLogger(__name__)
        
    def run(self, corrected_files: List[str] = None) -> Dict[str, Any]:
        """Convert tidal corrected CSV files to GeoJSON format."""
        try:
            self.logger.info("Starting GeoJSON conversion (test mode)")
            
            # If no files provided, find tidal corrected files
            if not corrected_files:
                site_path = self.config.get_site_path() if hasattr(self.config, 'get_site_path') else f"/home/walter_littor_al/geotools_sites/{self.config['site_name']}"
                
                tidal_corrected_dir = os.path.join(site_path, "TIDAL_CORRECTED")
                if os.path.exists(tidal_corrected_dir):
                    corrected_files = glob.glob(os.path.join(tidal_corrected_dir, "*_tidal_corrected.csv"))
                else:
                    # Fallback to regular shoreline files
                    shoreline_dir = os.path.join(site_path, "SHORELINE")
                    corrected_files = glob.glob(os.path.join(shoreline_dir, "*_geo.csv"))
            
            if not corrected_files:
                self.logger.warning("No files found for GeoJSON conversion")
                return {
                    'converted_count': 0,
                    'metadata_records': 0,
                    'public_urls': [],
                    'total_input_files': 0
                }
            
            self.logger.info(f"Found {len(corrected_files)} files for conversion")
            
            converted_count = 0
            public_urls = []
            metadata_records = []
            
            # Create output directory
            output_dir = f"/tmp/geojson_test_{self.config['site_name']}"
            os.makedirs(output_dir, exist_ok=True)
            
            for csv_file in corrected_files:
                try:
                    # Convert CSV to GeoJSON
                    geojson_data = self._csv_to_geojson(csv_file)
                    
                    if geojson_data:
                        # Generate file names
                        base_name = os.path.basename(csv_file).replace('_tidal_corrected.csv', '').replace('_geo.csv', '')
                        geojson_filename = f"{base_name}.geojson"
                        
                        # Save to local file (simulating cloud upload)
                        local_path = os.path.join(output_dir, geojson_filename)
                        with open(local_path, 'w') as f:
                            json.dump(geojson_data, f, indent=2)
                        
                        # Simulate public URL
                        public_url = f"https://storage.googleapis.com/littoral-public-data/shorelines/{geojson_filename}"
                        
                        # Create metadata record
                        metadata = self._create_metadata_record(csv_file, geojson_data, public_url)
                        metadata_records.append(metadata)
                        
                        public_urls.append(public_url)
                        converted_count += 1
                        
                        self.logger.info(f"Successfully converted: {base_name}")
                        
                except Exception as e:
                    self.logger.error(f"Failed to convert {csv_file}: {e}")
                    continue
            
            self.logger.info(f"GeoJSON conversion complete: {converted_count}/{len(corrected_files)} files converted")
            
            return {
                'converted_count': converted_count,
                'metadata_records': len(metadata_records),
                'public_urls': public_urls,
                'total_input_files': len(corrected_files),
                'output_directory': output_dir
            }
            
        except Exception as e:
            self.logger.error(f"Error in GeoJSON conversion: {e}")
            raise
    
    def _csv_to_geojson(self, csv_file: str) -> Dict[str, Any]:
        """Convert CSV coordinates to GeoJSON format."""
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Assume columns are xm, ym (UTM coordinates)
            if 'xm' not in df.columns or 'ym' not in df.columns:
                self.logger.error(f"CSV file {csv_file} missing required columns (xm, ym)")
                return None
            
            # Convert UTM to WGS84 for web compatibility
            coordinates = self._convert_coordinates_to_wgs84(df['xm'].values, df['ym'].values)
            
            if len(coordinates) < 2:
                self.logger.warning(f"Insufficient coordinates in {csv_file}")
                return None
            
            # Create GeoJSON LineString
            geojson = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "properties": {
                        "site_name": self.config['site_name'],
                        "source_file": os.path.basename(csv_file),
                        "processing_date": datetime.now().isoformat(),
                        "data_source": "littoral_pipeline",
                        "coordinate_system": "WGS84",
                        "total_points": len(coordinates)
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coordinates
                    }
                }]
            }
            
            return geojson
            
        except Exception as e:
            self.logger.error(f"Error converting CSV to GeoJSON: {e}")
            return None
    
    def _convert_coordinates_to_wgs84(self, x_coords, y_coords):
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
            
        except ImportError:
            self.logger.error("pyproj not available. Install with: pip install pyproj")
            # Fallback: assume coordinates are already in a reasonable system
            return [[float(x), float(y)] for x, y in zip(x_coords, y_coords)]
        except Exception as e:
            self.logger.error(f"Error converting coordinates: {e}")
            # Fallback: return original coordinates
            return [[float(x), float(y)] for x, y in zip(x_coords, y_coords)]
    
    def _create_metadata_record(self, csv_file: str, geojson_data: Dict[str, Any], public_url: str) -> Dict[str, Any]:
        """Create metadata record for BigQuery."""
        try:
            import re
            
            # Extract information from filename and data
            filename = os.path.basename(csv_file)
            
            # Parse date from filename (e.g., 20240116T052149)
            date_match = re.search(r'(\d{8})', filename)
            image_date = None
            if date_match:
                date_str = date_match.group(1)
                image_date = datetime.strptime(date_str, '%Y%m%d').date()
            
            # Calculate metrics from GeoJSON
            feature = geojson_data['features'][0]
            coordinates = feature['geometry']['coordinates']
            
            # Calculate total length (approximate)
            total_length_m = self._calculate_line_length(coordinates)
            
            # Create metadata record
            record = {
                'shoreline_id': f"{self.config['site_name']}_{filename.replace('.csv', '')}",
                'site_name': self.config['site_name'],
                'image_date': image_date,
                'processing_date': datetime.now().isoformat(),
                'geojson_url': public_url,
                'total_length_m': total_length_m,
                'num_features': 1,
                'avg_feature_length_m': total_length_m,
                'quality_score': min(1.0, len(coordinates) / 100.0),  # Simple quality metric
                'data_source': 'littoral_pipeline',
                'processing_version': '1.0'
            }
            
            return record
            
        except Exception as e:
            self.logger.error(f"Error creating metadata record: {e}")
            return None
    
    def _calculate_line_length(self, coordinates) -> float:
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
            self.logger.error(f"Error calculating line length: {e}")
            return 0.0


# Test the class
if __name__ == '__main__':
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # Simple config for testing
    class TestConfig:
        def __init__(self, site_name):
            self.config = {'site_name': site_name}
        
        def __getitem__(self, key):
            return self.config[key]
        
        def get_site_path(self):
            return f"/home/walter_littor_al/geotools_sites/{self.config['site_name']}"
    
    config = TestConfig('Fenfushi')
    converter = TestableGeoJSONConversion(config)
    
    result = converter.run()
    print(f"Conversion result: {json.dumps(result, indent=2, default=str)}")