#!/usr/bin/env python3
"""
Test script for actual cloud GeoJSON conversion and upload.
This uses the production GeoJSONConversion class to upload real files to Google Cloud.
"""

import os
import sys
import glob
import logging
from datetime import datetime

# Add the pipeline modules to the path
sys.path.append('/home/walter_littor_al/Littoral_S2download')

from pipeline.pipeline_advanced import GeoJSONConversion
from pipeline.pipeline_config import PipelineConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_cloud_geojson_upload(site_name='Fenfushi', max_files=2):
    """Test the actual cloud GeoJSON conversion and upload."""
    
    print(f"🚀 Testing Cloud GeoJSON Upload for {site_name}")
    print("=" * 60)
    
    try:
        # Setup configuration
        config = PipelineConfig()
        config.config['site_name'] = site_name
        
        # Set site path
        site_path = f'/home/walter_littor_al/geotools_sites/{site_name}'
        config.config['site_path'] = site_path
        
        if not os.path.exists(site_path):
            print(f"❌ Site directory not found: {site_path}")
            return False
        
        print(f"📁 Site path: {site_path}")
        
        # Find tidal corrected files
        tidal_dir = os.path.join(site_path, "TIDAL_CORRECTED")
        if os.path.exists(tidal_dir):
            csv_files = glob.glob(os.path.join(tidal_dir, "*_tidal_corrected.csv"))
            print(f"📄 Found {len(csv_files)} tidal corrected files")
        else:
            # Fallback to regular shoreline files
            shoreline_dir = os.path.join(site_path, "SHORELINE")
            csv_files = glob.glob(os.path.join(shoreline_dir, "*_geo.csv"))
            print(f"📄 Found {len(csv_files)} shoreline files (no tidal correction)")
        
        if not csv_files:
            print("❌ No CSV files found for conversion")
            return False
        
        # Limit files for testing
        if max_files and len(csv_files) > max_files:
            csv_files = csv_files[:max_files]
            print(f"🔀 Limited to {max_files} files for testing")
        
        for i, file in enumerate(csv_files, 1):
            print(f"  {i}. {os.path.basename(file)}")
        
        # Initialize the production GeoJSON converter
        print("\n🔧 Initializing GeoJSON converter...")
        converter = GeoJSONConversion(config)
        print("✅ Converter initialized with cloud clients")
        
        # Test the conversion and upload
        print("\n🔄 Starting conversion and upload...")
        start_time = datetime.now()
        
        results = converter.run(csv_files)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Display results
        print("\n📊 UPLOAD RESULTS")
        print("-" * 40)
        print(f"⏱️  Processing time: {duration:.2f} seconds")
        print(f"📁 Input files: {results.get('total_input_files', 0)}")
        print(f"✅ Converted files: {results.get('converted_count', 0)}")
        print(f"📝 Metadata records: {results.get('metadata_records', 0)}")
        print(f"🌐 Public URLs: {len(results.get('public_urls', []))}")
        
        # Show some public URLs
        public_urls = results.get('public_urls', [])
        if public_urls:
            print(f"\n🔗 PUBLIC URLs (showing first {min(3, len(public_urls))}):")
            for i, url in enumerate(public_urls[:3], 1):
                print(f"  {i}. {url}")
                
            # Test accessibility of first URL
            print(f"\n🌐 Testing URL accessibility...")
            try:
                import requests
                response = requests.head(public_urls[0], timeout=10)
                if response.status_code == 200:
                    print(f"✅ First URL is accessible (HTTP {response.status_code})")
                else:
                    print(f"⚠️  First URL returned HTTP {response.status_code}")
            except Exception as e:
                print(f"❌ URL accessibility test failed: {e}")
        
        # Check BigQuery upload
        if results.get('metadata_records', 0) > 0:
            print(f"\n📊 Verifying BigQuery upload...")
            try:
                from google.cloud import bigquery
                bq_client = bigquery.Client()
                
                # Query the latest records
                query = f"""
                SELECT site_id, date_processed, shoreline_length_m, geojson_path
                FROM `{bq_client.project}.shoreline_metadata.shoreline_data`
                WHERE site_id = '{site_name}'
                ORDER BY date_processed DESC
                LIMIT 3
                """
                
                query_job = bq_client.query(query)
                rows = list(query_job)
                
                print(f"✅ Found {len(rows)} recent records in BigQuery:")
                for row in rows:
                    print(f"  - {row.date_processed}: {row.shoreline_length_m:.1f}m long")
                    
            except Exception as e:
                print(f"❌ BigQuery verification failed: {e}")
        
        # Success metrics
        success = (results.get('converted_count', 0) > 0 and 
                  results.get('metadata_records', 0) > 0)
        
        if success:
            print(f"\n🎉 SUCCESS: Cloud upload completed successfully!")
            print(f"   - {results.get('converted_count', 0)} files uploaded to cloud storage")
            print(f"   - {results.get('metadata_records', 0)} metadata records in BigQuery")
        else:
            print(f"\n❌ FAILED: Upload did not complete successfully")
            if 'error' in results:
                print(f"   Error: {results['error']}")
        
        return success
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("Cloud GeoJSON Upload Test")
    print("=" * 60)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌍 Project: {os.environ.get('GOOGLE_CLOUD_PROJECT', 'default')}")
    
    # Test with Fenfushi (limiting to 2 files for initial test)
    success = test_cloud_geojson_upload('Fenfushi', max_files=2)
    
    if success:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())