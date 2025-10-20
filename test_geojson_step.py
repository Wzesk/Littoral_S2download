#!/usr/bin/env python3
"""
Integration test for GeoJSON conversion step within the pipeline framework.
"""

import os
import sys
import json
import logging
from datetime import datetime

# Add pipeline to path
sys.path.insert(0, os.path.dirname(__file__))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_geojson_step():
    """Test the GeoJSON conversion step using the test framework."""
    
    print("Testing GeoJSON Conversion Step for Fenfushi")
    print("=" * 60)
    
    # Import after path is set
    from test_framework.tests.step_testers import GeoJSONTester
    
    # Create test configuration
    step_config = {
        'enabled': True,
        'timeout': 600,
        'max_files': 2  # Limit for testing
    }
    
    # Initialize tester
    tester = GeoJSONTester('geojson_convert', step_config)
    
    # Run the test
    site_name = 'Fenfushi'
    site_path = '/home/walter_littor_al/geotools_sites/Fenfushi'
    
    print(f"Running GeoJSON conversion test for {site_name}")
    print(f"Site path: {site_path}")
    
    try:
        success, metrics = tester._execute_step(site_name, site_path)
        
        print(f"\nTest Results:")
        print(f"Success: {success}")
        print(f"Metrics: {json.dumps(metrics, indent=2, default=str)}")
        
        if success:
            print("\n✅ GeoJSON conversion step test PASSED!")
            
            # Summary of key metrics
            if 'converted_files' in metrics:
                print(f"📄 Files converted: {metrics['converted_files']}")
            if 'execution_time_seconds' in metrics:
                print(f"⏱️  Execution time: {metrics['execution_time_seconds']} seconds")
            if 'success_rate' in metrics:
                print(f"📊 Success rate: {metrics['success_rate'] * 100:.1f}%")
            if 'public_url_accessible' in metrics:
                print(f"🌐 Public URL accessible: {metrics['public_url_accessible']}")
            if 'bigquery_upload_success' in metrics:
                print(f"📊 BigQuery upload: {metrics['bigquery_upload_success']}")
                
        else:
            print(f"\n❌ GeoJSON conversion step test FAILED")
            if 'error' in metrics:
                print(f"Error: {metrics['error']}")
    
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_geojson_step()