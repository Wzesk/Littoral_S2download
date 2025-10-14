#!/usr/bin/env python3
"""
Simple test script for the Littoral Pipeline.

This script performs basic tests to ensure the pipeline components work correctly.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config_creation():
    """Test configuration file creation."""
    print("Testing configuration creation...")
    
    from pipeline.pipeline_config import PipelineConfig, create_example_config
    
    # Test default config creation
    config = PipelineConfig()
    assert config['site_table_path'] is not None
    assert config['save_path'] is not None
    
    # Test config file creation
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
        temp_config = f.name
    
    try:
        create_example_config(temp_config)
        assert os.path.exists(temp_config)
        
        # Test loading the created config
        config2 = PipelineConfig(temp_config)
        assert config2['site_name'] == 'example_site'
        
    finally:
        if os.path.exists(temp_config):
            os.unlink(temp_config)
    
    print("✅ Configuration creation test passed")


def test_config_validation():
    """Test configuration validation."""
    print("Testing configuration validation...")
    
    from pipeline.pipeline_config import PipelineConfig
    
    # Test with invalid site name
    config = PipelineConfig()
    config['site_name'] = 'nonexistent_site'
    config['site_table_path'] = '/nonexistent/path.csv'
    
    errors = config.validate_config()
    assert len(errors) > 0  # Should have validation errors
    
    print("✅ Configuration validation test passed")


def test_import_structure():
    """Test that all modules can be imported without errors."""
    print("Testing module imports...")
    
    try:
        from pipeline.pipeline_config import PipelineConfig
        from pipeline.pipeline_functions import MountManager, ImageDownloader
        from pipeline.pipeline_advanced import ShorelineFiltering, TidalCorrection
        from pipeline.pipeline_orchestrator import PipelineOrchestrator, UpdateModeManager
        print("✅ All modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        raise


def test_cli_help():
    """Test CLI help functionality."""
    print("Testing CLI help...")
    
    import subprocess
    
    try:
        result = subprocess.run([
            sys.executable, 'littoral_pipeline.py', '--help'
        ], capture_output=True, text=True, timeout=30)
        
        assert result.returncode == 0
        assert 'Littoral Processing Pipeline' in result.stdout
        print("✅ CLI help test passed")
        
    except subprocess.TimeoutExpired:
        print("❌ CLI help test timed out")
        raise
    except Exception as e:
        print(f"❌ CLI help test failed: {e}")
        raise


def test_dry_run():
    """Test dry run functionality."""
    print("Testing dry run...")
    
    import subprocess
    
    try:
        # Test with a site that should exist
        result = subprocess.run([
            sys.executable, 'littoral_pipeline.py', 
            '--site', 'Fenfushi', '--dry-run'
        ], capture_output=True, text=True, timeout=30)
        
        # Should succeed even if site doesn't exist since it's a dry run
        print(f"Dry run exit code: {result.returncode}")
        print(f"Stdout length: {len(result.stdout)}")
        if result.stderr:
            print(f"Stderr: {result.stderr[:200]}...")
        
        print("✅ Dry run test completed")
        
    except subprocess.TimeoutExpired:
        print("❌ Dry run test timed out")
        raise
    except Exception as e:
        print(f"❌ Dry run test failed: {e}")
        raise


def test_config_file_creation():
    """Test creating config files via CLI."""
    print("Testing config file creation via CLI...")
    
    import subprocess
    
    temp_config = 'test_cli_config.yaml'
    
    try:
        result = subprocess.run([
            sys.executable, 'littoral_pipeline.py',
            '--create-config', temp_config
        ], capture_output=True, text=True, timeout=30)
        
        assert result.returncode == 0
        assert os.path.exists(temp_config)
        print("✅ Config file creation via CLI test passed")
        
    finally:
        if os.path.exists(temp_config):
            os.unlink(temp_config)


def main():
    """Run all tests."""
    print("=" * 60)
    print("LITTORAL PIPELINE TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_import_structure,
        test_config_creation,
        test_config_validation,
        test_cli_help,
        test_config_file_creation,
        test_dry_run,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The pipeline is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())