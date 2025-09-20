#!/usr/bin/env python3
"""
Test suite for cloud storage mounting functionality.

This module contains tests for the mounting system used in the littoral pipeline.
These tests verify that the MountManager and mount utilities work correctly.
"""

import sys
import os
import unittest
import subprocess
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add parent directory to path to import pipeline modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_config import PipelineConfig
from pipeline_functions import MountManager
import mount_utils


class TestMountManager(unittest.TestCase):
    """Test the MountManager class functionality."""
    
    def setUp(self):
        """Set up test configuration."""
        self.config = PipelineConfig()
        self.config.config['site_name'] = 'test_site'
        
        # Use temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.config.config['mounting']['geotools_mount'] = os.path.join(self.temp_dir, 'geotools')
        self.config.config['mounting']['tide_mount'] = os.path.join(self.temp_dir, 'tide')
        
        self.mount_manager = MountManager(self.config.config)
    
    def tearDown(self):
        """Clean up test environment."""
        # Clean up any mounts
        self.mount_manager.unmount_all()
        
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_mount_manager_initialization(self):
        """Test that MountManager initializes correctly."""
        self.assertIsNotNone(self.mount_manager.config)
        self.assertEqual(self.mount_manager.mounted_paths, [])
        self.assertIsNotNone(self.mount_manager.logger)
    
    @patch('subprocess.run')
    def test_mount_geotools_success(self, mock_run):
        """Test successful geotools bucket mounting."""
        # Mock successful mount command
        mock_run.return_value = MagicMock(returncode=0, stdout='', stderr='')
        
        # Mock os.listdir for mount verification
        with patch('os.listdir', return_value=['item1', 'item2']):
            result = self.mount_manager.mount_geotools_bucket()
        
        self.assertTrue(result)
        self.assertIn(self.config.config['mounting']['geotools_mount'], 
                     self.mount_manager.mounted_paths)
    
    @patch('subprocess.run')
    def test_mount_geotools_failure(self, mock_run):
        """Test failed geotools bucket mounting."""
        # Mock failed mount command
        mock_run.return_value = MagicMock(returncode=1, stdout='', stderr='Mount failed')
        
        result = self.mount_manager.mount_geotools_bucket()
        
        self.assertFalse(result)
        self.assertEqual(len(self.mount_manager.mounted_paths), 0)
    
    @patch('subprocess.run')
    def test_mount_tide_success(self, mock_run):
        """Test successful tide bucket mounting."""
        # Mock successful mount command
        mock_run.return_value = MagicMock(returncode=0, stdout='', stderr='')
        
        # Mock os.listdir for mount verification
        with patch('os.listdir', return_value=['fes2022b', 'region0']):
            result = self.mount_manager.mount_tide_bucket()
        
        self.assertTrue(result)
        self.assertIn(self.config.config['mounting']['tide_mount'], 
                     self.mount_manager.mounted_paths)
    
    @patch('subprocess.run')
    def test_mount_all_required_success(self, mock_run):
        """Test mounting all required buckets successfully."""
        # Mock successful mount commands
        mock_run.return_value = MagicMock(returncode=0, stdout='', stderr='')
        
        # Mock os.listdir for mount verification
        with patch('os.listdir', return_value=['item1', 'item2']):
            result = self.mount_manager.mount_all_required()
        
        self.assertTrue(result)
        self.assertEqual(len(self.mount_manager.mounted_paths), 2)
    
    @patch('subprocess.run')
    def test_mount_all_required_partial_failure(self, mock_run):
        """Test mounting with partial failure."""
        # Mock mixed success/failure
        def side_effect(*args, **kwargs):
            if 'geotools' in args[0]:
                return MagicMock(returncode=0, stdout='', stderr='')
            else:
                return MagicMock(returncode=1, stdout='', stderr='Failed')
        
        mock_run.side_effect = side_effect
        
        # Mock os.listdir for successful mount verification
        with patch('os.listdir', return_value=['item1', 'item2']):
            result = self.mount_manager.mount_all_required()
        
        self.assertFalse(result)  # Should fail if any mount fails
    
    def test_unmount_if_mounted(self):
        """Test the _unmount_if_mounted method."""
        test_path = os.path.join(self.temp_dir, 'test_mount')
        
        # Add path to mounted_paths
        self.mount_manager.mounted_paths.append(test_path)
        
        # Call unmount
        self.mount_manager._unmount_if_mounted(test_path)
        
        # Path should be removed from mounted_paths
        self.assertNotIn(test_path, self.mount_manager.mounted_paths)


class TestMountUtils(unittest.TestCase):
    """Test the mount utility functions."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_geotools_path = os.path.join(self.temp_dir, 'geotools')
        self.test_tide_path = os.path.join(self.temp_dir, 'tide')
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('subprocess.run')
    @patch('os.listdir')
    def test_mount_geotools_success(self, mock_listdir, mock_run):
        """Test successful geotools mounting with mount_utils."""
        mock_run.return_value = MagicMock(returncode=0, stdout='', stderr='')
        mock_listdir.return_value = ['site1', 'site2']
        
        result = mount_utils.mount_geotools(self.test_geotools_path)
        
        self.assertTrue(result)
        mock_run.assert_called()
    
    @patch('subprocess.run')
    def test_mount_geotools_failure(self, mock_run):
        """Test failed geotools mounting with mount_utils."""
        mock_run.return_value = MagicMock(returncode=1, stdout='', stderr='Failed to mount')
        
        result = mount_utils.mount_geotools(self.test_geotools_path)
        
        self.assertFalse(result)
    
    @patch('subprocess.run')
    @patch('os.listdir')
    def test_mount_tide_success(self, mock_listdir, mock_run):
        """Test successful tide mounting with mount_utils."""
        mock_run.return_value = MagicMock(returncode=0, stdout='', stderr='')
        mock_listdir.return_value = ['fes2022b', 'region0']
        
        result = mount_utils.mount_tide(self.test_tide_path)
        
        self.assertTrue(result)
        # Verify the correct command with flags was called
        call_args = str(mock_run.call_args)
        self.assertIn('--implicit-dirs', call_args)
        self.assertIn('--dir-mode 777', call_args)
        self.assertIn('--file-mode 777', call_args)
    
    @patch('subprocess.run')
    def test_unmount_success(self, mock_run):
        """Test successful unmounting."""
        mock_run.return_value = MagicMock(returncode=0, stdout='', stderr='')
        
        result = mount_utils.unmount_geotools(self.test_geotools_path)
        
        self.assertTrue(result)
        mock_run.assert_called()
    
    @patch('subprocess.run')
    def test_unmount_not_mounted(self, mock_run):
        """Test unmounting when not mounted (should succeed)."""
        mock_run.return_value = MagicMock(returncode=1, stdout='', stderr='')
        
        result = mount_utils.unmount_geotools(self.test_geotools_path)
        
        self.assertTrue(result)  # Should return True for "not mounted"
    
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_status_mounted(self, mock_listdir, mock_exists):
        """Test status check when buckets are mounted."""
        mock_exists.return_value = True
        mock_listdir.return_value = ['item1', 'item2']
        
        # This should not raise an exception
        mount_utils.status()
    
    @patch('os.path.exists')
    def test_status_not_mounted(self, mock_exists):
        """Test status check when buckets are not mounted."""
        mock_exists.return_value = False
        
        # This should not raise an exception
        mount_utils.status()


class TestPipelineConfiguration(unittest.TestCase):
    """Test mounting-related configuration."""
    
    def test_default_mount_config(self):
        """Test that default configuration has correct mount settings."""
        config = PipelineConfig()
        
        mounting_config = config.config['mounting']
        
        # Verify required keys exist
        self.assertIn('enable_geotools', mounting_config)
        self.assertIn('enable_tide', mounting_config)
        self.assertIn('geotools_bucket', mounting_config)
        self.assertIn('geotools_mount', mounting_config)
        self.assertIn('tide_bucket', mounting_config)
        self.assertIn('tide_mount', mounting_config)
        
        # Verify default values
        self.assertEqual(mounting_config['geotools_bucket'], 'coastal_geotools_demo')
        self.assertEqual(mounting_config['tide_bucket'], 'aviso-fes2022')
        self.assertTrue(mounting_config['enable_geotools'])
        self.assertTrue(mounting_config['enable_tide'])


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete mounting system."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up integration test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('subprocess.run')
    @patch('os.listdir')
    def test_pipeline_mount_integration(self, mock_listdir, mock_run):
        """Test that pipeline mounting integrates correctly."""
        # Mock successful mounting
        mock_run.return_value = MagicMock(returncode=0, stdout='', stderr='')
        mock_listdir.return_value = ['item1', 'item2']
        
        # Create config with test paths
        config = PipelineConfig()
        config.config['site_name'] = 'test_site'
        config.config['mounting']['geotools_mount'] = os.path.join(self.temp_dir, 'geotools')
        config.config['mounting']['tide_mount'] = os.path.join(self.temp_dir, 'tide')
        
        # Test MountManager
        mount_manager = MountManager(config.config)
        result = mount_manager.mount_all_required()
        
        self.assertTrue(result)
        self.assertEqual(len(mount_manager.mounted_paths), 2)
        
        # Test cleanup
        mount_manager.unmount_all()
        self.assertEqual(len(mount_manager.mounted_paths), 0)


if __name__ == '__main__':
    # Configure test output
    unittest.main(verbosity=2)