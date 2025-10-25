"""
Configuration management for the Littoral Pipeline.

This module handles all configuration settings, paths, and parameters
needed to run the littoral processing pipeline.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd


class PipelineConfig:
    """Configuration manager for the littoral pipeline."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize pipeline configuration.
        
        Args:
            config_file: Path to configuration file (JSON or YAML)
        """
        self.config = self._load_default_config()
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration settings."""
        return {
            # Basic paths and settings
            "site_table_path": "/home/walter_littor_al/Littoral_S2download/littoral_sites.csv",
            "save_path": "/home/walter_littor_al/geotools_sites",
            "site_name": None,  # Must be specified
            
            # Folder structure
            "folders": {
                "tiff": "TARGETS",
                "upsampled": "UP",
                "normalized": "NORMALIZED",
                "masked": "MASK",
                "nir": "RAWNIR",
                "rgb": "RAWRGB",
                "clear_output": "CLEAR",
                "clear_tiff": "/TARGETS/cloudless",
                "shoreline": "SHORELINE",
                "filtered_shorelines": "FILTERED_SHORELINES",
                "tidal_corrected": "TIDAL_CORRECTED"
            },
            
            # Processing parameters
            "processing": {
                "coregistration": {
                    "cleanup": True
                },
                "normalization": {
                    "min_threshold": 25,
                    "max_threshold": 230
                },
                "segmentation": {
                    "model_path": "/home/walter_littor_al/littoral_segment/seg_models/yolo8_params"
                },
                "filtering": {
                    # Global filtering controls
                    "enable_shoreline_filtering": True,  # Enable/disable ShorelineFiltering step (step 12)
                    "enable_boundary_filtering": True,   # Enable/disable filtering in BoundaryRefinement step (step 10)
                    
                    # ShorelineFiltering (step 12) parameters
                    "cluster_eps": 0.05,
                    "cluster_min_samples": 10,
                    "defective_threshold": 50,  # Apply defective filtering if >50 shorelines
                    "defective_point_ratio": 0.1,  # 10% defective points threshold
                    "distance_multiplier": 5,  # Distance threshold multiplier
                    
                    # BoundaryRefinement filtering (step 10) parameters  
                    "boundary_filter_threshold": 50,  # Apply boundary filtering if >50 shorelines
                    "iqr_multiplier": 2.0,  # Multiplier for IQR-based outlier detection
                    "location_eps": 1.5,  # DBSCAN epsilon for location clustering
                    "min_points": 3,  # Minimum points required for valid shoreline
                    "enable_shape_filter": True,  # Enable/disable shape similarity filter
                    "shape_similarity_threshold": 0.3,  # Minimum shape similarity
                    "boundary_filter_verbose": True  # Print progress information
                },
                "tide_correction": {
                    "model": "fes2022b",
                    "tide_model_dir": "/home/walter_littor_al/tide_model",
                    "reference_elevation": 0,
                    "beach_slope": 0.08,
                    "epsg": 4326,
                    "method": "bilinear",
                    "extrapolate": True,
                    "cutoff": 10.0
                }
            },
            
            # System paths
            "system_paths": {
                "littoral_src": "/home/walter_littor_al/Littoral_S2download/src",
                "coreg_src": "/home/walter_littor_al/littoral_coreg/src",
                "cloud_impute_src": "/home/walter_littor_al/littoral_cloud_impute",
                "real_esrgan_src": "/home/walter_littor_al/Real-ESRGAN",
                "segment_src": "/home/walter_littor_al/littoral_segment",
                "refine_src": "/home/walter_littor_al",
                "proj_data": "/opt/conda/envs/littoral_pipeline/share/proj"
            },
            
            # Mount settings - Mounting is now MANDATORY for pipeline execution
            "mounting": {
                "enable_geotools": True,        # Enable/disable geotools bucket mounting
                "enable_tide": True,            # Enable/disable tide bucket mounting
                "geotools_bucket": "coastal_geotools_demo",
                "geotools_mount": "/home/walter_littor_al/geotools_sites",
                "tide_bucket": "aviso-fes2022",
                "tide_mount": "/home/walter_littor_al/tide_model"
            },
            
            # Pipeline control
            "pipeline": {
                "run_mode": "full",  # 'full' or 'update'
                "skip_steps": [],  # Steps to skip: ['download', 'coregister', etc.]
                "max_images": None,  # Limit number of images to process
                "log_level": "INFO"
            }
        }
    
    def load_config(self, config_file: str):
        """
        Load configuration from file.
        
        Args:
            config_file: Path to configuration file (JSON or YAML)
        """
        with open(config_file, 'r') as f:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                file_config = yaml.safe_load(f)
            else:
                file_config = json.load(f)
        
        # Deep merge with default config
        self._deep_merge(self.config, file_config)
    
    def save_config(self, config_file: str):
        """
        Save current configuration to file.
        
        Args:
            config_file: Path to save configuration
        """
        with open(config_file, 'w') as f:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            else:
                json.dump(self.config, f, indent=2)
    
    def _deep_merge(self, base_dict: Dict, update_dict: Dict):
        """Deep merge two dictionaries."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get_site_path(self) -> str:
        """Get the full path to the site directory."""
        if not self.config['site_name']:
            raise ValueError("Site name must be specified in configuration")
        return os.path.join(self.config['save_path'], self.config['site_name'])
    
    def get_folder_path(self, folder_key: str) -> str:
        """
        Get the full path to a specific folder.
        
        Args:
            folder_key: Key from the folders configuration
            
        Returns:
            Full path to the folder
        """
        if folder_key not in self.config['folders']:
            raise ValueError(f"Unknown folder key: {folder_key}")
        
        folder_name = self.config['folders'][folder_key]
        if folder_name.startswith('/'):
            # Absolute path from site root
            return os.path.join(self.get_site_path(), folder_name.lstrip('/'))
        else:
            # Relative path from site root
            return os.path.join(self.get_site_path(), folder_name)
    
    def get_processing_filename(self) -> Optional[str]:
        """Get the most recent processing CSV filename."""
        import glob
        site_path = self.get_site_path()
        pattern = os.path.join(site_path, "*_processing.csv")
        files = glob.glob(pattern)
        if files:
            return os.path.basename(max(files, key=os.path.getctime))
        return None
    
    def create_processing_filename(self) -> str:
        """Create a new processing filename with current date."""
        current_date = datetime.now().strftime("%Y%m%d")
        return f"{current_date}_processing.csv"
    
    def get_existing_images(self) -> List[str]:
        """
        Get list of already processed images for update mode.
        
        Returns:
            List of image names that have been processed
        """
        processing_file = self.get_processing_filename()
        if not processing_file:
            return []
        
        try:
            df = pd.read_csv(os.path.join(self.get_site_path(), processing_file))
            if 'name' in df.columns:
                return df['name'].tolist()
        except Exception:
            pass
        
        return []
    
    def validate_config(self) -> List[str]:
        """
        Validate configuration and return list of any issues.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Check required fields
        if not self.config.get('site_name'):
            errors.append("Site name must be specified")
        
        if not self.config.get('site_table_path'):
            errors.append("Site table path must be specified")
        
        if not os.path.exists(self.config['site_table_path']):
            errors.append(f"Site table file not found: {self.config['site_table_path']}")
        
        # Check system paths
        for path_key, path_value in self.config['system_paths'].items():
            if not os.path.exists(path_value):
                errors.append(f"System path not found ({path_key}): {path_value}")
        
        # Check if site exists in site table
        if os.path.exists(self.config['site_table_path']):
            try:
                df = pd.read_csv(self.config['site_table_path'])
                if 'site_name' in df.columns:
                    available_sites = df['site_name'].tolist()
                    if self.config['site_name'] not in available_sites:
                        errors.append(f"Site '{self.config['site_name']}' not found in site table. Available sites: {available_sites}")
            except Exception as e:
                errors.append(f"Error reading site table: {e}")
        
        return errors
    
    def __getitem__(self, key):
        """Allow dictionary-style access to config."""
        return self.config[key]
    
    def __setitem__(self, key, value):
        """Allow dictionary-style setting of config."""
        self.config[key] = value
    
    def get(self, key, default=None):
        """Get configuration value with default."""
        return self.config.get(key, default)


def create_example_config(output_file: str = "pipeline_config.yaml"):
    """
    Create an example configuration file.
    
    Args:
        output_file: Path where to save the example config
    """
    config = PipelineConfig()
    
    # Set example values
    config['site_name'] = "example_site"
    config['pipeline']['run_mode'] = "full"
    config['pipeline']['log_level'] = "INFO"
    
    config.save_config(output_file)
    print(f"Example configuration saved to: {output_file}")


if __name__ == "__main__":
    # Create example configuration
    create_example_config()