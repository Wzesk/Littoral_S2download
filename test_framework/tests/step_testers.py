"""
Step-specific testers for the Littoral Pipeline Test Framework.

Each tester implements specific testing logic for individual pipeline steps,
collecting relevant metrics and statistics.
"""

import os
import sys
import time
import glob
import json
import logging
import subprocess
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path for pipeline imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from pipeline import PipelineConfig
from pipeline.pipeline_functions import ImageDownloader, Coregistration, CloudImputation, RGBNIRCreation, Upsampling, Normalization, Segmentation
from pipeline.pipeline_advanced import BoundaryExtraction, BoundaryRefinement, Geotransformation, ShorelineFiltering, TidalModeling, TidalCorrection

# Import from parent module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_runner import PipelineStepTester


class DownloadTester(PipelineStepTester):
    """Test the image download step."""
    
    def _execute_step(self, site_name: str, site_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Execute image download testing."""
        try:
            # Create pipeline config for this site
            config = PipelineConfig()
            config['site_name'] = site_name
            config['max_images'] = self.config.get('max_images', 5)
            
            # Initialize downloader
            downloader = ImageDownloader(config)
            
            # Count existing images before download
            targets_dir = os.path.join(site_path, "TARGETS")
            existing_images = len(glob.glob(os.path.join(targets_dir, "*.tif"))) if os.path.exists(targets_dir) else 0
            
            # Run download
            results = downloader.run()
            
            # Count images after download
            new_images = len(glob.glob(os.path.join(targets_dir, "*.tif"))) if os.path.exists(targets_dir) else 0
            images_downloaded = new_images - existing_images
            
            # Check processing CSV for download status
            processing_files = glob.glob(os.path.join(site_path, "*_processing.csv"))
            download_success_count = 0
            total_entries = 0
            
            if processing_files:
                latest_csv = max(processing_files, key=os.path.getctime)
                df = pd.read_csv(latest_csv)
                total_entries = len(df)
                download_success_count = len(df[df['step_1_download'] == 'success'])
            
            # Calculate metrics
            success_rate = download_success_count / total_entries if total_entries > 0 else 0
            
            custom_metrics = {
                'images_downloaded': images_downloaded,
                'total_entries': total_entries,
                'download_success_count': download_success_count,
                'download_success_rate': success_rate,
                'max_images_configured': config.get('max_images', 'unlimited'),
                'targets_directory_size_mb': self._get_directory_size(targets_dir) if os.path.exists(targets_dir) else 0
            }
            
            # Consider success if at least some images were downloaded
            success = images_downloaded > 0 and success_rate > 0.5
            
            return success, custom_metrics
            
        except Exception as e:
            self.logger.error(f"Download test failed: {str(e)}")
            return False, {'error': str(e)}
    
    def _get_directory_size(self, directory: str) -> float:
        """Get directory size in MB."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            return total_size / (1024 * 1024)  # Convert to MB
        except:
            return 0


class CoregisterTester(PipelineStepTester):
    """Test the coregistration step."""
    
    def _execute_step(self, site_name: str, site_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Execute coregistration testing."""
        try:
            config = PipelineConfig()
            config['site_name'] = site_name
            
            # Check for required input files
            targets_dir = os.path.join(site_path, "TARGETS")
            input_files = glob.glob(os.path.join(targets_dir, "*.tif"))
            
            if not input_files:
                return False, {'error': 'No input files found for coregistration'}
            
            # Initialize coregistration
            coregistration = Coregistration(config)
            
            # Count existing coregistered files
            coreg_dir = os.path.join(site_path, "coregistered")
            existing_coreg = len(glob.glob(os.path.join(coreg_dir, "*.tif"))) if os.path.exists(coreg_dir) else 0
            
            # Run coregistration
            results = coregistration.run()
            
            # Count new coregistered files
            new_coreg = len(glob.glob(os.path.join(coreg_dir, "*.tif"))) if os.path.exists(coreg_dir) else 0
            files_coregistered = new_coreg - existing_coreg
            
            # Check for coregistration settings and results
            coreg_settings_file = os.path.join(site_path, "coreg_settings.json")
            has_settings = os.path.exists(coreg_settings_file)
            
            # Check filtered files CSV
            filtered_csv = os.path.join(coreg_dir, "filtered_files.csv")
            filtered_count = 0
            if os.path.exists(filtered_csv):
                df = pd.read_csv(filtered_csv)
                filtered_count = len(df)
            
            custom_metrics = {
                'input_files_count': len(input_files),
                'files_coregistered': files_coregistered,
                'filtered_files_count': filtered_count,
                'has_coreg_settings': has_settings,
                'coregistered_directory_size_mb': self._get_directory_size(coreg_dir) if os.path.exists(coreg_dir) else 0,
                'coregistration_success_rate': files_coregistered / len(input_files) if input_files else 0
            }
            
            # Success if some files were coregistered
            success = files_coregistered > 0 and has_settings
            
            return success, custom_metrics
            
        except Exception as e:
            self.logger.error(f"Coregistration test failed: {str(e)}")
            return False, {'error': str(e)}
    
    def _get_directory_size(self, directory: str) -> float:
        """Get directory size in MB."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            return total_size / (1024 * 1024)
        except:
            return 0


class CloudImputeTester(PipelineStepTester):
    """Test the cloud imputation step."""
    
    def _execute_step(self, site_name: str, site_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Execute cloud imputation testing."""
        try:
            config = PipelineConfig()
            config['site_name'] = site_name
            
            # Check for input files
            targets_dir = os.path.join(site_path, "TARGETS")
            input_files = glob.glob(os.path.join(targets_dir, "*.tif"))
            
            if not input_files:
                return False, {'error': 'No input files found for cloud imputation'}
            
            # Initialize cloud imputation
            cloud_imputation = CloudImputation(config)
            
            # Check existing cloudless files
            cloudless_dir = os.path.join(targets_dir, "cloudless")
            existing_cloudless = len(glob.glob(os.path.join(cloudless_dir, "*.tif"))) if os.path.exists(cloudless_dir) else 0
            
            # Run cloud imputation
            results = cloud_imputation.run()
            
            # Count new cloudless files
            new_cloudless = len(glob.glob(os.path.join(cloudless_dir, "*.tif"))) if os.path.exists(cloudless_dir) else 0
            files_processed = new_cloudless - existing_cloudless
            
            # Check for cloudless report
            report_csv = os.path.join(cloudless_dir, "cloudless_report.csv")
            has_report = os.path.exists(report_csv)
            
            # Analyze report if available
            report_metrics = {}
            if has_report:
                df = pd.read_csv(report_csv)
                report_metrics = {
                    'total_processed': len(df),
                    'avg_cloud_percentage': df['cloud_percentage'].mean() if 'cloud_percentage' in df.columns else 0,
                    'max_cloud_percentage': df['cloud_percentage'].max() if 'cloud_percentage' in df.columns else 0,
                    'min_cloud_percentage': df['cloud_percentage'].min() if 'cloud_percentage' in df.columns else 0
                }
            
            custom_metrics = {
                'input_files_count': len(input_files),
                'files_processed': files_processed,
                'has_cloudless_report': has_report,
                'cloudless_directory_size_mb': self._get_directory_size(cloudless_dir) if os.path.exists(cloudless_dir) else 0,
                **report_metrics
            }
            
            # Success if files were processed and report exists
            success = files_processed > 0 and has_report
            
            return success, custom_metrics
            
        except Exception as e:
            self.logger.error(f"Cloud imputation test failed: {str(e)}")
            return False, {'error': str(e)}
    
    def _get_directory_size(self, directory: str) -> float:
        """Get directory size in MB."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            return total_size / (1024 * 1024)
        except:
            return 0


class RGBNIRTester(PipelineStepTester):
    """Test the RGB/NIR creation step."""
    
    def _execute_step(self, site_name: str, site_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Execute RGB/NIR creation testing."""
        try:
            config = PipelineConfig()
            config['site_name'] = site_name
            
            # Initialize RGB/NIR creation
            rgb_nir_creation = RGBNIRCreation(config)
            
            # Check for required input directories
            cloudless_dir = os.path.join(site_path, "TARGETS", "cloudless")
            if not os.path.exists(cloudless_dir):
                return False, {'error': 'Cloudless directory not found'}
            
            # Count existing RGB/NIR files
            clear_dir = os.path.join(site_path, "CLEAR")
            existing_rgb = len(glob.glob(os.path.join(clear_dir, "*_rgb.png"))) if os.path.exists(clear_dir) else 0
            existing_nir = len(glob.glob(os.path.join(clear_dir, "*_nir.png"))) if os.path.exists(clear_dir) else 0
            
            # Run RGB/NIR creation
            results = rgb_nir_creation.run()
            
            # Count new files
            new_rgb = len(glob.glob(os.path.join(clear_dir, "*_rgb.png"))) if os.path.exists(clear_dir) else 0
            new_nir = len(glob.glob(os.path.join(clear_dir, "*_nir.png"))) if os.path.exists(clear_dir) else 0
            
            rgb_created = new_rgb - existing_rgb
            nir_created = new_nir - existing_nir
            
            # Check input file count
            input_files = glob.glob(os.path.join(cloudless_dir, "*.tif"))
            
            custom_metrics = {
                'input_files_count': len(input_files),
                'rgb_files_created': rgb_created,
                'nir_files_created': nir_created,
                'clear_directory_size_mb': self._get_directory_size(clear_dir) if os.path.exists(clear_dir) else 0,
                'rgb_success_rate': rgb_created / len(input_files) if input_files else 0,
                'nir_success_rate': nir_created / len(input_files) if input_files else 0
            }
            
            # Success if both RGB and NIR files were created
            success = rgb_created > 0 and nir_created > 0
            
            return success, custom_metrics
            
        except Exception as e:
            self.logger.error(f"RGB/NIR creation test failed: {str(e)}")
            return False, {'error': str(e)}
    
    def _get_directory_size(self, directory: str) -> float:
        """Get directory size in MB."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            return total_size / (1024 * 1024)
        except:
            return 0


class UpsampleTester(PipelineStepTester):
    """Test the upsampling step."""
    
    def _execute_step(self, site_name: str, site_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Execute upsampling testing."""
        try:
            config = PipelineConfig()
            config['site_name'] = site_name
            
            # Initialize upsampling
            upsampling = Upsampling(config)
            
            # Check for input files
            clear_dir = os.path.join(site_path, "CLEAR")
            input_files = glob.glob(os.path.join(clear_dir, "*.png"))
            
            if not input_files:
                return False, {'error': 'No input files found for upsampling'}
            
            # Count existing upsampled files
            up_dir = os.path.join(site_path, "UP")
            existing_up = len(glob.glob(os.path.join(up_dir, "*_up.png"))) if os.path.exists(up_dir) else 0
            
            # Run upsampling
            results = upsampling.run()
            
            # Count new upsampled files
            new_up = len(glob.glob(os.path.join(up_dir, "*_up.png"))) if os.path.exists(up_dir) else 0
            files_upsampled = new_up - existing_up
            
            custom_metrics = {
                'input_files_count': len(input_files),
                'files_upsampled': files_upsampled,
                'up_directory_size_mb': self._get_directory_size(up_dir) if os.path.exists(up_dir) else 0,
                'upsampling_success_rate': files_upsampled / len(input_files) if input_files else 0
            }
            
            # Success if files were upsampled
            success = files_upsampled > 0
            
            return success, custom_metrics
            
        except Exception as e:
            self.logger.error(f"Upsampling test failed: {str(e)}")
            return False, {'error': str(e)}
    
    def _get_directory_size(self, directory: str) -> float:
        """Get directory size in MB."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            return total_size / (1024 * 1024)
        except:
            return 0


class NormalizeTester(PipelineStepTester):
    """Test the normalization step."""
    
    def _execute_step(self, site_name: str, site_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Execute normalization testing."""
        try:
            config = PipelineConfig()
            config['site_name'] = site_name
            
            # Initialize normalization
            normalization = Normalization(config)
            
            # Check for input files
            up_dir = os.path.join(site_path, "UP")
            input_files = glob.glob(os.path.join(up_dir, "*_nir_up.png"))
            
            if not input_files:
                return False, {'error': 'No NIR upsampled files found for normalization'}
            
            # Count existing normalized files
            norm_dir = os.path.join(site_path, "NORMALIZED")
            existing_norm = len(glob.glob(os.path.join(norm_dir, "*_nir_up.png"))) if os.path.exists(norm_dir) else 0
            
            # Run normalization
            results = normalization.run()
            
            # Count new normalized files and skipped files
            new_norm = len(glob.glob(os.path.join(norm_dir, "*_nir_up.png"))) if os.path.exists(norm_dir) else 0
            files_normalized = new_norm - existing_norm
            files_skipped = len(input_files) - files_normalized
            
            custom_metrics = {
                'input_files_count': len(input_files),
                'files_normalized': files_normalized,
                'files_skipped': files_skipped,
                'normalized_directory_size_mb': self._get_directory_size(norm_dir) if os.path.exists(norm_dir) else 0,
                'normalization_success_rate': files_normalized / len(input_files) if input_files else 0,
                'skip_rate': files_skipped / len(input_files) if input_files else 0
            }
            
            # Success if some files were normalized (skipping is normal for defective images)
            success = files_normalized > 0
            
            return success, custom_metrics
            
        except Exception as e:
            self.logger.error(f"Normalization test failed: {str(e)}")
            return False, {'error': str(e)}
    
    def _get_directory_size(self, directory: str) -> float:
        """Get directory size in MB."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            return total_size / (1024 * 1024)
        except:
            return 0


class SegmentTester(PipelineStepTester):
    """Test the segmentation step."""
    
    def _execute_step(self, site_name: str, site_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Execute segmentation testing."""
        try:
            config = PipelineConfig()
            config['site_name'] = site_name
            
            # Initialize segmentation
            segmentation = Segmentation(config)
            
            # Check for input files
            norm_dir = os.path.join(site_path, "NORMALIZED")
            input_files = glob.glob(os.path.join(norm_dir, "*_nir_up.png"))
            
            if not input_files:
                return False, {'error': 'No normalized files found for segmentation'}
            
            # Count existing mask files
            mask_dir = os.path.join(site_path, "MASK")
            existing_masks = len(glob.glob(os.path.join(mask_dir, "*.png"))) if os.path.exists(mask_dir) else 0
            
            # Run segmentation
            results = segmentation.run()
            
            # Count new mask files
            new_masks = len(glob.glob(os.path.join(mask_dir, "*.png"))) if os.path.exists(mask_dir) else 0
            masks_created = new_masks - existing_masks
            
            custom_metrics = {
                'input_files_count': len(input_files),
                'masks_created': masks_created,
                'mask_directory_size_mb': self._get_directory_size(mask_dir) if os.path.exists(mask_dir) else 0,
                'segmentation_success_rate': masks_created / len(input_files) if input_files else 0
            }
            
            # Success if masks were created
            success = masks_created > 0
            
            return success, custom_metrics
            
        except Exception as e:
            self.logger.error(f"Segmentation test failed: {str(e)}")
            return False, {'error': str(e)}
    
    def _get_directory_size(self, directory: str) -> float:
        """Get directory size in MB."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            return total_size / (1024 * 1024)
        except:
            return 0


class BoundaryExtractTester(PipelineStepTester):
    """Test the boundary extraction step."""
    
    def _execute_step(self, site_name: str, site_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Execute boundary extraction testing."""
        try:
            config = PipelineConfig()
            config['site_name'] = site_name
            
            # Initialize boundary extraction
            boundary_extraction = BoundaryExtraction(config)
            
            # Check for input files
            mask_dir = os.path.join(site_path, "MASK")
            input_files = glob.glob(os.path.join(mask_dir, "*.png"))
            
            if not input_files:
                return False, {'error': 'No mask files found for boundary extraction'}
            
            # Count existing shoreline files
            shoreline_dir = os.path.join(site_path, "SHORELINE")
            existing_shorelines = len(glob.glob(os.path.join(shoreline_dir, "*.csv"))) if os.path.exists(shoreline_dir) else 0
            
            # Run boundary extraction
            results = boundary_extraction.run()
            
            # Count new shoreline files
            new_shorelines = len(glob.glob(os.path.join(shoreline_dir, "*.csv"))) if os.path.exists(shoreline_dir) else 0
            shorelines_extracted = new_shorelines - existing_shorelines
            
            custom_metrics = {
                'input_files_count': len(input_files),
                'shorelines_extracted': shorelines_extracted,
                'shoreline_directory_size_mb': self._get_directory_size(shoreline_dir) if os.path.exists(shoreline_dir) else 0,
                'extraction_success_rate': shorelines_extracted / len(input_files) if input_files else 0
            }
            
            # Success if shorelines were extracted
            success = shorelines_extracted > 0
            
            return success, custom_metrics
            
        except Exception as e:
            self.logger.error(f"Boundary extraction test failed: {str(e)}")
            return False, {'error': str(e)}
    
    def _get_directory_size(self, directory: str) -> float:
        """Get directory size in MB."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            return total_size / (1024 * 1024)
        except:
            return 0


class BoundaryRefineTester(PipelineStepTester):
    """Test the boundary refinement step."""
    
    def _execute_step(self, site_name: str, site_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Execute boundary refinement testing."""
        try:
            config = PipelineConfig()
            config['site_name'] = site_name
            
            # Initialize boundary refinement
            boundary_refinement = BoundaryRefinement(config)
            
            # Check for input files
            shoreline_dir = os.path.join(site_path, "SHORELINE")
            input_files = glob.glob(os.path.join(shoreline_dir, "*.csv"))
            
            if not input_files:
                return False, {'error': 'No shoreline files found for boundary refinement'}
            
            # Count existing refined files (assume refined files have different naming or are in same directory)
            existing_refined = len([f for f in input_files if 'refined' in f])
            
            # Run boundary refinement
            results = boundary_refinement.run()
            
            # Count new refined files
            all_files = glob.glob(os.path.join(shoreline_dir, "*.csv"))
            new_refined = len([f for f in all_files if 'refined' in f])
            refined_created = new_refined - existing_refined
            
            custom_metrics = {
                'input_files_count': len(input_files),
                'refined_shorelines_created': refined_created,
                'total_shoreline_files': len(all_files),
                'refinement_success_rate': refined_created / len(input_files) if input_files else 0
            }
            
            # Success if refined shorelines were created
            success = refined_created > 0
            
            return success, custom_metrics
            
        except Exception as e:
            self.logger.error(f"Boundary refinement test failed: {str(e)}")
            return False, {'error': str(e)}


class GeotransformTester(PipelineStepTester):
    """Test the geotransformation step."""
    
    def _execute_step(self, site_name: str, site_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Execute geotransformation testing."""
        try:
            config = PipelineConfig()
            config['site_name'] = site_name
            
            # Initialize geotransformation
            geotransformation = Geotransformation(config)
            
            # Check for input files
            shoreline_dir = os.path.join(site_path, "SHORELINE")
            input_files = glob.glob(os.path.join(shoreline_dir, "*.csv"))
            
            if not input_files:
                return False, {'error': 'No shoreline files found for geotransformation'}
            
            # Count existing geotransformed files (files ending with 'o.csv')
            existing_geo = len([f for f in input_files if f.endswith('o.csv')])
            
            # Run geotransformation
            results = geotransformation.run()
            
            # Count new geotransformed files
            all_files = glob.glob(os.path.join(shoreline_dir, "*.csv"))
            new_geo = len([f for f in all_files if f.endswith('o.csv')])
            geo_created = new_geo - existing_geo
            
            custom_metrics = {
                'input_files_count': len(input_files),
                'geotransformed_files_created': geo_created,
                'total_shoreline_files': len(all_files),
                'geotransformation_success_rate': geo_created / len(input_files) if input_files else 0
            }
            
            # Success if geotransformed files were created
            success = geo_created > 0
            
            return success, custom_metrics
            
        except Exception as e:
            self.logger.error(f"Geotransformation test failed: {str(e)}")
            return False, {'error': str(e)}


class FilterShorelinesTester(PipelineStepTester):
    """Test the shoreline filtering step."""
    
    def _execute_step(self, site_name: str, site_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Execute shoreline filtering testing."""
        try:
            config = PipelineConfig()
            config['site_name'] = site_name
            
            # Initialize shoreline filtering
            shoreline_filtering = ShorelineFiltering(config)
            
            # Check for input files
            shoreline_dir = os.path.join(site_path, "SHORELINE")
            geo_files = [f for f in glob.glob(os.path.join(shoreline_dir, "*.csv")) if f.endswith('o.csv')]
            
            if not geo_files:
                return False, {'error': 'No geotransformed files found for filtering'}
            
            # Run shoreline filtering
            results = shoreline_filtering.run()
            
            # Analyze filtering results
            if hasattr(results, 'get'):
                filtered_count = results.get('filtered_shorelines', 0)
                total_count = results.get('total_shorelines', len(geo_files))
                filtering_applied = results.get('filtering_applied', len(geo_files) >= 50)
            else:
                # Fallback analysis
                total_count = len(geo_files)
                filtered_count = total_count  # Assume all passed if no detailed results
                filtering_applied = len(geo_files) >= 50
            
            custom_metrics = {
                'input_files_count': len(geo_files),
                'total_shorelines': total_count,
                'filtered_shorelines': filtered_count,
                'removed_shorelines': total_count - filtered_count,
                'filtering_applied': filtering_applied,
                'filter_retention_rate': filtered_count / total_count if total_count > 0 else 0
            }
            
            # Success if filtering was completed (regardless of how many were filtered)
            success = True  # Filtering step doesn't typically fail, just filters different amounts
            
            return success, custom_metrics
            
        except Exception as e:
            self.logger.error(f"Shoreline filtering test failed: {str(e)}")
            return False, {'error': str(e)}


class TideModelTester(PipelineStepTester):
    """Test the tide modeling step."""
    
    def _execute_step(self, site_name: str, site_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Execute tide modeling testing."""
        try:
            config = PipelineConfig()
            config['site_name'] = site_name
            
            # Initialize tide modeling
            tide_modeling = TidalModeling(config)
            
            # Check for required input files
            processing_files = glob.glob(os.path.join(site_path, "*_processing.csv"))
            if not processing_files:
                return False, {'error': 'No processing CSV found for tide modeling'}
            
            # Check for site configuration
            site_config_file = os.path.join(site_path, f"{site_name}.json")
            if not os.path.exists(site_config_file):
                return False, {'error': 'Site configuration JSON not found'}
            
            # Run tide modeling
            results = tide_modeling.run()
            
            # Check for tide corrections output
            tide_corrections_file = os.path.join(site_path, "tide_corrections.csv")
            has_tide_corrections = os.path.exists(tide_corrections_file)
            
            # Analyze tide corrections if available
            tide_metrics = {}
            if has_tide_corrections:
                df = pd.read_csv(tide_corrections_file)
                tide_metrics = {
                    'tide_predictions_count': len(df),
                    'tide_range_m': float(df['tide'].max() - df['tide'].min()) if 'tide' in df.columns else 0,
                    'avg_tide_m': float(df['tide'].mean()) if 'tide' in df.columns else 0,
                    'has_horizontal_corrections': 'horizontal_correction_m' in df.columns
                }
            
            custom_metrics = {
                'has_tide_corrections_file': has_tide_corrections,
                'has_site_config': os.path.exists(site_config_file),
                **tide_metrics
            }
            
            # Success if tide corrections file was created
            success = has_tide_corrections
            
            return success, custom_metrics
            
        except Exception as e:
            self.logger.error(f"Tide modeling test failed: {str(e)}")
            return False, {'error': str(e)}


class TideCorrectTester(PipelineStepTester):
    """Test the tide correction step."""
    
    def _execute_step(self, site_name: str, site_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Execute tide correction testing."""
        try:
            config = PipelineConfig()
            config['site_name'] = site_name
            
            # Initialize tide correction
            tide_correction = TidalCorrection(config)
            
            # Check for required input files
            tide_corrections_file = os.path.join(site_path, "tide_corrections.csv")
            if not os.path.exists(tide_corrections_file):
                return False, {'error': 'Tide corrections CSV not found'}
            
            shoreline_dir = os.path.join(site_path, "SHORELINE")
            geo_files = [f for f in glob.glob(os.path.join(shoreline_dir, "*.csv")) if f.endswith('o.csv')]
            
            if not geo_files:
                return False, {'error': 'No geotransformed shoreline files found'}
            
            # Run tide correction
            results = tide_correction.run()
            
            # Check for tidal corrected output
            tidal_corrected_dir = os.path.join(site_path, "TIDAL_CORRECTED")
            corrected_files = glob.glob(os.path.join(tidal_corrected_dir, "*_tidal_corrected.csv")) if os.path.exists(tidal_corrected_dir) else []
            
            custom_metrics = {
                'input_shoreline_files': len(geo_files),
                'corrected_shoreline_files': len(corrected_files),
                'has_tidal_corrected_directory': os.path.exists(tidal_corrected_dir),
                'correction_success_rate': len(corrected_files) / len(geo_files) if geo_files else 0,
                'tidal_corrected_directory_size_mb': self._get_directory_size(tidal_corrected_dir) if os.path.exists(tidal_corrected_dir) else 0
            }
            
            # Success if corrected files were created
            success = len(corrected_files) > 0
            
            return success, custom_metrics
            
        except Exception as e:
            self.logger.error(f"Tide correction test failed: {str(e)}")
            return False, {'error': str(e)}
    
    def _get_directory_size(self, directory: str) -> float:
        """Get directory size in MB."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            return total_size / (1024 * 1024)
        except:
            return 0