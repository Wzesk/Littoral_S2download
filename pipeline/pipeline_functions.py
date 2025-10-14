"""
Pipeline processing functions for the Littoral Processing Pipeline.

This module contains all the individual processing steps extracted from the
notebook, organized into clean functions with proper error handling and logging.
"""

import os
import sys
import logging
import glob
import json
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Import optional dependencies conditionally
try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import cv2
except ImportError:
    cv2 = None

# Configure matplotlib to suppress font warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*font.*')
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 10,
    'axes.unicode_minus': False,
    'figure.max_open_warning': 0
})

# Set up logging
logger = logging.getLogger(__name__)


class PipelineStep:
    """Base class for pipeline steps with common functionality."""
    
    def __init__(self, config, step_name: str):
        self.config = config
        self.step_name = step_name
        self.logger = logging.getLogger(f"{__name__}.{step_name}")
    
    def update_status(self, file_statuses: Optional[Dict] = None, metrics: Optional[Dict] = None):
        """Update processing status in the tracking CSV."""
        try:
            processing_filename = self.config.get_processing_filename()
            if not processing_filename:
                self.logger.warning("No processing file found for status update")
                return
            
            csv_path = os.path.join(self.config.get_site_path(), processing_filename)
            if not os.path.exists(csv_path):
                self.logger.warning(f"Processing file not found: {csv_path}")
                return
            
            df = pd.read_csv(csv_path)
            
            if file_statuses:
                # Update individual file statuses
                for img_name, status in file_statuses.items():
                    mask = df['name'] == img_name
                    df.loc[mask, self.step_name] = status
            else:
                # Update all files with same status
                df[self.step_name] = 'success'
            
            # Add metrics as new columns if provided
            if metrics:
                for metric_name, metric_value in metrics.items():
                    df[metric_name] = metric_value
            
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Updated {self.step_name} status in {processing_filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to update status: {e}")


class MountManager:
    """Manages mounting and unmounting of cloud storage buckets."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.MountManager")
        self.mounted_paths = []
    
    def _unmount_if_mounted(self, mount_path: str):
        """Unmount a path if it's currently mounted."""
        try:
            # Always try to unmount first to ensure clean state
            cmd = f"fusermount -u {mount_path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info(f"Unmounted existing mount at {mount_path}")
            # Remove from tracked paths if present
            if mount_path in self.mounted_paths:
                self.mounted_paths.remove(mount_path)
        except Exception as e:
            self.logger.debug(f"Unmount attempt for {mount_path}: {e}")
    
    def mount_geotools_bucket(self) -> bool:
        """Mount the geotools bucket."""
        try:
            mount_path = self.config['mounting']['geotools_mount']
            bucket = self.config['mounting']['geotools_bucket']
            
            self.logger.info(f"Mounting geotools bucket: {bucket} -> {mount_path}")
            
            # Unmount if already mounted
            self._unmount_if_mounted(mount_path)
            
            # Create mount directory
            os.makedirs(mount_path, exist_ok=True)
            self.logger.debug(f"Mount directory created/verified: {mount_path}")
            
            # Mount bucket
            cmd = f"gcsfuse {bucket} {mount_path}"
            self.logger.debug(f"Executing command: {cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.mounted_paths.append(mount_path)
                self.logger.info(f"Successfully mounted {bucket} to {mount_path}")
                
                # Verify mount is working
                try:
                    contents = os.listdir(mount_path)
                    self.logger.debug(f"Mount verification: {len(contents)} items accessible")
                except Exception as e:
                    self.logger.warning(f"Mount appears successful but cannot list contents: {e}")
                
                return True
            else:
                self.logger.error(f"Failed to mount geotools bucket")
                self.logger.error(f"Command: {cmd}")
                self.logger.error(f"Return code: {result.returncode}")
                self.logger.error(f"STDOUT: {result.stdout.strip() if result.stdout else '(empty)'}")
                self.logger.error(f"STDERR: {result.stderr.strip() if result.stderr else '(empty)'}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error mounting geotools bucket: {e}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def mount_tide_bucket(self) -> bool:
        """Mount the tide model bucket."""
        try:
            mount_path = self.config['mounting']['tide_mount']
            bucket = self.config['mounting']['tide_bucket']
            
            self.logger.info(f"Mounting tide bucket: {bucket} -> {mount_path}")
            
            # Unmount if already mounted
            self._unmount_if_mounted(mount_path)
            
            # Create mount directory
            os.makedirs(mount_path, exist_ok=True)
            self.logger.debug(f"Mount directory created/verified: {mount_path}")
            
            # Mount bucket with special options for AVISO
            cmd = f"gcsfuse --implicit-dirs --dir-mode 777 --file-mode 777 {bucket} {mount_path}"
            self.logger.debug(f"Executing command: {cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.mounted_paths.append(mount_path)
                self.logger.info(f"Successfully mounted {bucket} to {mount_path}")
                
                # Verify mount is working
                try:
                    contents = os.listdir(mount_path)
                    self.logger.debug(f"Mount verification: {len(contents)} items accessible")
                except Exception as e:
                    self.logger.warning(f"Mount appears successful but cannot list contents: {e}")
                
                return True
            else:
                self.logger.error(f"Failed to mount tide bucket")
                self.logger.error(f"Command: {cmd}")
                self.logger.error(f"Return code: {result.returncode}")
                self.logger.error(f"STDOUT: {result.stdout.strip() if result.stdout else '(empty)'}")
                self.logger.error(f"STDERR: {result.stderr.strip() if result.stderr else '(empty)'}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error mounting tide bucket: {e}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def mount_all_required(self) -> bool:
        """Mount all required buckets. Returns True only if ALL succeed."""
        success_count = 0
        total_mounts = 0
        
        # Mount geotools bucket if enabled
        if self.config['mounting']['enable_geotools']:
            total_mounts += 1
            if self.mount_geotools_bucket():
                success_count += 1
            else:
                self.logger.error("CRITICAL: Geotools bucket mounting failed - pipeline cannot proceed")
        
        # Mount tide bucket if enabled  
        if self.config['mounting']['enable_tide']:
            total_mounts += 1
            if self.mount_tide_bucket():
                success_count += 1
            else:
                self.logger.error("CRITICAL: Tide bucket mounting failed - pipeline cannot proceed")
        
        if success_count == total_mounts and total_mounts > 0:
            self.logger.info(f"Successfully mounted all {total_mounts} required buckets")
            return True
        else:
            self.logger.error(f"Mounting failed: {success_count}/{total_mounts} buckets mounted successfully")
            return False
    
    def unmount_all(self):
        """Unmount all mounted paths."""
        for mount_path in self.mounted_paths:
            try:
                cmd = f"fusermount -u {mount_path}"
                subprocess.run(cmd, shell=True, capture_output=True)
                self.logger.info(f"Unmounted {mount_path}")
            except Exception as e:
                self.logger.error(f"Error unmounting {mount_path}: {e}")
        
        self.mounted_paths.clear()


class ImageDownloader(PipelineStep):
    """Downloads satellite imagery using Earth Engine."""
    
    def __init__(self, config):
        super().__init__(config, 'step_1_download')
        self._setup_paths()
    
    def _setup_paths(self):
        """Setup system paths for Earth Engine download."""
        sys.path.append(self.config['system_paths']['littoral_src'])
    
    def run(self, existing_images: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Download satellite imagery.
        
        Args:
            existing_images: List of already downloaded images (for update mode)
            
        Returns:
            Dictionary with download results
        """
        try:
            from littoral import ee_s2, littoral_sites
            
            self.logger.info(f"Starting image download for site: {self.config['site_name']}")
            
            # Connect to Earth Engine
            ee_s2.connect()
            
            # Load site parameters
            proj_params = littoral_sites.load_site_parameters_cg(
                self.config['site_name'], 
                self.config['save_path'], 
                self.config['site_table_path']
            )
            
            # Get filtered image collection
            se2_col = ee_s2.get_filtered_image_collection(proj_params)
            
            self.logger.info(f"Found {se2_col.size().getInfo()} images in collection")
            
            # Process images
            results = ee_s2.process_collection_images_tofiles(proj_params, se2_col)
            
            # Create processing dataframe
            df = pd.DataFrame(results)
            current_date = datetime.now().strftime("%Y%m%d")
            
            # Add pipeline tracking columns
            df['pipeline_run_date'] = current_date
            df['step_1_download'] = 'success'
            
            # Initialize all other steps as pending
            for i in range(2, 15):
                step_names = {
                    2: 'step_2_filter',
                    3: 'step_3_coregister', 
                    4: 'step_4_cloud_impute',
                    5: 'step_5_rgb_nir_creation',
                    6: 'step_6_upsample',
                    7: 'step_7_normalize',
                    8: 'step_8_segment',
                    9: 'step_9_boundary_extract',
                    10: 'step_10_boundary_refine',
                    11: 'step_11_geotransform',
                    12: 'step_12_filter_shorelines',
                    13: 'step_13_tide_model',
                    14: 'step_14_tide_correct'
                }
                if i in step_names:
                    df[step_names[i]] = 'pending'
            
            # Save processing file
            processing_filename = self.config.create_processing_filename()
            output_path = os.path.join(self.config.get_site_path(), processing_filename)
            df.to_csv(output_path, index=False)
            
            self.logger.info(f"Downloaded {len(df)} images successfully")
            self.logger.info(f"Processing tracking saved to: {processing_filename}")
            
            return {
                'downloaded_count': len(df),
                'processing_file': processing_filename,
                'results': results
            }
            
        except Exception as e:
            self.logger.error(f"Error in image download: {e}")
            raise


class Coregistration(PipelineStep):
    """Handles image coregistration."""
    
    def __init__(self, config):
        super().__init__(config, 'step_3_coregister')
        self._setup_paths()
    
    def _setup_paths(self):
        """Setup system paths for coregistration."""
        sys.path.append(self.config['system_paths']['coreg_src'])
        
        # Fix PROJ database context
        os.environ['PROJ_DATA'] = self.config['system_paths']['proj_data']
        from pyproj import datadir
        datadir.set_data_dir(self.config['system_paths']['proj_data'])
    
    def run(self) -> Dict[str, Any]:
        """Run coregistration process."""
        try:
            from littoral_coregistration import LittoralCoregistration
            
            self.logger.info("Starting coregistration")
            
            top_level = self.config.get_site_path()
            coreg = LittoralCoregistration(top_level)
            
            cleanup = self.config['processing']['coregistration']['cleanup']
            results = coreg.run(cleanup=cleanup)
            
            # Update status
            metrics = {'coregistered_files': len(results.get('coregistered_files', []))}
            self.update_status(metrics=metrics)
            
            self.logger.info(f"Coregistration complete. Files saved to: {coreg.coregistered_dir}")
            
            return {
                'coregistered_files': len(results.get('coregistered_files', [])),
                'output_dir': coreg.coregistered_dir,
                'results': results
            }
            
        except Exception as e:
            self.logger.error(f"Error in coregistration: {e}")
            raise


class CloudImputation(PipelineStep):
    """Handles cloud imputation."""
    
    def __init__(self, config):
        super().__init__(config, 'step_4_cloud_impute')
        self._setup_paths()
    
    def _setup_paths(self):
        """Setup system paths for cloud imputation."""
        sys.path.append(self.config['system_paths']['cloud_impute_src'])
    
    def run(self) -> Dict[str, Any]:
        """Run cloud imputation process."""
        try:
            import vpint_cloud_impute
            
            self.logger.info("Starting cloud imputation")
            
            folder_path = self.config.get_folder_path('tiff')
            cloud_results = vpint_cloud_impute.batch_remove_clouds_folder(folder_path)
            
            # Update status
            self.update_status()
            
            # Read report if available
            cloudless_folder = os.path.join(folder_path, "cloudless")
            report_info = {}
            
            if os.path.exists(cloudless_folder):
                csv_files = [f for f in os.listdir(cloudless_folder) if f.endswith('.csv')]
                if csv_files:
                    report_path = os.path.join(cloudless_folder, csv_files[0])
                    report_df = pd.read_csv(report_path)
                    report_info = {
                        'report_path': report_path,
                        'processed_files': len(report_df)
                    }
            
            self.logger.info("Cloud imputation complete")
            
            return {
                'cloud_results': cloud_results,
                'report_info': report_info
            }
            
        except Exception as e:
            self.logger.error(f"Error in cloud imputation: {e}")
            raise


class RGBNIRCreation(PipelineStep):
    """Creates RGB and NIR images from cloud-free data."""
    
    def __init__(self, config):
        super().__init__(config, 'step_5_rgb_nir_creation')
        self._setup_paths()
    
    def _setup_paths(self):
        """Setup system paths for RGB/NIR creation."""
        sys.path.append(self.config['system_paths']['littoral_src'])
    
    def run(self) -> Dict[str, Any]:
        """Run RGB/NIR creation process."""
        try:
            from littoral import ee_s2
            
            self.logger.info("Starting RGB/NIR creation")
            
            # Setup paths
            existing_nir_folder = self.config.get_folder_path('nir')
            existing_rgb_folder = self.config.get_folder_path('rgb')
            clear_output_folder = self.config.get_folder_path('clear_output')
            clear_tiff_folder = self.config.get_folder_path('clear_tiff')
            
            # Create output folder
            os.makedirs(clear_output_folder, exist_ok=True)
            
            # Process cloud imputed images
            processed_files = ee_s2.process_cloud_imputed_images(
                existing_nir_folder=existing_nir_folder,
                existing_rgb_folder=existing_rgb_folder,
                clear_tiff_folder=clear_tiff_folder,
                clear_output_folder=clear_output_folder
            )
            
            # Update status
            metrics = {'clear_images_created': len(processed_files)}
            self.update_status(metrics=metrics)
            
            self.logger.info(f"Created {len(processed_files)} clear images")
            
            return {
                'processed_files': processed_files,
                'output_folder': clear_output_folder
            }
            
        except Exception as e:
            self.logger.error(f"Error in RGB/NIR creation: {e}")
            raise


class Upsampling(PipelineStep):
    """Handles image upsampling using Real-ESRGAN."""
    
    def __init__(self, config):
        super().__init__(config, 'step_6_upsample')
        self._setup_paths()
    
    def _setup_paths(self):
        """Setup system paths for upsampling."""
        sys.path.append(self.config['system_paths']['real_esrgan_src'])
    
    def run(self) -> Dict[str, Any]:
        """Run upsampling process."""
        try:
            import RealESRGAN.model as re
            
            self.logger.info("Starting upsampling")
            
            up_input = self.config.get_folder_path('clear_output')
            upsampled_images = re.upsample_folder(up_input)
            
            # Update status
            metrics = {'upsampled_images': len(upsampled_images)}
            self.update_status(metrics=metrics)
            
            self.logger.info(f"Upsampled {len(upsampled_images)} images")
            
            return {
                'upsampled_images': upsampled_images,
                'input_folder': up_input
            }
            
        except Exception as e:
            self.logger.error(f"Error in upsampling: {e}")
            raise


class Normalization(PipelineStep):
    """Handles image normalization and defective image removal."""
    
    def __init__(self, config):
        super().__init__(config, 'step_7_normalize')
    
    def run(self) -> Dict[str, Any]:
        """Run normalization process."""
        try:
            self.logger.info("Starting normalization")
            
            up_folder = self.config.get_folder_path('upsampled')
            norm_out_folder = self.config.get_folder_path('normalized')
            
            os.makedirs(norm_out_folder, exist_ok=True)
            
            # Get processing parameters
            min_thresh = self.config['processing']['normalization']['min_threshold']
            max_thresh = self.config['processing']['normalization']['max_threshold']
            
            # Find NIR images
            nir_images = glob.glob(os.path.join(up_folder, '*_nir_up.png'))
            
            normalized_files = []
            skipped_files = []
            
            for img_path in nir_images:
                img = Image.open(img_path)
                arr = np.array(img)
                
                # Check if image has grey values between thresholds
                if np.any((arr > min_thresh) & (arr < max_thresh)):
                    # Normalize image
                    arr = ((arr - arr.min()) / (np.ptp(arr) + 1e-6) * 255).astype(np.uint8)
                    out_path = os.path.join(norm_out_folder, os.path.basename(img_path))
                    Image.fromarray(arr).save(out_path)
                    normalized_files.append(os.path.basename(img_path))
                else:
                    # Skip image with no grey values
                    self.logger.debug(f"Skipping normalization for {img_path} - no grey values")
                    skipped_files.append(os.path.basename(img_path))
            
            # Update status
            metrics = {
                'normalized_files': len(normalized_files),
                'skipped_normalization': len(skipped_files)
            }
            self.update_status(metrics=metrics)
            
            self.logger.info(f"Normalized: {len(normalized_files)} images, Skipped: {len(skipped_files)} images")
            
            return {
                'normalized_files': normalized_files,
                'skipped_files': skipped_files,
                'output_folder': norm_out_folder
            }
            
        except Exception as e:
            self.logger.error(f"Error in normalization: {e}")
            raise


class Segmentation(PipelineStep):
    """Handles image segmentation using YOLO."""
    
    def __init__(self, config):
        super().__init__(config, 'step_8_segment')
        self._setup_paths()
        self._yolo_model = None
    
    def _setup_paths(self):
        """Setup system paths for segmentation."""
        sys.path.append(self.config['system_paths']['segment_src'])
    
    def _load_model(self):
        """Load YOLO segmentation model."""
        if self._yolo_model is None:
            from seg_models.yolov8_seg import YOLOV8
            model_path = self.config['processing']['segmentation']['model_path']
            self._yolo_model = YOLOV8(folder=model_path)
        return self._yolo_model
    
    def run(self) -> Dict[str, Any]:
        """Run segmentation process."""
        try:
            self.logger.info("Starting segmentation")
            
            seg_input_folder = self.config.get_folder_path('normalized')
            yolo_model = self._load_model()
            
            mask_paths = yolo_model.mask_from_folder(seg_input_folder)
            
            # Update status
            metrics = {'segmentation_masks': len(mask_paths)}
            self.update_status(metrics=metrics)
            
            self.logger.info(f"Created {len(mask_paths)} segmentation masks")
            
            return {
                'mask_paths': mask_paths,
                'input_folder': seg_input_folder
            }
            
        except Exception as e:
            self.logger.error(f"Error in segmentation: {e}")
            raise


def get_processing_filename(site_path: str) -> Optional[str]:
    """Get the most recent processing CSV filename."""
    import glob
    pattern = os.path.join(site_path, "*_processing.csv")
    files = glob.glob(pattern)
    if files:
        return os.path.basename(max(files, key=os.path.getctime))
    return None


def update_processing_status(site_path: str, filename: str, step_name: str, 
                           file_statuses: Optional[Dict] = None, 
                           metrics: Optional[Dict] = None):
    """
    Update the processing CSV with step completion status.
    
    Args:
        site_path: Path to site directory
        filename: Processing CSV filename 
        step_name: Name of the pipeline step column
        file_statuses: Dict mapping image names to status ('success', 'failed', 'filtered')
        metrics: Dict of additional metrics to record
    """
    csv_path = os.path.join(site_path, filename)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        if file_statuses:
            # Update individual file statuses
            for img_name, status in file_statuses.items():
                mask = df['name'] == img_name
                df.loc[mask, step_name] = status
        else:
            # Update all files with same status
            df[step_name] = 'success'
            
        # Add metrics as new columns if provided
        if metrics:
            for metric_name, metric_value in metrics.items():
                df[metric_name] = metric_value
                
        df.to_csv(csv_path, index=False)
        logger.info(f"Updated {step_name} status in {filename}")
    else:
        logger.warning(f"Processing file not found: {csv_path}")