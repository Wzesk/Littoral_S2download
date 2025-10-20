"""
Update mode logic and pipeline orchestrator for the Littoral Processing Pipeline.

This module handles the main pipeline execution logic, including update mode
functionality that only processes new images.
"""

import os
import sys
import logging
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

from .pipeline_config import PipelineConfig
from .pipeline_functions import (
    ImageDownloader, Coregistration, CloudImputation, 
    RGBNIRCreation, Upsampling, Normalization, Segmentation
)
from .pipeline_advanced import (
    BoundaryExtraction, BoundaryRefinement, Geotransformation,
    ShorelineFiltering, TidalModeling, TidalCorrection, GeoJSONConversion
)
from .mount_verification import verify_required_mounts, verify_tide_mount


class UpdateModeManager:
    """Manages update mode functionality for the pipeline."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.UpdateModeManager")
    
    def get_existing_images(self) -> List[str]:
        """
        Get list of images that have already been processed.
        
        Returns:
            List of image names that have been processed
        """
        try:
            processing_file = self.config.get_processing_filename()
            if not processing_file:
                self.logger.info("No existing processing file found - running full pipeline")
                return []
            
            csv_path = os.path.join(self.config.get_site_path(), processing_file)
            if not os.path.exists(csv_path):
                self.logger.info("Processing file not found - running full pipeline")
                return []
            
            df = pd.read_csv(csv_path)
            if 'name' in df.columns:
                existing_images = df['name'].tolist()
                self.logger.info(f"Found {len(existing_images)} existing processed images")
                return existing_images
            else:
                self.logger.warning("Processing file missing 'name' column")
                return []
                
        except Exception as e:
            self.logger.error(f"Error reading existing images: {e}")
            return []
    
    def filter_new_images(self, all_images: List[str], existing_images: List[str]) -> List[str]:
        """
        Filter out images that have already been processed.
        
        Args:
            all_images: List of all available images
            existing_images: List of already processed images
            
        Returns:
            List of new images to process
        """
        new_images = [img for img in all_images if img not in existing_images]
        self.logger.info(f"Found {len(new_images)} new images to process")
        return new_images
    
    def merge_processing_results(self, old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge old and new processing results.
        
        Args:
            old_df: Existing processing dataframe
            new_df: New processing results dataframe
            
        Returns:
            Merged dataframe
        """
        try:
            # Ensure both dataframes have the same columns
            all_columns = set(old_df.columns) | set(new_df.columns)
            
            for col in all_columns:
                if col not in old_df.columns:
                    old_df[col] = 'pending'
                if col not in new_df.columns:
                    new_df[col] = 'pending'
            
            # Reorder columns to match
            column_order = sorted(all_columns)
            old_df = old_df[column_order]
            new_df = new_df[column_order]
            
            # Concatenate dataframes
            merged_df = pd.concat([old_df, new_df], ignore_index=True)
            
            self.logger.info(f"Merged processing results: {len(old_df)} existing + {len(new_df)} new = {len(merged_df)} total")
            
            return merged_df
            
        except Exception as e:
            self.logger.error(f"Error merging processing results: {e}")
            return new_df  # Fallback to new results only
    
    def backup_existing_results(self, backup_suffix: str = None) -> str:
        """
        Create backup of existing processing results.
        
        Args:
            backup_suffix: Optional suffix for backup file
            
        Returns:
            Path to backup file
        """
        try:
            processing_file = self.config.get_processing_filename()
            if not processing_file:
                return ""
            
            if backup_suffix is None:
                backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            original_path = os.path.join(self.config.get_site_path(), processing_file)
            backup_filename = f"backup_{backup_suffix}_{processing_file}"
            backup_path = os.path.join(self.config.get_site_path(), backup_filename)
            
            shutil.copy2(original_path, backup_path)
            self.logger.info(f"Created backup: {backup_filename}")
            
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
            return ""


class PipelineOrchestrator:
    """Main pipeline orchestrator that manages all processing steps."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.PipelineOrchestrator")
        self.update_manager = UpdateModeManager(config)
        
        # Initialize all pipeline steps
        self.steps = {
            'download': ImageDownloader(config),
            'coregister': Coregistration(config),
            'cloud_impute': CloudImputation(config),
            'rgb_nir_creation': RGBNIRCreation(config),
            'upsample': Upsampling(config),
            'normalize': Normalization(config),
            'segment': Segmentation(config),
            'boundary_extract': BoundaryExtraction(config),
            'boundary_refine': BoundaryRefinement(config),
            'geotransform': Geotransformation(config),
            'filter_shorelines': ShorelineFiltering(config),
            'tide_model': TidalModeling(config),
            'tide_correct': TidalCorrection(config),
            'geojson_convert': GeoJSONConversion(config)
        }
    
    def setup_logging(self, log_level: str = "INFO"):
        """Setup logging configuration."""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # Ensure the site directory exists before creating log file
        site_path = self.config.get_site_path()
        os.makedirs(site_path, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    os.path.join(site_path, "pipeline.log"),
                    mode='a'
                )
            ]
        )
    
    def validate_prerequisites(self) -> bool:
        """Validate that all prerequisites are met before running pipeline."""
        self.logger.info("Validating pipeline prerequisites")
        
        # Validate configuration
        config_errors = self.config.validate_config()
        if config_errors:
            for error in config_errors:
                self.logger.error(f"Configuration error: {error}")
            return False
        
        # Check if site directory exists
        site_path = self.config.get_site_path()
        if not os.path.exists(site_path):
            self.logger.info(f"Creating site directory: {site_path}")
            os.makedirs(site_path, exist_ok=True)
        
        # Verify that required cloud storage buckets are mounted
        self.logger.info("Checking cloud storage mount status...")
        if not verify_required_mounts(self.logger):
            self.logger.error("CRITICAL ERROR: Required cloud storage buckets are not mounted")
            self.logger.error("PIPELINE CANNOT PROCEED - STOPPING EXECUTION")
            return False
        
        self.logger.info("✅ All required cloud storage buckets are mounted")
        self.logger.info("Prerequisites validation complete")
        return True
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete pipeline from start to finish.
        
        Returns:
            Dictionary with pipeline results
        """
        self.logger.info("=" * 60)
        self.logger.info(f"STARTING FULL PIPELINE - {self.config['site_name']}")
        self.logger.info("=" * 60)
        
        results = {}
        
        try:
            # Step 1: Download imagery
            if 'download' not in self.config['pipeline']['skip_steps']:
                self.logger.info("Step 1: Downloading imagery")
                results['download'] = self.steps['download'].run()
            
            # Step 2: Coregistration
            if 'coregister' not in self.config['pipeline']['skip_steps']:
                self.logger.info("Step 2: Coregistration")
                results['coregister'] = self.steps['coregister'].run()
            
            # Step 3: Cloud imputation
            if 'cloud_impute' not in self.config['pipeline']['skip_steps']:
                self.logger.info("Step 3: Cloud imputation")
                results['cloud_impute'] = self.steps['cloud_impute'].run()
            
            # Step 4: RGB/NIR creation
            if 'rgb_nir_creation' not in self.config['pipeline']['skip_steps']:
                self.logger.info("Step 4: RGB/NIR creation")
                results['rgb_nir_creation'] = self.steps['rgb_nir_creation'].run()
            
            # Step 5: Upsampling
            if 'upsample' not in self.config['pipeline']['skip_steps']:
                self.logger.info("Step 5: Upsampling")
                results['upsample'] = self.steps['upsample'].run()
            
            # Step 6: Normalization
            if 'normalize' not in self.config['pipeline']['skip_steps']:
                self.logger.info("Step 6: Normalization")
                results['normalize'] = self.steps['normalize'].run()
            
            # Step 7: Segmentation
            if 'segment' not in self.config['pipeline']['skip_steps']:
                self.logger.info("Step 7: Segmentation")
                results['segment'] = self.steps['segment'].run()
            
            # Step 8: Boundary extraction
            if 'boundary_extract' not in self.config['pipeline']['skip_steps']:
                self.logger.info("Step 8: Boundary extraction")
                results['boundary_extract'] = self.steps['boundary_extract'].run()
            
            # Step 9: Boundary refinement
            if 'boundary_refine' not in self.config['pipeline']['skip_steps']:
                self.logger.info("Step 9: Boundary refinement")
                results['boundary_refine'] = self.steps['boundary_refine'].run()
            
            # Step 10: Geotransformation
            if 'geotransform' not in self.config['pipeline']['skip_steps']:
                self.logger.info("Step 10: Geotransformation")
                results['geotransform'] = self.steps['geotransform'].run()
            
            # Step 11: Shoreline filtering
            if 'filter_shorelines' not in self.config['pipeline']['skip_steps']:
                self.logger.info("Step 11: Shoreline filtering")
                results['filter_shorelines'] = self.steps['filter_shorelines'].run()
            
            # Step 12: Tidal modeling (requires tide bucket mount)
            if 'tide_model' not in self.config['pipeline']['skip_steps']:
                self.logger.info("Step 12: Tidal modeling")
                if not verify_tide_mount(self.logger):
                    self.logger.error("Failed to access tide bucket")
                    raise RuntimeError("Cannot proceed with tidal modeling")
                
                results['tide_model'] = self.steps['tide_model'].run()
            
            # Step 13: Tidal correction
            if 'tide_correct' not in self.config['pipeline']['skip_steps']:
                self.logger.info("Step 13: Tidal correction")
                filtered_files = results.get('filter_shorelines', {}).get('filtered_files', [])
                results['tide_correct'] = self.steps['tide_correct'].run(filtered_files)
            
            # Step 14: GeoJSON conversion and metadata upload
            if 'geojson_convert' not in self.config['pipeline']['skip_steps']:
                self.logger.info("Step 14: GeoJSON conversion and metadata upload")
                corrected_files = results.get('tide_correct', {}).get('corrected_shoreline_paths', [])
                results['geojson_convert'] = self.steps['geojson_convert'].run(corrected_files)
            
            self.logger.info("=" * 60)
            self.logger.info("FULL PIPELINE COMPLETE")
            self.logger.info("=" * 60)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def run_update_pipeline(self) -> Dict[str, Any]:
        """
        Run pipeline in update mode (only process new images).
        
        Returns:
            Dictionary with pipeline results
        """
        self.logger.info("=" * 60)
        self.logger.info(f"STARTING UPDATE PIPELINE - {self.config['site_name']}")
        self.logger.info("=" * 60)
        
        try:
            # Get existing processed images
            existing_images = self.update_manager.get_existing_images()
            
            if not existing_images:
                self.logger.info("No existing images found - running full pipeline")
                return self.run_full_pipeline()
            
            # Create backup of existing results
            backup_path = self.update_manager.backup_existing_results()
            
            # For update mode, we primarily need to:
            # 1. Download any new images
            # 2. Run the full pipeline on new images only
            # 3. Merge results with existing data
            
            # Download and check for new images
            self.logger.info("Checking for new images")
            download_results = self.steps['download'].run(existing_images)
            
            # If no new images, skip processing
            if download_results['downloaded_count'] == 0:
                self.logger.info("No new images found - update complete")
                return {'status': 'no_updates', 'existing_images': len(existing_images)}
            
            # Run full pipeline on new images
            # Note: The individual steps should handle filtering to only process new images
            self.logger.info(f"Processing {download_results['downloaded_count']} new images")
            
            # Continue with rest of pipeline (steps will process only new data)
            results = {'download': download_results}
            
            # Run remaining steps
            pipeline_steps = [
                'coregister', 'cloud_impute', 'rgb_nir_creation', 'upsample',
                'normalize', 'segment', 'boundary_extract', 'boundary_refine',
                'geotransform', 'filter_shorelines', 'tide_model', 'tide_correct'
            ]
            
            for step_name in pipeline_steps:
                if step_name not in self.config['pipeline']['skip_steps']:
                    self.logger.info(f"Step: {step_name}")
                    
                    if step_name == 'tide_model':
                        # Verify tide bucket for tidal modeling
                        if not verify_tide_mount(self.logger):
                            self.logger.error("Failed to access tide bucket")
                            raise RuntimeError("Cannot proceed with tidal modeling")
                    
                    if step_name == 'tide_correct':
                        # Pass filtered shorelines from filtering step
                        filtered_files = results.get('filter_shorelines', {}).get('filtered_files', [])
                        results[step_name] = self.steps[step_name].run(filtered_files)
                    else:
                        results[step_name] = self.steps[step_name].run()
            
            self.logger.info("=" * 60)
            self.logger.info("UPDATE PIPELINE COMPLETE")
            self.logger.info("=" * 60)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Update pipeline failed: {e}")
            raise
    
    def run(self) -> Dict[str, Any]:
        """
        Run the pipeline based on configured mode.
        
        Returns:
            Dictionary with pipeline results
        """
        # Setup logging
        self.setup_logging(self.config['pipeline']['log_level'])
        
        # Validate prerequisites
        if not self.validate_prerequisites():
            raise RuntimeError("Prerequisites validation failed")
        
        # Run pipeline based on mode
        run_mode = self.config['pipeline']['run_mode']
        
        if run_mode == 'update':
            return self.run_update_pipeline()
        else:
            return self.run_full_pipeline()
    
    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a summary report of pipeline execution.
        
        Args:
            results: Pipeline execution results
            
        Returns:
            Summary report as string
        """
        try:
            processing_filename = self.config.get_processing_filename()
            if not processing_filename:
                return "No processing file available for summary"
            
            final_df = pd.read_csv(os.path.join(self.config.get_site_path(), processing_filename))
            
            report_lines = []
            report_lines.append("=" * 60)
            report_lines.append(f"LITTORAL PIPELINE SUMMARY - {self.config['site_name']}")
            report_lines.append(f"Run Date: {final_df['pipeline_run_date'].iloc[0]}")
            report_lines.append(f"Run Mode: {self.config['pipeline']['run_mode']}")
            report_lines.append("=" * 60)
            
            # Count step completions
            step_columns = [col for col in final_df.columns if col.startswith('step_')]
            
            step_names = {
                'step_1_download': '1. Download Imagery',
                'step_2_filter': '2. Initial Filter',
                'step_3_coregister': '3. Coregister',
                'step_4_cloud_impute': '4. Cloud Imputation',
                'step_5_rgb_nir_creation': '5. RGB/NIR Creation',
                'step_6_upsample': '6. Upsampling',
                'step_7_normalize': '7. Normalization',
                'step_8_segment': '8. Segmentation',
                'step_9_boundary_extract': '9. Boundary Extraction',
                'step_10_boundary_refine': '10. Boundary Refinement',
                'step_11_geotransform': '11. Geotransformation',
                'step_12_filter_shorelines': '12. Shoreline Filtering',
                'step_13_tide_model': '13. Tidal Modeling',
                'step_14_tide_correct': '14. Tidal Correction'
            }
            
            for step_col in step_columns:
                if step_col in final_df.columns:
                    success_count = (final_df[step_col] == 'success').sum()
                    total_count = len(final_df)
                    name = step_names.get(step_col, step_col)
                    report_lines.append(f"{name}: {success_count}/{total_count}")
            
            # Display key metrics
            report_lines.append("\n" + "=" * 60)
            report_lines.append("KEY METRICS:")
            report_lines.append("=" * 60)
            
            metrics_to_show = [
                'coregistered_files', 'clear_images_created', 'upsampled_images',
                'normalized_files', 'skipped_normalization', 'segmentation_masks',
                'extracted_boundaries', 'total_shorelines', 'filtered_shorelines',
                'corrected_shorelines', 'tidal_predictions', 'tide_range_m',
                'avg_correction_m'
            ]
            
            for metric in metrics_to_show:
                if metric in final_df.columns:
                    value = final_df[metric].iloc[0] if not pd.isna(final_df[metric].iloc[0]) else 'N/A'
                    report_lines.append(f"{metric.replace('_', ' ').title()}: {value}")
            
            report_lines.append("\n" + "=" * 60)
            if 'pipeline_complete' in final_df.columns and final_df['pipeline_complete'].iloc[0]:
                report_lines.append("✅ PIPELINE STATUS: COMPLETE")
            else:
                report_lines.append("⚠️  PIPELINE STATUS: IN PROGRESS")
            report_lines.append("=" * 60)
            
            report_lines.append(f"\nDetailed results saved to: {processing_filename}")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {e}")
            return f"Error generating summary: {e}"