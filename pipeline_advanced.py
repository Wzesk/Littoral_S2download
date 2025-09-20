"""
Additional pipeline processing functions for boundary extraction, refinement,
geotransformation, filtering, and tidal correction.
"""

import os
import sys
import logging
import glob
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates

# Import optional dependencies conditionally
try:
    from sklearn.cluster import DBSCAN
except ImportError:
    DBSCAN = None

from pipeline_functions import PipelineStep

logger = logging.getLogger(__name__)


class BoundaryExtraction(PipelineStep):
    """Extracts boundaries from segmentation masks."""
    
    def __init__(self, config):
        super().__init__(config, 'step_9_boundary_extract')
        self._setup_paths()
    
    def _setup_paths(self):
        """Setup system paths for boundary extraction."""
        sys.path.append(self.config['system_paths']['refine_src'])
    
    def run(self) -> Dict[str, Any]:
        """Run boundary extraction process."""
        try:
            from littoral_refine import extract_boundary
            
            self.logger.info("Starting boundary extraction")
            
            mask_folder = self.config.get_folder_path('masked')
            shoreline_paths = extract_boundary.get_shorelines_from_folder(mask_folder)
            
            # Update status
            metrics = {'extracted_boundaries': len(shoreline_paths)}
            self.update_status(metrics=metrics)
            
            self.logger.info(f"Extracted {len(shoreline_paths)} boundaries")
            
            return {
                'shoreline_paths': shoreline_paths,
                'mask_folder': mask_folder
            }
            
        except Exception as e:
            self.logger.error(f"Error in boundary extraction: {e}")
            raise


class BoundaryRefinement(PipelineStep):
    """Refines extracted boundaries."""
    
    def __init__(self, config):
        super().__init__(config, 'step_10_boundary_refine')
        self._setup_paths()
    
    def _setup_paths(self):
        """Setup system paths for boundary refinement."""
        sys.path.append(self.config['system_paths']['refine_src'])
    
    def run(self) -> Dict[str, Any]:
        """Run boundary refinement process."""
        try:
            from littoral_refine import refine_boundary
            
            self.logger.info("Starting boundary refinement")
            
            site_path = self.config.get_site_path()
            
            # Always use the original refinement function like the notebook
            self.logger.info("Running boundary refinement using refine_boundary.refine_shorelines()")
            refine_boundary.refine_shorelines(site_path)
            
            # Update status
            self.update_status()
            
            self.logger.info("Boundary refinement complete")
            
            return {
                'site_path': site_path
            }
            
        except Exception as e:
            self.logger.error(f"Error in boundary refinement: {e}")
            raise
    

class Geotransformation(PipelineStep):
    """Handles geotransformation of shorelines."""
    
    def __init__(self, config):
        super().__init__(config, 'step_11_geotransform')
        self._setup_paths()
    
    def _setup_paths(self):
        """Setup system paths for geotransformation."""
        sys.path.append(self.config['system_paths']['littoral_src'])
        
        # Fix PROJ database context
        os.environ['PROJ_DATA'] = self.config['system_paths']['proj_data']
        from pyproj import datadir
        datadir.set_data_dir(self.config['system_paths']['proj_data'])
    
    def run(self) -> Dict[str, Any]:
        """Run geotransformation process."""
        try:
            from littoral import geo_transform
            
            self.logger.info("Starting geotransformation")
            
            # Setup paths
            cloudless_report_path = os.path.join(
                self.config.get_folder_path('tiff'), 
                "cloudless", 
                "cloudless_report.csv"
            )
            coreg_path = os.path.join(
                self.config.get_site_path(), 
                "coregistered", 
                "filtered_files.csv"
            )
            shoreline_path = self.config.get_folder_path('shoreline')
            
            # Run geotransformation
            geo_transform_results = geo_transform.batch_geotransform(
                shoreline_path, 
                cloudless_report_path, 
                coreg_path
            )
            
            # Update status
            self.update_status()
            
            self.logger.info("Geotransformation complete")
            
            return {
                'geo_transform_results': geo_transform_results,
                'shoreline_path': shoreline_path
            }
            
        except Exception as e:
            self.logger.error(f"Error in geotransformation: {e}")
            raise


class ShorelineFiltering(PipelineStep):
    """Filters and validates geotransformed shorelines."""
    
    def __init__(self, config):
        super().__init__(config, 'step_12_filter_shorelines')
    
    def _filter_main_cluster(self, x: np.ndarray, y: np.ndarray, lengths_norm: np.ndarray, 
                           eps: float = 0.05, min_samples: int = 10) -> np.ndarray:
        """
        Identifies the main cluster in 3D space and removes outliers.
        
        Args:
            x: X coordinates (normalized or raw)
            y: Y coordinates (normalized or raw) 
            lengths_norm: Normalized lengths
            eps: DBSCAN epsilon (distance threshold)
            min_samples: Minimum samples for a cluster
            
        Returns:
            Boolean mask for points in the main cluster
        """
        points = np.stack([x, y, lengths_norm], axis=1)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_
        
        # Find the largest cluster (excluding noise label -1)
        unique, counts = np.unique(labels[labels != -1], return_counts=True)
        if len(counts) == 0:
            # No clusters found, return all as outliers
            return np.zeros_like(labels, dtype=bool)
        
        main_cluster = unique[np.argmax(counts)]
        mask = labels == main_cluster
        return mask
    
    def _get_datetime_from_filename(self, filename: str) -> pd.Timestamp:
        """Extract datetime from filename."""
        return pd.to_datetime(filename[:15], format='%Y%m%dT%H%M%S')
    
    def _get_spline_and_normals(self, x: np.ndarray, y: np.ndarray, num_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Generate spline and normal vectors from coordinates."""
        try:
            from geomdl import BSpline, utilities
            
            points = list(zip(x, y))
            if points[0] != points[-1]:
                points.append(points[0])
            
            curve = BSpline.Curve()
            curve.degree = min(3, len(points) - 1)
            curve.ctrlpts = points
            curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
            curve.delta = 1.0 / (num_points - 1)
            spline_points = np.array(curve.evalpts)
            
            normals = []
            for j in range(len(spline_points)):
                if j == 0:
                    tangent = spline_points[j+1] - spline_points[j]
                elif j == len(spline_points) - 1:
                    tangent = spline_points[j] - spline_points[j-1]
                else:
                    tangent = spline_points[j+1] - spline_points[j-1]
                
                normal = np.array([-tangent[1], tangent[0]])
                norm_len = np.linalg.norm(normal)
                normals.append(normal / norm_len if norm_len > 0 else np.zeros_like(normal))
            
            return spline_points, np.array(normals)
            
        except ImportError:
            self.logger.warning("geomdl not available, using simple interpolation")
            # Fallback to simple linear interpolation
            return np.column_stack([x, y]), np.zeros((len(x), 2))
    
    def _find_closest_point_and_distance(self, pt: np.ndarray, other_spline: np.ndarray) -> Tuple[float, np.ndarray]:
        """Find closest point on other spline and return distance and direction."""
        distances = np.linalg.norm(other_spline - pt, axis=1)
        closest_idx = np.argmin(distances)
        closest_point = other_spline[closest_idx]
        distance = distances[closest_idx]
        direction = closest_point - pt
        
        # Normalize direction vector
        if distance > 0:
            direction = direction / distance
        else:
            direction = np.array([0, 0])
            
        return distance, direction
    
    def run(self) -> Dict[str, Any]:
        """Run shoreline filtering process."""
        try:
            self.logger.info("Starting shoreline filtering")
            
            shoreline_path = self.config.get_folder_path('shoreline')
            
            # Get geotransformed files
            geo_files = [f for f in os.listdir(shoreline_path) if f.endswith('o.csv')]
            
            if not geo_files:
                self.logger.warning("No geotransformed files found")
                return {'filtered_files': [], 'defective_files': []}
            
            self.logger.info(f"Found {len(geo_files)} geotransformed shorelines")
            
            # Check if shoreline filtering is enabled
            if not self.config['processing']['filtering']['enable_shoreline_filtering']:
                self.logger.info("Shoreline filtering is disabled - returning all files")
                return {
                    'filtered_files': geo_files,
                    'defective_files': [],
                    'total_files': len(geo_files),
                    'shoreline_path': shoreline_path
                }
            
            # Calculate centroids and lengths for clustering
            centroids = []
            lengths = []
            
            for csv_file in geo_files:
                csv_path = os.path.join(shoreline_path, csv_file)
                df = pd.read_csv(csv_path)
                x = df.iloc[:, 0].values
                y = df.iloc[:, 1].values
                
                # Calculate centroid
                centroid_x = np.mean(x)
                centroid_y = np.mean(y)
                centroids.append((centroid_x, centroid_y))
                
                # Calculate polyline length
                dx = np.diff(x)
                dy = np.diff(y)
                segment_lengths = np.sqrt(dx**2 + dy**2)
                total_length = np.sum(segment_lengths)
                lengths.append(total_length)
            
            centroids = np.array(centroids)
            lengths = np.array(lengths)
            
            # Normalize coordinates and lengths to [0, 1]
            centroid_x_norm = (centroids[:, 0] - centroids[:, 0].min()) / (np.ptp(centroids[:, 0]) + 1e-6)
            centroid_y_norm = (centroids[:, 1] - centroids[:, 1].min()) / (np.ptp(centroids[:, 1]) + 1e-6)
            lengths_norm = (lengths - lengths.min()) / (np.ptp(lengths) + 1e-6)
            
            # Apply initial clustering filter
            if len(geo_files) < 50:
                filtered_geo_files = geo_files
                self.logger.info("Skipping clustering filter (< 50 shorelines)")
            else:
                eps = self.config['processing']['filtering']['cluster_eps']
                min_samples = self.config['processing']['filtering']['cluster_min_samples']
                mask = self._filter_main_cluster(centroid_x_norm, centroid_y_norm, lengths_norm, eps, min_samples)
                filtered_geo_files = [geo_files[i] for i in range(len(geo_files)) if mask[i]]
                self.logger.info(f"Clustering filter applied: {len(filtered_geo_files)}/{len(geo_files)} retained")
            
            # Apply defective shoreline filtering if we have enough shorelines
            defective_threshold = self.config['processing']['filtering']['defective_threshold']
            defective_files = []
            
            if len(filtered_geo_files) > defective_threshold:
                self.logger.info("Applying defective shoreline filtering")
                
                # Sort shorelines by datetime
                shoreline_info = []
                for fname in filtered_geo_files:
                    try:
                        dt = self._get_datetime_from_filename(fname)
                        shoreline_info.append((dt, fname))
                    except:
                        self.logger.warning(f"Could not parse datetime from {fname}")
                        continue
                
                shoreline_info.sort()
                sorted_files = [f for _, f in shoreline_info]
                
                # Precompute splines for all shorelines
                shoreline_splines = []
                for fname in sorted_files:
                    df = pd.read_csv(os.path.join(shoreline_path, fname))
                    x = df['xm'].values if 'xm' in df.columns else df.iloc[:, 0].values
                    y = df['ym'].values if 'ym' in df.columns else df.iloc[:, 1].values
                    spline, _ = self._get_spline_and_normals(x, y)
                    shoreline_splines.append(spline)
                
                # Calculate global mean distance
                all_distances = []
                for i in range(1, len(shoreline_splines) - 1):
                    curr_spline = shoreline_splines[i]
                    prev_spline = shoreline_splines[i-1]
                    next_spline = shoreline_splines[i+1]
                    
                    for pt in curr_spline[::10]:  # Sample every 10th point for efficiency
                        dist_prev, _ = self._find_closest_point_and_distance(pt, prev_spline)
                        dist_next, _ = self._find_closest_point_and_distance(pt, next_spline)
                        all_distances.extend([dist_prev, dist_next])
                
                global_mean_dist = np.mean(all_distances)
                self.logger.info(f"Global mean distance: {global_mean_dist:.3f}")
                
                # Filter defective shorelines
                defective_indices = set()
                distance_multiplier = self.config['processing']['filtering']['distance_multiplier']
                defective_point_ratio = self.config['processing']['filtering']['defective_point_ratio']
                
                for i in range(1, len(shoreline_splines) - 1):
                    curr_spline = shoreline_splines[i]
                    prev_spline = shoreline_splines[i-1]
                    next_spline = shoreline_splines[i+1]
                    
                    defective_points = 0
                    total_points = len(curr_spline[::10])  # Sample points
                    
                    for pt in curr_spline[::10]:
                        dist_prev, dir_prev = self._find_closest_point_and_distance(pt, prev_spline)
                        dist_next, dir_next = self._find_closest_point_and_distance(pt, next_spline)
                        
                        # Check if distances are above threshold
                        if dist_prev > distance_multiplier * global_mean_dist or dist_next > distance_multiplier * global_mean_dist:
                            # Check if direction vectors are within 90 degrees
                            dot_product = np.dot(dir_prev, dir_next)
                            if dot_product > 0:  # Directions are closer than 90 degrees
                                defective_points += 1
                    
                    # If more than threshold of points are defective, mark shoreline as defective
                    if defective_points > defective_point_ratio * total_points:
                        defective_indices.add(i)
                        self.logger.debug(f"Shoreline {i} marked as defective: {defective_points}/{total_points} defective points")
                
                # Filter out defective shorelines
                filtered2_sorted_files = [sorted_files[i] for i in range(len(sorted_files)) if i not in defective_indices]
                defective_files = [sorted_files[i] for i in defective_indices]
                
                self.logger.info(f"Filtered out {len(defective_indices)} defective shorelines")
                filtered_geo_files = filtered2_sorted_files
            else:
                self.logger.info(f"Skipping defective shoreline filtering (≤{defective_threshold} shorelines)")
            
            # Update status
            metrics = {
                'total_shorelines': len(geo_files),
                'filtered_shorelines': len(filtered_geo_files),
                'filtering_applied': len(geo_files) >= 50
            }
            self.update_status(metrics=metrics)
            
            self.logger.info(f"Shoreline filtering complete: {len(filtered_geo_files)} shorelines retained")
            
            return {
                'filtered_files': filtered_geo_files,
                'defective_files': defective_files,
                'total_files': len(geo_files),
                'shoreline_path': shoreline_path
            }
            
        except Exception as e:
            self.logger.error(f"Error in shoreline filtering: {e}")
            raise


class TidalModeling(PipelineStep):
    """Handles tidal modeling and prediction."""
    
    def __init__(self, config):
        super().__init__(config, 'step_13_tide_model')
        self._setup_paths()
    
    def _setup_paths(self):
        """Setup system paths for tidal modeling."""
        sys.path.append(self.config['system_paths']['littoral_src'])
        
        # Fix PROJ database context
        os.environ['PROJ_DATA'] = self.config['system_paths']['proj_data']
        from pyproj import datadir
        datadir.set_data_dir(self.config['system_paths']['proj_data'])
    
    def run(self) -> Dict[str, Any]:
        """Run tidal modeling process."""
        try:
            from littoral import littoral_tide_correction
            
            self.logger.info("Starting tidal modeling")
            
            # Get image datetimes from processing file
            processing_filename = self.config.get_processing_filename()
            if not processing_filename:
                raise ValueError("No processing file found for tidal modeling")
            
            df_proc = pd.read_csv(os.path.join(self.config.get_site_path(), processing_filename))
            image_names = df_proc['name'] if 'name' in df_proc.columns else df_proc.iloc[:, 0]
            
            # Extract datetime strings and convert to timestamps
            sample_date_strs = image_names.str.split('_').str[0]
            times = pd.to_datetime(sample_date_strs, format='%Y%m%dT%H%M%S', utc=True)
            
            # Get center location from site configuration
            site_json_path = os.path.join(self.config.get_site_path(), f"{self.config['site_name']}.json")
            if not os.path.exists(site_json_path):
                raise FileNotFoundError(f"Site configuration file not found: {site_json_path}")
            
            with open(site_json_path) as f:
                settings = json.load(f)
                aoi = settings['aoi']
                center_location = [sum(x) / len(x) for x in zip(*aoi)]
            
            # Convert times to numpy array for pyTMD compatibility
            times_array = times.dt.tz_localize(None).to_numpy()
            
            # Model tides
            tide_config = self.config['processing']['tide_correction']
            tide_df = littoral_tide_correction.model_tides(
                center_location[1],  # latitude
                center_location[0],  # longitude
                times_array,
                model=tide_config['model'],
                directory=tide_config['tide_model_dir'],
                epsg=tide_config['epsg'],
                method=tide_config['method'],
                extrapolate=tide_config['extrapolate'],
                cutoff=tide_config['cutoff']
            )
            
            # Calculate tide corrections
            tide_df = littoral_tide_correction.calculate_tide_corrections(
                tide_df,
                reference_elevation=tide_config['reference_elevation'],
                beach_slope=tide_config['beach_slope']
            )
            
            # Save tide corrections
            csv_path = os.path.join(self.config.get_site_path(), "tide_corrections.csv")
            tide_df.to_csv(csv_path, index=False)
            
            # Update status
            avg_correction = 0.0
            if 'horizontal_correction_m' in tide_df.columns:
                avg_correction = float(tide_df['horizontal_correction_m'].mean())
            elif 'tide' in tide_df.columns:
                avg_correction = float(tide_df['tide'].mean())
            
            metrics = {
                'tidal_predictions': len(tide_df),
                'tide_range_m': float(tide_df['tide'].max() - tide_df['tide'].min()) if 'tide' in tide_df.columns else 0.0,
                'avg_correction_m': avg_correction
            }
            self.update_status(metrics=metrics)
            
            self.logger.info(f"Tidal modeling complete. Results saved to: {csv_path}")
            
            return {
                'tide_df': tide_df,
                'csv_path': csv_path,
                'center_location': center_location
            }
            
        except Exception as e:
            self.logger.error(f"Error in tidal modeling: {e}")
            raise


class TidalCorrection(PipelineStep):
    """Applies tidal corrections to shorelines."""
    
    def __init__(self, config):
        super().__init__(config, 'step_14_tide_correct')
        self._setup_paths()
    
    def _setup_paths(self):
        """Setup system paths for tidal correction."""
        sys.path.append(self.config['system_paths']['littoral_src'])
    
    def run(self, filtered_shoreline_files: List[str]) -> Dict[str, Any]:
        """
        Run tidal correction process.
        
        Args:
            filtered_shoreline_files: List of filtered shoreline files to correct
            
        Returns:
            Dictionary with correction results
        """
        try:
            from littoral import littoral_tide_correction
            
            self.logger.info("Starting tidal correction")
            
            # Check for tidal corrections file
            tidal_corrections_csv = os.path.join(self.config.get_site_path(), "tide_corrections.csv")
            if not os.path.exists(tidal_corrections_csv):
                raise FileNotFoundError(f"Tidal corrections file not found: {tidal_corrections_csv}")
            
            shoreline_path = self.config.get_folder_path('shoreline')
            
            # Apply tidal corrections
            corrected_shoreline_paths = littoral_tide_correction.apply_tidal_corrections_to_shorelines(
                filtered_shoreline_files=filtered_shoreline_files,
                shoreline_path=shoreline_path,
                tidal_corrections_path=tidal_corrections_csv,
                output_folder="TIDAL_CORRECTED"
            )
            
            # Update status - FINAL STEP
            metrics = {
                'corrected_shorelines': len(corrected_shoreline_paths),
                'pipeline_complete': True
            }
            self.update_status(metrics=metrics)
            
            self.logger.info(f"Tidal correction complete. {len(corrected_shoreline_paths)} files corrected")
            
            return {
                'corrected_shoreline_paths': corrected_shoreline_paths,
                'tidal_corrections_csv': tidal_corrections_csv
            }
            
        except Exception as e:
            self.logger.error(f"Error in tidal correction: {e}")
            raise