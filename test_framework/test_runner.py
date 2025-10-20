#!/usr/bin/env python3
"""
Pipeline Test Framework

This framework provides comprehensive testing capabilities for the Littoral Processing Pipeline,
including performance benchmarking, quality assessment, and regression testing.

Usage:
    python test_runner.py --config test_config.yaml
    python test_runner.py --list-configs
    python test_runner.py --config quick_smoke_test.yaml --output-dir custom_results/
"""

import os
import sys
import argparse
import logging
import yaml
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import time
import traceback
import subprocess
import psutil

# Add parent directory to path for pipeline imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import PipelineConfig, PipelineOrchestrator
from pipeline.pipeline_functions import PipelineStep
import pipeline.mount_verification as mount_verification


class TestMetrics:
    """Container for test metrics and statistics."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.success = False
        self.error_message = None
        self.memory_usage = {}
        self.cpu_usage = {}
        self.custom_metrics = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration.total_seconds() if self.duration else None,
            'success': self.success,
            'error_message': self.error_message,
            'memory_usage_mb': self.memory_usage,
            'cpu_usage_percent': self.cpu_usage,
            'custom_metrics': self.custom_metrics
        }


class PipelineStepTester:
    """Base class for testing individual pipeline steps."""
    
    def __init__(self, step_name: str, config: Dict[str, Any]):
        self.step_name = step_name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{step_name}")
        
    def setup(self, site_name: str, site_path: str) -> bool:
        """Setup test environment for this step."""
        return True
        
    def run_test(self, site_name: str, site_path: str) -> TestMetrics:
        """Run the test for this step and collect metrics."""
        metrics = TestMetrics()
        metrics.start_time = datetime.now()
        
        try:
            # Monitor system resources
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run the actual test
            success, custom_metrics = self._execute_step(site_name, site_path)
            
            # Collect final metrics
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            metrics.memory_usage = {
                'initial_mb': initial_memory,
                'final_mb': final_memory,
                'peak_mb': max(initial_memory, final_memory)
            }
            
            metrics.success = success
            metrics.custom_metrics = custom_metrics
            
        except Exception as e:
            self.logger.error(f"Error in {self.step_name}: {str(e)}")
            metrics.success = False
            metrics.error_message = str(e)
            
        metrics.end_time = datetime.now()
        metrics.duration = metrics.end_time - metrics.start_time
        return metrics
        
    def _execute_step(self, site_name: str, site_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Execute the specific step logic. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _execute_step")
        
    def cleanup(self, site_name: str, site_path: str) -> None:
        """Clean up after test completion."""
        pass


class TestFramework:
    """Main test framework orchestrator."""
    
    def __init__(self, config_path: str, output_dir: Optional[str] = None):
        self.config_path = config_path
        self.output_dir = output_dir or "test_framework/results"
        self.logger = logging.getLogger(__name__)
        
        # Load test configuration
        with open(config_path, 'r') as f:
            self.test_config = yaml.safe_load(f)
            
        self.test_set = self.test_config['test_set']
        self.test_name = self.test_set['name']
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize results storage
        self.results = {
            'test_info': {
                'name': self.test_name,
                'description': self.test_set['description'],
                'start_time': None,
                'end_time': None,
                'total_duration': None,
                'sites_tested': self.test_set['sites'],
                'steps_tested': [s['name'] for s in self.test_set['steps'] if s['enabled']]
            },
            'site_results': {},
            'summary_statistics': {}
        }
        
    def run_tests(self) -> bool:
        """Run all tests defined in the configuration."""
        self.logger.info(f"Starting test suite: {self.test_name}")
        self.results['test_info']['start_time'] = datetime.now().isoformat()
        
        try:
            # Verify prerequisites
            if not self._verify_prerequisites():
                return False
            
            # Run tests for each site
            overall_success = True
            for site_name in self.test_set['sites']:
                site_success = self._run_site_tests(site_name)
                overall_success = overall_success and site_success
                
            # Generate summary
            self._generate_summary()
            
            # Save results
            self._save_results()
            
            return overall_success
            
        except Exception as e:
            self.logger.error(f"Test suite failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
        finally:
            self.results['test_info']['end_time'] = datetime.now().isoformat()
            
    def _verify_prerequisites(self) -> bool:
        """Verify that all prerequisites are met."""
        self.logger.info("Verifying test prerequisites...")
        
        # Check mounts
        try:
            mount_verification.verify_required_mounts()
            self.logger.info("✅ Cloud storage mounts verified")
        except Exception as e:
            self.logger.error(f"❌ Mount verification failed: {e}")
            return False
            
        return True
        
    def _run_site_tests(self, site_name: str) -> bool:
        """Run all enabled tests for a specific site."""
        self.logger.info(f"Running tests for site: {site_name}")
        
        site_results = {
            'site_name': site_name,
            'start_time': datetime.now().isoformat(),
            'step_results': {},
            'overall_success': True
        }
        
        try:
            # Get site configuration
            pipeline_config = PipelineConfig()
            pipeline_config['site_name'] = site_name
            site_path = pipeline_config.get_site_path()
            
            # Run each enabled step
            for step_config in self.test_set['steps']:
                if not step_config['enabled']:
                    continue
                    
                step_name = step_config['name']
                self.logger.info(f"  Testing step: {step_name}")
                
                # Create step tester
                tester = self._create_step_tester(step_name, step_config)
                
                # Run the test
                metrics = tester.run_test(site_name, site_path)
                site_results['step_results'][step_name] = metrics.to_dict()
                
                if not metrics.success:
                    site_results['overall_success'] = False
                    self.logger.error(f"  ❌ Step {step_name} failed: {metrics.error_message}")
                else:
                    self.logger.info(f"  ✅ Step {step_name} completed in {metrics.duration}")
                    
                # Cleanup if configured
                if self.test_set['execution']['cleanup_after']:
                    tester.cleanup(site_name, site_path)
                    
        except Exception as e:
            self.logger.error(f"Site test failed for {site_name}: {str(e)}")
            site_results['overall_success'] = False
            site_results['error_message'] = str(e)
            
        site_results['end_time'] = datetime.now().isoformat()
        self.results['site_results'][site_name] = site_results
        
        return site_results['overall_success']
        
    def _create_step_tester(self, step_name: str, step_config: Dict[str, Any]) -> PipelineStepTester:
        """Create appropriate tester for the given step."""
        # Import step-specific testers with proper path
        import sys
        import os
        test_framework_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, test_framework_dir)
        
        from tests.step_testers import (
            DownloadTester, CoregisterTester, CloudImputeTester,
            RGBNIRTester, UpsampleTester, NormalizeTester,
            SegmentTester, BoundaryExtractTester, BoundaryRefineTester,
            GeotransformTester, FilterShorelinesTester, TideModelTester,
            TideCorrectTester, GeoJSONTester
        )
        
        tester_classes = {
            'download': DownloadTester,
            'coregister': CoregisterTester,
            'cloud_impute': CloudImputeTester,
            'rgb_nir_creation': RGBNIRTester,
            'upsample': UpsampleTester,
            'normalize': NormalizeTester,
            'segment': SegmentTester,
            'boundary_extract': BoundaryExtractTester,
            'boundary_refine': BoundaryRefineTester,
            'geotransform': GeotransformTester,
            'filter_shorelines': FilterShorelinesTester,
            'tide_model': TideModelTester,
            'tide_correct': TideCorrectTester,
            'geojson_convert': GeoJSONTester
        }
        
        tester_class = tester_classes.get(step_name)
        if not tester_class:
            raise ValueError(f"No tester available for step: {step_name}")
            
        return tester_class(step_name, step_config)
        
    def _generate_summary(self) -> None:
        """Generate summary statistics from test results."""
        summary = {
            'total_sites': len(self.test_set['sites']),
            'successful_sites': 0,
            'failed_sites': 0,
            'step_statistics': {},
            'performance_summary': {}
        }
        
        # Analyze results
        all_durations = []
        step_success_counts = {}
        step_durations = {}
        
        for site_name, site_result in self.results['site_results'].items():
            if site_result['overall_success']:
                summary['successful_sites'] += 1
            else:
                summary['failed_sites'] += 1
                
            # Analyze step results
            for step_name, step_result in site_result['step_results'].items():
                if step_name not in step_success_counts:
                    step_success_counts[step_name] = {'success': 0, 'fail': 0}
                    step_durations[step_name] = []
                    
                if step_result['success']:
                    step_success_counts[step_name]['success'] += 1
                    if step_result['duration_seconds']:
                        step_durations[step_name].append(step_result['duration_seconds'])
                        all_durations.append(step_result['duration_seconds'])
                else:
                    step_success_counts[step_name]['fail'] += 1
                    
        # Calculate step statistics
        for step_name in step_success_counts:
            total = step_success_counts[step_name]['success'] + step_success_counts[step_name]['fail']
            success_rate = step_success_counts[step_name]['success'] / total if total > 0 else 0
            
            durations = step_durations[step_name]
            duration_stats = {}
            if durations:
                duration_stats = {
                    'mean_seconds': np.mean(durations),
                    'median_seconds': np.median(durations),
                    'std_seconds': np.std(durations),
                    'min_seconds': np.min(durations),
                    'max_seconds': np.max(durations)
                }
                
            summary['step_statistics'][step_name] = {
                'success_rate': success_rate,
                'total_runs': total,
                'successful_runs': step_success_counts[step_name]['success'],
                'failed_runs': step_success_counts[step_name]['fail'],
                'duration_statistics': duration_stats
            }
            
        # Overall performance summary
        if all_durations:
            summary['performance_summary'] = {
                'total_processing_time': sum(all_durations),
                'average_step_time': np.mean(all_durations),
                'fastest_step_time': np.min(all_durations),
                'slowest_step_time': np.max(all_durations)
            }
            
        self.results['summary_statistics'] = summary
        
    def _save_results(self) -> None:
        """Save test results to various formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{self.test_name}_{timestamp}"
        
        # Save detailed JSON results
        json_path = os.path.join(self.output_dir, f"{base_filename}_detailed.json")
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        self.logger.info(f"Detailed results saved to: {json_path}")
        
        # Save CSV summary
        csv_path = os.path.join(self.output_dir, f"{base_filename}_summary.csv")
        self._save_csv_summary(csv_path)
        self.logger.info(f"CSV summary saved to: {csv_path}")
        
        # Save step-by-step CSV
        steps_csv_path = os.path.join(self.output_dir, f"{base_filename}_steps.csv")
        self._save_steps_csv(steps_csv_path)
        self.logger.info(f"Step details saved to: {steps_csv_path}")
        
    def _save_csv_summary(self, csv_path: str) -> None:
        """Save summary statistics as CSV."""
        summary_data = []
        
        for site_name, site_result in self.results['site_results'].items():
            row = {
                'test_name': self.test_name,
                'site_name': site_name,
                'overall_success': site_result['overall_success'],
                'start_time': site_result['start_time'],
                'end_time': site_result['end_time'],
                'total_steps': len(site_result['step_results']),
                'successful_steps': sum(1 for r in site_result['step_results'].values() if r['success']),
                'failed_steps': sum(1 for r in site_result['step_results'].values() if not r['success'])
            }
            summary_data.append(row)
            
        df = pd.DataFrame(summary_data)
        df.to_csv(csv_path, index=False)
        
    def _save_steps_csv(self, csv_path: str) -> None:
        """Save detailed step results as CSV."""
        steps_data = []
        
        for site_name, site_result in self.results['site_results'].items():
            for step_name, step_result in site_result['step_results'].items():
                row = {
                    'test_name': self.test_name,
                    'site_name': site_name,
                    'step_name': step_name,
                    'success': step_result['success'],
                    'duration_seconds': step_result['duration_seconds'],
                    'start_time': step_result['start_time'],
                    'end_time': step_result['end_time'],
                    'memory_initial_mb': step_result.get('memory_usage_mb', {}).get('initial_mb'),
                    'memory_final_mb': step_result.get('memory_usage_mb', {}).get('final_mb'),
                    'memory_peak_mb': step_result.get('memory_usage_mb', {}).get('peak_mb'),
                    'error_message': step_result.get('error_message'),
                }
                
                # Add custom metrics
                if 'custom_metrics' in step_result:
                    for metric_name, metric_value in step_result['custom_metrics'].items():
                        row[f'metric_{metric_name}'] = metric_value
                        
                steps_data.append(row)
                
        df = pd.DataFrame(steps_data)
        df.to_csv(csv_path, index=False)


def main():
    """Command line interface for the test framework."""
    parser = argparse.ArgumentParser(
        description="Littoral Pipeline Test Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Path to test configuration file'
    )
    
    parser.add_argument(
        '--list-configs',
        action='store_true',
        help='List available test configurations'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        help='Output directory for test results (default: test_framework/results)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.list_configs:
        config_dir = Path("test_framework/config")
        configs = list(config_dir.glob("*.yaml"))
        print("Available test configurations:")
        for config in configs:
            print(f"  {config.stem}")
        return
        
    if not args.config:
        parser.error("--config is required (or use --list-configs to see available options)")
        
    # Run tests
    framework = TestFramework(args.config, args.output_dir)
    success = framework.run_tests()
    
    if success:
        print("✅ All tests completed successfully")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()