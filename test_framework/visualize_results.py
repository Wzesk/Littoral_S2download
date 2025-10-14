#!/usr/bin/env python3
"""
Test Results Visualization and Analysis Utilities

This script provides utilities for analyzing and visualizing test results
from the Littoral Pipeline Test Framework.

Usage:
    python visualize_results.py --results-dir test_framework/results/
    python visualize_results.py --compare baseline_results.csv new_results.csv
    python visualize_results.py --summary test_framework/results/
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for better plots
plt.style.use('default')
sns.set_palette("husl")


class TestResultsAnalyzer:
    """Analyzer for test results with visualization capabilities."""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.summary_files = list(self.results_dir.glob("*_summary.csv"))
        self.steps_files = list(self.results_dir.glob("*_steps.csv"))
        self.detailed_files = list(self.results_dir.glob("*_detailed.json"))
        
    def load_latest_results(self) -> tuple:
        """Load the most recent test results."""
        if not self.summary_files:
            raise ValueError("No summary CSV files found in results directory")
            
        # Get the most recent files
        latest_summary = max(self.summary_files, key=os.path.getctime)
        latest_steps = max(self.steps_files, key=os.path.getctime)
        
        summary_df = pd.read_csv(latest_summary)
        steps_df = pd.read_csv(latest_steps)
        
        return summary_df, steps_df
        
    def generate_performance_report(self, output_dir: Optional[str] = None) -> str:
        """Generate a comprehensive performance report."""
        if output_dir is None:
            output_dir = self.results_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
        summary_df, steps_df = self.load_latest_results()
        
        # Create report
        report_path = output_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        html_content = self._generate_html_report(summary_df, steps_df)
        
        with open(report_path, 'w') as f:
            f.write(html_content)
            
        # Generate plots
        self._create_performance_plots(summary_df, steps_df, output_dir)
        
        return str(report_path)
        
    def _generate_html_report(self, summary_df: pd.DataFrame, steps_df: pd.DataFrame) -> str:
        """Generate HTML performance report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Littoral Pipeline Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: #e8f4f8; border-radius: 5px; }}
                .success {{ color: green; }}
                .failure {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .chart-container {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Littoral Pipeline Performance Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h2>Overall Summary</h2>
            <div class="metric">
                <strong>Total Sites Tested:</strong> {len(summary_df)}
            </div>
            <div class="metric">
                <strong>Successful Sites:</strong> <span class="success">{len(summary_df[summary_df['overall_success'] == True])}</span>
            </div>
            <div class="metric">
                <strong>Failed Sites:</strong> <span class="failure">{len(summary_df[summary_df['overall_success'] == False])}</span>
            </div>
            <div class="metric">
                <strong>Overall Success Rate:</strong> {len(summary_df[summary_df['overall_success'] == True]) / len(summary_df) * 100:.1f}%
            </div>
            
            <h2>Site Performance Summary</h2>
            {summary_df.to_html(index=False, classes='table')}
            
            <h2>Step Performance Analysis</h2>
            {self._generate_step_performance_table(steps_df)}
            
            <h2>Duration Analysis</h2>
            {self._generate_duration_analysis(steps_df)}
            
        </body>
        </html>
        """
        return html
        
    def _generate_step_performance_table(self, steps_df: pd.DataFrame) -> str:
        """Generate step performance analysis table."""
        step_stats = []
        
        for step_name in steps_df['step_name'].unique():
            step_data = steps_df[steps_df['step_name'] == step_name]
            success_rate = len(step_data[step_data['success'] == True]) / len(step_data) * 100
            avg_duration = step_data['duration_seconds'].mean()
            
            step_stats.append({
                'Step': step_name,
                'Total Runs': len(step_data),
                'Success Rate (%)': f"{success_rate:.1f}",
                'Avg Duration (s)': f"{avg_duration:.1f}" if not pd.isna(avg_duration) else "N/A",
                'Min Duration (s)': f"{step_data['duration_seconds'].min():.1f}" if not pd.isna(step_data['duration_seconds'].min()) else "N/A",
                'Max Duration (s)': f"{step_data['duration_seconds'].max():.1f}" if not pd.isna(step_data['duration_seconds'].max()) else "N/A"
            })
            
        step_stats_df = pd.DataFrame(step_stats)
        return step_stats_df.to_html(index=False, classes='table')
        
    def _generate_duration_analysis(self, steps_df: pd.DataFrame) -> str:
        """Generate duration analysis summary."""
        valid_durations = steps_df.dropna(subset=['duration_seconds'])
        
        if len(valid_durations) == 0:
            return "<p>No duration data available.</p>"
            
        total_time = valid_durations['duration_seconds'].sum()
        avg_time = valid_durations['duration_seconds'].mean()
        median_time = valid_durations['duration_seconds'].median()
        
        return f"""
        <div class="metric">
            <strong>Total Processing Time:</strong> {total_time:.1f} seconds ({total_time/60:.1f} minutes)
        </div>
        <div class="metric">
            <strong>Average Step Time:</strong> {avg_time:.1f} seconds
        </div>
        <div class="metric">
            <strong>Median Step Time:</strong> {median_time:.1f} seconds
        </div>
        """
        
    def _create_performance_plots(self, summary_df: pd.DataFrame, steps_df: pd.DataFrame, output_dir: Path):
        """Create performance visualization plots."""
        # Plot 1: Success rate by step
        plt.figure(figsize=(12, 6))
        step_success = steps_df.groupby('step_name')['success'].agg(['count', 'sum']).reset_index()
        step_success['success_rate'] = step_success['sum'] / step_success['count'] * 100
        
        plt.bar(range(len(step_success)), step_success['success_rate'])
        plt.xlabel('Pipeline Steps')
        plt.ylabel('Success Rate (%)')
        plt.title('Success Rate by Pipeline Step')
        plt.xticks(range(len(step_success)), step_success['step_name'], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / 'success_rates_by_step.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Duration by step
        valid_durations = steps_df.dropna(subset=['duration_seconds'])
        if len(valid_durations) > 0:
            plt.figure(figsize=(12, 6))
            step_durations = valid_durations.groupby('step_name')['duration_seconds'].mean()
            
            plt.bar(range(len(step_durations)), step_durations.values)
            plt.xlabel('Pipeline Steps')
            plt.ylabel('Average Duration (seconds)')
            plt.title('Average Duration by Pipeline Step')
            plt.xticks(range(len(step_durations)), step_durations.index, rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_dir / 'duration_by_step.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot 3: Site comparison
        if len(summary_df) > 1:
            plt.figure(figsize=(10, 6))
            site_success = summary_df.groupby('site_name').agg({
                'successful_steps': 'mean',
                'total_steps': 'mean'
            }).reset_index()
            site_success['success_rate'] = site_success['successful_steps'] / site_success['total_steps'] * 100
            
            plt.bar(range(len(site_success)), site_success['success_rate'])
            plt.xlabel('Sites')
            plt.ylabel('Success Rate (%)')
            plt.title('Success Rate by Site')
            plt.xticks(range(len(site_success)), site_success['site_name'], rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_dir / 'success_rates_by_site.png', dpi=300, bbox_inches='tight')
            plt.close()
            
    def compare_test_runs(self, baseline_file: str, comparison_file: str, output_dir: Optional[str] = None) -> str:
        """Compare two test runs and generate a comparison report."""
        baseline_df = pd.read_csv(baseline_file)
        comparison_df = pd.read_csv(comparison_file)
        
        if output_dir is None:
            output_dir = self.results_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
        # Generate comparison analysis
        comparison_report = self._generate_comparison_report(baseline_df, comparison_df)
        
        report_path = output_dir / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_path, 'w') as f:
            f.write(comparison_report)
            
        return str(report_path)
        
    def _generate_comparison_report(self, baseline_df: pd.DataFrame, comparison_df: pd.DataFrame) -> str:
        """Generate HTML comparison report."""
        # Calculate metrics for both datasets
        baseline_success_rate = len(baseline_df[baseline_df['success'] == True]) / len(baseline_df) * 100
        comparison_success_rate = len(comparison_df[comparison_df['success'] == True]) / len(comparison_df) * 100
        
        baseline_avg_duration = baseline_df['duration_seconds'].mean()
        comparison_avg_duration = comparison_df['duration_seconds'].mean()
        
        success_rate_change = comparison_success_rate - baseline_success_rate
        duration_change = comparison_avg_duration - baseline_avg_duration
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Results Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: #e8f4f8; border-radius: 5px; }}
                .improvement {{ color: green; }}
                .regression {{ color: red; }}
                .neutral {{ color: orange; }}
            </style>
        </head>
        <body>
            <h1>Test Results Comparison Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Overall Comparison</h2>
            <div class="metric">
                <strong>Baseline Success Rate:</strong> {baseline_success_rate:.1f}%
            </div>
            <div class="metric">
                <strong>Comparison Success Rate:</strong> {comparison_success_rate:.1f}%
            </div>
            <div class="metric">
                <strong>Success Rate Change:</strong> 
                <span class="{'improvement' if success_rate_change > 0 else 'regression' if success_rate_change < 0 else 'neutral'}">
                    {success_rate_change:+.1f}%
                </span>
            </div>
            
            <div class="metric">
                <strong>Baseline Avg Duration:</strong> {baseline_avg_duration:.1f}s
            </div>
            <div class="metric">
                <strong>Comparison Avg Duration:</strong> {comparison_avg_duration:.1f}s
            </div>
            <div class="metric">
                <strong>Duration Change:</strong> 
                <span class="{'improvement' if duration_change < 0 else 'regression' if duration_change > 0 else 'neutral'}">
                    {duration_change:+.1f}s
                </span>
            </div>
            
        </body>
        </html>
        """
        return html


def main():
    """Command line interface for test results analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze and visualize Littoral Pipeline test results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--results-dir', '-r',
        default='test_framework/results',
        help='Directory containing test results'
    )
    
    parser.add_argument(
        '--generate-report',
        action='store_true',
        help='Generate comprehensive performance report'
    )
    
    parser.add_argument(
        '--compare',
        nargs=2,
        metavar=('BASELINE', 'COMPARISON'),
        help='Compare two test result files'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        help='Output directory for generated reports'
    )
    
    args = parser.parse_args()
    
    analyzer = TestResultsAnalyzer(args.results_dir)
    
    if args.generate_report:
        try:
            report_path = analyzer.generate_performance_report(args.output_dir)
            print(f"✅ Performance report generated: {report_path}")
        except Exception as e:
            print(f"❌ Failed to generate report: {e}")
            return 1
            
    if args.compare:
        try:
            report_path = analyzer.compare_test_runs(
                args.compare[0], 
                args.compare[1], 
                args.output_dir
            )
            print(f"✅ Comparison report generated: {report_path}")
        except Exception as e:
            print(f"❌ Failed to generate comparison: {e}")
            return 1
            
    if not args.generate_report and not args.compare:
        # Default: show summary of available results
        print(f"Test results directory: {analyzer.results_dir}")
        print(f"Summary files found: {len(analyzer.summary_files)}")
        print(f"Steps files found: {len(analyzer.steps_files)}")
        print(f"Detailed files found: {len(analyzer.detailed_files)}")
        
        if analyzer.summary_files:
            print("\nAvailable test results:")
            for f in analyzer.summary_files:
                print(f"  {f.name}")
                
    return 0


if __name__ == "__main__":
    sys.exit(main())