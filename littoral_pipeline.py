#!/usr/bin/env python3
"""
Littoral Processing Pipeline - Command Line Interface

This script provides a command-line interface for running the littoral processing
pipeline. It can process satellite imagery for coastal monitoring, including
downloading, cloud removal, segmentation, boundary extraction, and tidal correction.

Usage:
    python littoral_pipeline.py --site SITE_NAME [options]
    python littoral_pipeline.py --config CONFIG_FILE [options]
    python littoral_pipeline.py --create-config OUTPUT_FILE
    python littoral_pipeline.py --list-sites

Examples:
    # List all available sites
    python littoral_pipeline.py --list-sites

    # Run full pipeline for a site
    python littoral_pipeline.py --site Fenfushi

    # Run update pipeline (only new images)
    python littoral_pipeline.py --site Fenfushi --update

    # Run with custom configuration
    python littoral_pipeline.py --config my_config.yaml

    # Create example configuration file
    python littoral_pipeline.py --create-config pipeline_config.yaml

    # Skip certain steps
    python littoral_pipeline.py --site Fenfushi --skip-steps download,coregister

    # Run only a specific step
    python littoral_pipeline.py --site Fenfushi --step geojson_convert

    # Set custom log level
    python littoral_pipeline.py --site Fenfushi --log-level DEBUG
"""

import os
import sys
import argparse
import logging
from typing import Optional, List
import traceback

# Add pipeline modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.pipeline_config import PipelineConfig, create_example_config
from pipeline.pipeline_orchestrator import PipelineOrchestrator


def setup_basic_logging(log_level: str = "INFO"):
    """Setup basic logging before full pipeline logging is configured."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Littoral Processing Pipeline - Coastal Monitoring Satellite Image Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Main action arguments (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        '--site',
        type=str,
        help='Site name to process (must exist in site table)'
    )
    action_group.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (YAML or JSON)'
    )
    action_group.add_argument(
        '--create-config',
        type=str,
        metavar='OUTPUT_FILE',
        help='Create example configuration file and exit'
    )
    action_group.add_argument(
        '--list-sites',
        action='store_true',
        help='List all available sites from the site table and exit'
    )
    
    # Pipeline mode options
    mode_group = parser.add_argument_group('Pipeline Mode')
    mode_group.add_argument(
        '--update',
        action='store_true',
        help='Run in update mode (only process new images)'
    )
    mode_group.add_argument(
        '--full',
        action='store_true',
        help='Run full pipeline (default)'
    )
    
    # Processing options
    processing_group = parser.add_argument_group('Processing Options')
    processing_group.add_argument(
        '--skip-steps',
        type=str,
        help='Comma-separated list of steps to skip (e.g., download,coregister)'
    )
    processing_group.add_argument(
        '--step',
        type=str,
        help='Run only a specific pipeline step (e.g., geojson_convert, tide_correct)'
    )
    processing_group.add_argument(
        '--max-images',
        type=int,
        help='Maximum number of images to process'
    )
    processing_group.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    # Path overrides
    path_group = parser.add_argument_group('Path Overrides')
    path_group.add_argument(
        '--site-table',
        type=str,
        help='Path to site table CSV file'
    )
    path_group.add_argument(
        '--save-path',
        type=str,
        help='Base path for saving results'
    )
    
    # Advanced options
    advanced_group = parser.add_argument_group('Advanced Options')
    advanced_group.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration and show what would be done without executing'
    )
    advanced_group.add_argument(
        '--force',
        action='store_true',
        help='Force execution even if validation warnings exist'
    )
    advanced_group.add_argument(
        '--summary-only',
        action='store_true',
        help='Only generate and display summary report (no processing)'
    )
    
    return parser.parse_args()


def list_available_sites(site_table_path: Optional[str] = None):
    """
    List all available sites from the site table.
    
    Args:
        site_table_path: Optional path to site table CSV. If None, uses default path.
    """
    try:
        import pandas as pd
        
        # Use default site table path if not specified
        if site_table_path is None:
            config = PipelineConfig()
            site_table_path = config['site_table_path']
        
        # Check if site table exists
        if not os.path.exists(site_table_path):
            print(f"❌ Site table not found: {site_table_path}")
            return
        
        # Load site data directly with pandas
        sites_df = pd.read_csv(site_table_path)
        
        if 'site_name' not in sites_df.columns:
            print(f"❌ Site table does not contain 'site_name' column")
            print(f"   Available columns: {list(sites_df.columns)}")
            return
        
        site_names = sites_df['site_name'].tolist()
        
        print(f"📍 Available Sites ({len(site_names)} total):")
        print(f"   Site table: {site_table_path}")
        print()
        
        # Display sites with additional information if available
        for i, site_name in enumerate(site_names, 1):
            site_data = sites_df[sites_df['site_name'] == site_name].iloc[0]
            
            # Get additional info if available
            info_parts = [f"{i:2d}. {site_name}"]
            
            if 'start' in sites_df.columns and pd.notna(site_data['start']):
                start_date = site_data['start']
                info_parts.append(f"start: {start_date}")
            
            if 'end' in sites_df.columns and pd.notna(site_data['end']):
                end_date = site_data['end']
                info_parts.append(f"end: {end_date}")
                
            if 'last_run' in sites_df.columns and pd.notna(site_data['last_run']):
                last_run = site_data['last_run']
                info_parts.append(f"last run: {last_run}")
            
            print("   " + " | ".join(info_parts))
        
        print(f"\n💡 Usage: python {os.path.basename(__file__)} --site SITE_NAME")
        
    except Exception as e:
        print(f"❌ Error reading site table: {e}")
        logging.debug(f"Full error: {traceback.format_exc()}")


def create_config_from_args(args) -> PipelineConfig:
    """
    Create pipeline configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configured PipelineConfig object
    """
    if args.config:
        # Load from configuration file
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Configuration file not found: {args.config}")
        config = PipelineConfig(args.config)
    else:
        # Create from command line arguments
        config = PipelineConfig()
        
        # Set site name
        if args.site:
            config['site_name'] = args.site
    
    # Apply command line overrides
    if args.update:
        config['pipeline']['run_mode'] = 'update'
    elif args.full:
        config['pipeline']['run_mode'] = 'full'
    
    if args.skip_steps:
        skip_list = [step.strip() for step in args.skip_steps.split(',')]
        config['pipeline']['skip_steps'] = skip_list
    
    if args.step:
        config['pipeline']['run_mode'] = 'single_step'
        config['pipeline']['single_step'] = args.step.strip()
    
    if args.max_images:
        config['pipeline']['max_images'] = args.max_images
    
    if args.log_level:
        config['pipeline']['log_level'] = args.log_level
    
    if args.site_table:
        config['site_table_path'] = args.site_table
    
    if args.save_path:
        config['save_path'] = args.save_path
    
    return config


def validate_and_display_config(config: PipelineConfig, force: bool = False) -> bool:
    """
    Validate configuration and display settings.
    
    Args:
        config: Pipeline configuration
        force: Force execution even with warnings
        
    Returns:
        True if validation passes or forced, False otherwise
    """
    print("=" * 60)
    print("PIPELINE CONFIGURATION")
    print("=" * 60)
    print(f"Site Name: {config['site_name']}")
    print(f"Run Mode: {config['pipeline']['run_mode']}")
    print(f"Save Path: {config['save_path']}")
    print(f"Site Table: {config['site_table_path']}")
    print(f"Log Level: {config['pipeline']['log_level']}")
    
    if config['pipeline']['skip_steps']:
        print(f"Skip Steps: {', '.join(config['pipeline']['skip_steps'])}")
    
    if config['pipeline'].get('single_step'):
        print(f"Single Step: {config['pipeline']['single_step']}")
    
    if config['pipeline']['max_images']:
        print(f"Max Images: {config['pipeline']['max_images']}")
    
    print("=" * 60)
    
    # Validate configuration
    errors = config.validate_config()
    if errors:
        print("CONFIGURATION ERRORS:")
        print("=" * 30)
        for error in errors:
            print(f"❌ {error}")
        print("=" * 60)
        
        if not force:
            print("Use --force to proceed despite errors")
            return False
        else:
            print("⚠️  Proceeding despite errors (--force specified)")
    else:
        print("✅ Configuration validation passed")
        print("=" * 60)
    
    return True


def main():
    """Main entry point for the pipeline CLI."""
    args = parse_arguments()
    
    # Handle config creation
    if args.create_config:
        try:
            create_example_config(args.create_config)
            print(f"✅ Example configuration created: {args.create_config}")
            print("\nEdit the configuration file and run:")
            print(f"python {sys.argv[0]} --config {args.create_config}")
            return 0
        except Exception as e:
            print(f"❌ Error creating configuration: {e}")
            return 1
    
    # Handle list sites
    if args.list_sites:
        list_available_sites(args.site_table)
        return 0
    
    # Setup basic logging
    setup_basic_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Create configuration
        config = create_config_from_args(args)
        
        # Handle summary-only mode
        if args.summary_only:
            orchestrator = PipelineOrchestrator(config)
            summary = orchestrator.generate_summary_report({})
            print(summary)
            return 0
        
        # Validate and display configuration
        if not validate_and_display_config(config, args.force):
            # Display available sites if site not found
            if args.site and os.path.exists(config['site_table_path']):
                print("\n")
                list_available_sites(config['site_table_path'])
            return 1
        
        # Handle dry run
        if args.dry_run:
            print("\n🔍 DRY RUN - No processing will be performed")
            print("✅ Configuration is valid and pipeline would execute successfully")
            return 0
        
        # Create and run pipeline orchestrator
        print(f"\n🚀 Starting pipeline execution...")
        orchestrator = PipelineOrchestrator(config)
        
        # Execute pipeline
        results = orchestrator.run()
        
        # Generate and display summary
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION COMPLETE")
        print("=" * 60)
        
        summary = orchestrator.generate_summary_report(results)
        print(summary)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Pipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.log_level == 'DEBUG':
            logger.error(traceback.format_exc())
        else:
            print(f"\n❌ Pipeline failed: {e}")
            print("Use --log-level DEBUG for detailed error information")
        return 1


if __name__ == "__main__":
    sys.exit(main())