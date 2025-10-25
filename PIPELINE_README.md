# Littoral Processing Pipeline Script

A comprehensive Python script for processing satellite imagery in coastal monitoring applications. This pipeline automates the complete workflow from satellite image download to tidal-corrected shoreline extraction.

## Overview

The Littoral Processing Pipeline converts the Jupyter notebook workflow into a robust, configurable Python script that can:

- Process satellite imagery for coastal monitoring
- Run in full mode (complete pipeline) or update mode (only new images)
- Handle multiple sites with different configurations
- Provide comprehensive logging and progress tracking
- Generate detailed processing reports

## Pipeline Steps

The pipeline consists of 14 main processing steps:

1. **Download Imagery** - Download satellite images using Earth Engine
2. **Coregister** - Image coregistration for alignment
3. **Cloud Imputation** - Remove clouds from images using VPint
4. **RGB/NIR Creation** - Create RGB and NIR composites from cloud-free data
5. **Upsampling** - Enhance image resolution using Real-ESRGAN
6. **Normalization** - Normalize images and remove defective ones
7. **Segmentation** - Segment shorelines using YOLO
8. **Boundary Extraction** - Extract boundaries from segmentation masks
9. **Boundary Refinement** - Refine and smooth extracted boundaries
10. **Geotransformation** - Transform boundaries to geographic coordinates
11. **Shoreline Filtering** - Filter and validate shorelines
12. **Tidal Modeling** - Model tidal corrections using FES2022
13. **Tidal Correction** - Apply tidal corrections to shorelines
14. **GeoJSON Convert** - Convert processed shorelines to GeoJSON format and upload metadata

## Installation

### Prerequisites

- Python 3.8+
- Access to Google Earth Engine
- Google Cloud Storage buckets mounted via gcsfuse
- All required dependencies from the original notebook environment

### Required Python Packages

```bash
# Core packages
pip install pandas numpy matplotlib pillow opencv-python
pip install scikit-learn geomdl pyproj

# Google Earth Engine
pip install earthengine-api

# Additional packages (install as needed)
pip install pyyaml  # For YAML configuration files
```

### System Dependencies

- `gcsfuse` for mounting Google Cloud Storage buckets
- Earth Engine authentication configured
- Access to required model files and data

## Usage

### Basic Usage

```bash
# Run full pipeline for a site
python littoral_pipeline.py --site Fenfushi

# Run update pipeline (only process new images)
python littoral_pipeline.py --site Fenfushi --update

# Use custom configuration file
python littoral_pipeline.py --config my_config.yaml
```

### Create Configuration File

```bash
# Create example configuration file
python littoral_pipeline.py --create-config pipeline_config.yaml

# Edit the configuration file as needed
# Then run with:
python littoral_pipeline.py --config pipeline_config.yaml
```

### Advanced Options

```bash
# Skip certain processing steps
python littoral_pipeline.py --site Fenfushi --skip-steps download,coregister

# Set custom log level
python littoral_pipeline.py --site Fenfushi --log-level DEBUG

# Dry run (validate configuration without processing)
python littoral_pipeline.py --site Fenfushi --dry-run

# Force execution despite validation warnings
python littoral_pipeline.py --site Fenfushi --force

# Generate summary report only
python littoral_pipeline.py --site Fenfushi --summary-only
```

## Configuration

The pipeline uses a configuration system that allows you to customize all settings:

### Configuration File Structure

```yaml
# Basic settings
site_name: "Fenfushi"
site_table_path: "/path/to/littoral_sites.csv"
save_path: "/path/to/geotools_sites"

# Folder structure
folders:
  tiff: "TARGETS"
  upsampled: "UP"
  normalized: "NORMALIZED"
  # ... other folders

# Processing parameters
processing:
  normalization:
    min_threshold: 25
    max_threshold: 230
  filtering:
    defective_threshold: 50
  tide_correction:
    model: "fes2022b"
    beach_slope: 0.08
  # ... other parameters

# Pipeline control
pipeline:
  run_mode: "full"  # or "update"
  skip_steps: []
  log_level: "INFO"
```

### Command Line Overrides

Most configuration options can be overridden via command line arguments:

- `--site SITE_NAME` - Set site name
- `--update` - Run in update mode
- `--skip-steps STEPS` - Skip specific processing steps
- `--log-level LEVEL` - Set logging level
- `--site-table PATH` - Override site table path
- `--save-path PATH` - Override save path

## Update Mode

The pipeline supports an update mode that only processes new images:

```bash
python littoral_pipeline.py --site Fenfushi --update
```

In update mode, the pipeline:
1. Checks for existing processed images
2. Downloads only new images since last run
3. Processes only the new images through the complete pipeline
4. Merges results with existing data
5. Creates backups of previous results

## File Structure

```
Littoral_S2download/
├── littoral_pipeline.py          # Main CLI script
├── pipeline_config.py            # Configuration management
├── pipeline_functions.py         # Core pipeline functions
├── pipeline_advanced.py          # Advanced processing functions
├── pipeline_orchestrator.py      # Pipeline orchestration
├── PIPELINE_README.md            # This file
└── pipeline_config.yaml          # Example configuration
```

## Output

The pipeline generates:

1. **Processing CSV** - Tracks status of each image through all pipeline steps
2. **Log Files** - Detailed execution logs
3. **Summary Report** - Final pipeline execution summary
4. **Processed Data** - All intermediate and final results organized in folders

### Example Output Structure

```
geotools_sites/
└── site_name/
    ├── 20240101_processing.csv    # Processing tracking
    ├── pipeline.log               # Execution log
    ├── TARGETS/                   # Downloaded images
    ├── UP/                        # Upsampled images
    ├── NORMALIZED/                # Normalized images
    ├── MASK/                      # Segmentation masks
    ├── SHORELINE/                 # Extracted shorelines
    ├── TIDAL_CORRECTED/           # Final corrected shorelines
    └── tide_corrections.csv       # Tidal correction data
```

## Error Handling

The pipeline includes comprehensive error handling:

- Validation of all prerequisites before execution
- Graceful handling of individual step failures
- Detailed logging for debugging
- Automatic cleanup of mounted filesystems
- Backup creation in update mode

## Logging

Multiple logging levels are available:

- `DEBUG` - Detailed debugging information
- `INFO` - General information (default)
- `WARNING` - Warning messages only
- `ERROR` - Error messages only

Logs are written to both console and file (`pipeline.log` in site directory).

## Examples

### Complete Workflow Example

```bash
# 1. Create configuration file
python littoral_pipeline.py --create-config my_config.yaml

# 2. Edit configuration file for your site
# (edit my_config.yaml)

# 3. Run pipeline
python littoral_pipeline.py --config my_config.yaml

# 4. Later, run update to process new images
python littoral_pipeline.py --config my_config.yaml --update

# 5. Generate summary report
python littoral_pipeline.py --config my_config.yaml --summary-only
```

### Site-specific Example

```bash
# Run for specific site with custom settings
python littoral_pipeline.py --site Fenfushi \
  --log-level DEBUG \
  --skip-steps download \
  --save-path /custom/path
```

## Troubleshooting

### Common Issues

1. **Mount failures** - Ensure gcsfuse is installed and you have access to buckets
2. **Import errors** - Verify all dependencies are in Python path
3. **Earth Engine errors** - Check EE authentication
4. **Memory issues** - Consider processing fewer images with `--max-images`

### Debug Mode

Use debug mode for detailed error information:

```bash
python littoral_pipeline.py --site Fenfushi --log-level DEBUG
```

## Contributing

To extend the pipeline:

1. Add new processing functions to `pipeline_functions.py` or `pipeline_advanced.py`
2. Update the orchestrator in `pipeline_orchestrator.py`
3. Add configuration options to `pipeline_config.py`
4. Update CLI interface in `littoral_pipeline.py`

## License

This pipeline is based on the original Littoral Processing Notebook and inherits the same licensing terms.