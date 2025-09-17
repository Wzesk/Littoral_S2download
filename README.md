<div align="center">
  <h1 style="font-size: 3em;">LITTORAL</h1>
  <h2>Littoral_S2Download Module</h2>
</div>

---

## Overview

This repository contains the complete Littoral shoreline analysis pipeline for processing Sentinel-2 satellite imagery to extract and analyze coastal shorelines. The pipeline combines automated satellite data processing with advanced machine learning techniques to produce accurate, tidally-corrected shoreline boundaries.

The pipeline is designed with two categories of processing steps:

**Modular ML Components** (Open Source & Standardized Interfaces):
- **Coregistration**: Aligns satellite images to reference coordinates
- **Cloud Imputation**: Removes cloud artifacts using advanced inpainting
- **Upsampling**: Enhances image resolution using Real-ESRGAN
- **Segmentation**: Identifies land/water boundaries using YOLO v8

**Process Management Steps** (Pipeline-Specific):
- Project setup and folder structure management
- Satellite imagery download and filtering
- Image quality assessment and refinement
- Geospatial transformations and coordinate mapping
- Shoreline filtering and quality control
- Tidal corrections and temporal analysis

### Core Files

| File | Purpose | Details |
|------|---------|---------|
| `ee_s2.py` | S2 Download Engine | - Connects to Google Earth Engine<br>- Filters S2 collections by date and cloud coverage<br>- Downloads RGB+NIR bands<br>- Generates cloud masks |
| `tario.py` | Storage Management | - Efficient TAR archive handling<br>- Compresses downloaded imagery<br>- Manages image retrieval |
| `littoral_sites.py` | Site Tracking | - Manages site coordinates<br>- Tracks processing status<br>- Handles site metadata |

## Installation

### Option 1: Conda (Current Method)
```bash
# Create and activate conda environment
conda create --name littoral_pipeline python=3.10
conda activate littoral_pipeline
conda env update --file environment.yml --prune
```

### Option 2: Hatch (Modern Alternative)
```bash
# Install hatch if not already installed
pip install hatch

# Create and activate environment
hatch env create
hatch shell
```

## Repository Structure
```
littoral_s2download/
├── src/
│   └── littoral/           # Main package
│       ├── __init__.py
│       ├── ee_s2.py        # S2 download engine
│       ├── tario.py        # TAR archive handling
│       └── littoral_sites.py # Site management
├── environment.yml         # Conda environment
├── pyproject.toml         # Hatch configuration
└── littoral_sites.csv     # Site coordinates
```

## Pipeline Architecture

The Littoral pipeline is implemented in the `littoral_pipeline.ipynb` notebook and consists of two types of processing components:

### Modular ML Components

These components use standardized interfaces and are designed to be easily updated as ML models improve:

#### 1. Coregistration (`littoral_coreg`)
- **Purpose**: Aligns satellite images to reference coordinates for consistent analysis
- **Input**: Raw satellite imagery from multiple acquisition dates
- **Output**: Geometrically aligned images with consistent spatial reference
- **Interface**: `LittoralCoregistration.run()`
- **Technology**: Advanced feature matching and geometric transformation

#### 2. Cloud Imputation (`littoral_cloud_impute`) 
- **Purpose**: Removes cloud artifacts and fills gaps using intelligent inpainting
- **Input**: Satellite images with cloud contamination
- **Output**: Cloud-free imagery with reconstructed surface features
- **Interface**: `vpint_cloud_impute.batch_remove_clouds_folder()`
- **Technology**: VPint (Video/Photo Inpainting) deep learning models

#### 3. Upsampling (`Real-ESRGAN`)
- **Purpose**: Enhances image resolution for improved boundary detection
- **Input**: Cloud-free satellite imagery at native resolution
- **Output**: Super-resolved images with enhanced spatial detail
- **Interface**: `RealESRGAN.upsample_folder()`
- **Technology**: Real-ESRGAN generative adversarial networks

#### 4. Segmentation (`littoral_segment`)
- **Purpose**: Identifies land/water boundaries using semantic segmentation
- **Input**: High-resolution, cloud-free imagery
- **Output**: Binary masks delineating shoreline boundaries
- **Interface**: `YOLOV8.mask_from_folder()`
- **Technology**: YOLO v8 object detection and segmentation

### Process Management Steps

These steps handle the pipeline-specific data management and analysis workflows:

#### 1. Project Setup and Folder Structure
- Site configuration and parameter loading
- Directory structure creation for organized data storage
- Integration with Google Cloud Storage for large-scale processing

#### 2. Initial Download (`ee_s2`)
- Sentinel-2 imagery acquisition from Google Earth Engine
- Cloud coverage filtering and quality assessment
- Multi-spectral band extraction (RGB + NIR)

#### 3. Image Filtering and Quality Control
- Automated detection of defective or corrupted images
- Image normalization and enhancement
- Quality metrics calculation and reporting

#### 4. Boundary Refinement (`littoral_refine`)
- Shoreline extraction from segmentation masks
- B-spline curve fitting for smooth boundary representation
- Outlier detection and filtering using spatial clustering

#### 5. Geospatial Transformations (`geo_transform`)
- Conversion from pixel coordinates to geographic coordinates
- Integration with satellite metadata for accurate georeferencing
- Coordinate system transformations and projections

#### 6. Shoreline Filtering and Analysis
- Statistical analysis of shoreline positions over time
- 3D visualization of temporal shoreline changes
- Anomaly detection and quality filtering

#### 7. Tidal Corrections (`littoral_tide_correction`)
- Tidal modeling using FES2022b global tide models
- Calculation of water level corrections for each acquisition time
- Application of tidal offsets to normalize shoreline positions

## Complete Pipeline Workflow

The `littoral_pipeline.ipynb` notebook executes the following sequence of operations:

| Step | Component Type | Module/Function | Input | Output | Purpose |
|------|----------------|-----------------|-------|--------|---------|
| 1 | Process | Site Setup | Site coordinates | Project structure | Initialize processing environment |
| 2 | Process | `ee_s2.get_filtered_image_collection()` | AOI, date range | S2 image collection | Download satellite imagery |
| 3 | **Modular** | `LittoralCoregistration.run()` | Raw imagery | Aligned images | Geometric alignment |
| 4 | **Modular** | `vpint_cloud_impute.batch_remove_clouds_folder()` | Aligned images | Cloud-free images | Remove cloud artifacts |
| 5 | Process | `ee_s2.process_cloud_imputed_images()` | Cloud-free TIFFs | RGB/NIR pairs | Create analysis-ready images |
| 6 | **Modular** | `RealESRGAN.upsample_folder()` | Standard resolution | High resolution | Enhance image detail |
| 7 | Process | Image Normalization | Raw upsampled | Normalized images | Quality filtering and enhancement |
| 8 | **Modular** | `YOLOV8.mask_from_folder()` | Normalized images | Binary masks | Land/water segmentation |
| 9 | Process | `extract_boundary.get_shorelines_from_folder()` | Binary masks | Pixel coordinates | Extract shoreline polylines |
| 10 | Process | `refine_boundary.refine_shorelines()` | Raw polylines | Refined boundaries | Smooth and filter shorelines |
| 11 | Process | `geo_transform.batch_geotransform()` | Pixel coordinates | Geographic coordinates | Convert to lat/lon |
| 12 | Process | Statistical Filtering | All shorelines | Filtered shorelines | Remove outliers and artifacts |
| 13 | Process | `littoral_tide_correction.model_tides()` | Timestamps, location | Tidal predictions | Calculate water levels |
| 14 | Process | `apply_tidal_corrections_to_shorelines()` | Shorelines + tides | Corrected shorelines | Normalize for tidal effects |

**Bold** steps indicate **Modular ML Components** with standardized interfaces that can be easily updated as models improve.

## Usage

### Running the Complete Pipeline

The primary way to use this system is through the `littoral_pipeline.ipynb` notebook:

```python
# 1. Setup environment and site selection
import sys
sys.path.append('/path/to/Littoral_S2download/src')
from littoral import ee_s2, littoral_sites

# Load site configuration
site_table_path = "/path/to/littoral_sites.csv"
site_name = "your_site_name"

# 2. Download imagery
ee_s2.connect()
proj_params = littoral_sites.load_site_parameters_cg(site_name, save_path, site_table_path)
se2_col = ee_s2.get_filtered_image_collection(proj_params)
results = ee_s2.process_collection_images_tofiles(proj_params, se2_col)

# 3. Run modular ML components
# Coregistration
from littoral_coregistration import LittoralCoregistration
coreg = LittoralCoregistration(top_level_dir)
results = coreg.run()

# Cloud imputation
import vpint_cloud_impute
vpint_cloud_impute.batch_remove_clouds_folder(folder_path)

# Upsampling
from RealESRGAN.model import RealESRGAN
upsampled_images = RealESRGAN.upsample_folder(input_folder)

# Segmentation
from littoral_segment.seg_models.yolov8_seg import YOLOV8
yolo_model = YOLOV8(folder='/path/to/yolo8_params')
mask_paths = yolo_model.mask_from_folder(input_folder)

# 4. Process management steps continue...
```

### Individual Component Usage

```python
from littoral.ee_s2 import connect, get_image_collection, process_collection_images

# Initialize Earth Engine
connect(project_id='your-project-id')

# Configure extraction
data = {
    "aoi": [lon1, lat1, lon2, lat2], 
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "project_name": "shoreline_study",
    "path": "./output"
}

# Extract imagery
collection = get_image_collection(data)
results = process_collection_images(data, collection)
```

## Key Features

- **Modular Architecture**: ML components can be easily swapped as models improve
- **Standardized Interfaces**: Simple, consistent APIs for each processing step
- **End-to-End Pipeline**: Complete workflow from satellite data to tide-corrected shorelines
- **Quality Control**: Automated filtering and statistical analysis throughout
- **Scalable Processing**: Designed for batch processing of multiple coastal sites
- **Open Source Components**: Modular ML steps available for community improvement

## Dependencies

The pipeline integrates multiple specialized libraries:

- **Earth Engine API**: Satellite data access
- **littoral_coreg**: Image coregistration
- **littoral_cloud_impute**: Cloud removal using VPint
- **Real-ESRGAN**: Super-resolution enhancement  
- **littoral_segment**: YOLO v8 segmentation
- **littoral_refine**: Boundary extraction and refinement
- **pyTMD**: Tidal modeling and corrections

## Output

The pipeline produces:
- **Raw Imagery**: Downloaded Sentinel-2 RGB and NIR bands
- **Processed Images**: Cloud-free, coregistered, super-resolved imagery
- **Segmentation Masks**: Binary land/water classifications
- **Shoreline Boundaries**: Extracted polyline coordinates
- **Geospatial Data**: Geographic coordinates with metadata
- **Tidal Corrections**: Time-series of water level adjustments
- **Final Shorelines**: Tidally-corrected boundary positions ready for analysis
