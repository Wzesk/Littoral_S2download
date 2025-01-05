<div align="center">
  <h1 style="font-size: 3em;">LITTORAL</h1>
  <h2>Littoral_S2Download Module</h2>
</div>

---

## Overview

This module handles the extraction and download of Sentinel-2 (S2) satellite imagery using Google Earth Engine. It is the first step in Littoral's shoreline analysis pipeline, responsible for:

1. Downloading S2 imagery for specified coastal regions
2. Filtering images based on cloud coverage
3. Storing imagery efficiently in TAR archives
4. Tracking download progress and site status

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

## Processing Pipeline (Notebooks Being Migrated)

This module handles the first stage (S2 download) of the processing pipeline. The full pipeline is being modularized into separate repositories:

| Stage | Current Location | Migration Status | Purpose |
|-------|-----------------|------------------|---------|
| **1. Data Download** | `ee_s2.py` | ✅ This Module | S2 imagery acquisition |
| 2. Cloud Removal | `01_inpainting.ipynb` | 🔄 In Progress | Cloud inpainting |
| 3. Image Repair | `01.2_sequential_repair.ipynb` | 🔄 In Progress | Artifact removal |
| 4. Enhancement | `02_upsample.ipynb` | 📋 Planned | Resolution improvement |
| 5. Segmentation | `03_segment.ipynb` | 📋 Planned | Feature detection |
| 6. Edge Detection | `04_boundary.ipynb` | 📋 Planned | Boundary extraction |
| 7. Refinement | `05_boundary_refinement.ipynb` | 📋 Planned | Boundary optimization |
| 8. Georeferencing | `06_map_to_world.ipynb` | 📋 Planned | Coordinate mapping |

**Note:** The original notebooks can be found [here](https://drive.google.com/drive/folders/1jkVuJzrKiYb_d1GstZRgSN9_ng9k3LZI?usp=sharing). Each stage is being converted into its own module repository to create a more maintainable and scalable processing pipeline.

### Usage Example

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
