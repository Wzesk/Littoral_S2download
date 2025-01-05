"""
Littoral S2Download Package
=========================

This package provides tools for downloading and processing Sentinel-2 satellite imagery
for shoreline analysis.

Modules
-------
ee_s2
    Sentinel-2 data extraction using Google Earth Engine
tario
    TAR archive management for efficient storage
littoral_sites
    Site tracking and parameter management
"""

from . import ee_s2
from . import tario
from . import littoral_sites

__version__ = "0.1.0"
