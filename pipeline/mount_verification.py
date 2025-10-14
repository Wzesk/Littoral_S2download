"""
Mount verification utilities for the Littoral Pipeline

Simple functions to check if required cloud storage buckets are mounted
without attempting to mount them automatically.
"""

import os
import logging


def check_mount_status() -> dict:
    """
    Check if required cloud storage buckets are mounted.
    
    Returns:
        dict: Status of each required mount point
    """
    mount_points = {
        'geotools': '/home/walter_littor_al/geotools_sites',
        'tide_model': '/home/walter_littor_al/tide_model'
    }
    
    status = {}
    for name, path in mount_points.items():
        # Check if directory exists and is not empty (indicating it's mounted)
        if os.path.exists(path) and os.path.ismount(path):
            status[name] = True
        elif os.path.exists(path) and os.listdir(path):
            # Directory exists and has contents (likely mounted)
            status[name] = True
        else:
            status[name] = False
    
    return status


def verify_required_mounts(logger=None) -> bool:
    """
    Verify that all required cloud storage buckets are mounted.
    
    Args:
        logger: Optional logger instance
        
    Returns:
        bool: True if all required mounts are available
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    status = check_mount_status()
    
    all_mounted = True
    for name, is_mounted in status.items():
        if is_mounted:
            logger.info(f"✅ {name} bucket is mounted")
        else:
            logger.error(f"❌ {name} bucket is NOT mounted")
            all_mounted = False
    
    if not all_mounted:
        logger.error("")
        logger.error("MOUNT ERROR: Required cloud storage buckets are not mounted")
        logger.error("Please run the following command before starting the pipeline:")
        logger.error("  python mount_utils.py mount all")
        logger.error("")
        logger.error("After processing, you can unmount with:")
        logger.error("  python mount_utils.py unmount all")
        logger.error("")
    
    return all_mounted


def verify_tide_mount(logger=None) -> bool:
    """
    Verify that the tide model bucket is mounted.
    
    Args:
        logger: Optional logger instance
        
    Returns:
        bool: True if tide mount is available
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    status = check_mount_status()
    
    if status['tide_model']:
        logger.info("✅ Tide model bucket is mounted")
        return True
    else:
        logger.error("❌ Tide model bucket is NOT mounted")
        logger.error("Please ensure both buckets are mounted:")
        logger.error("  python mount_utils.py mount all")
        return False