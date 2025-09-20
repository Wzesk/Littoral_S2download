#!/usr/bin/env python3
"""
Simple utility functions for mounting and unmounting cloud storage buckets.

This provides convenient functions to mount/unmount buckets for manual inspection
without running the full pipeline.
"""

import os
import subprocess
import sys
from typing import Optional


def mount_geotools(mount_path: str = "/home/walter_littor_al/geotools_sites") -> bool:
    """
    Mount the geotools bucket for manual access.
    
    Args:
        mount_path: Path where to mount the bucket (default: /home/walter_littor_al/geotools_sites)
        
    Returns:
        True if mount successful, False otherwise
    """
    bucket = "coastal_geotools_demo"
    
    print(f"Mounting geotools bucket: {bucket} -> {mount_path}")
    
    try:
        # Unmount if already mounted
        unmount_cmd = f"fusermount -u {mount_path}"
        subprocess.run(unmount_cmd, shell=True, capture_output=True, text=True)
        
        # Create mount directory
        os.makedirs(mount_path, exist_ok=True)
        
        # Mount bucket
        cmd = f"gcsfuse {bucket} {mount_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully mounted {bucket} to {mount_path}")
            
            # Verify mount
            try:
                contents = os.listdir(mount_path)
                print(f"   Found {len(contents)} items in bucket")
                if contents:
                    print(f"   Sample items: {contents[:5]}")
                return True
            except Exception as e:
                print(f"❌ Mount appears successful but cannot list contents: {e}")
                return False
        else:
            print(f"❌ Failed to mount bucket")
            print(f"   Command: {cmd}")
            print(f"   Error: {result.stderr.strip() if result.stderr else 'No error details'}")
            return False
            
    except Exception as e:
        print(f"❌ Error during mount: {e}")
        return False


def mount_tide(mount_path: str = "/home/walter_littor_al/tide_model") -> bool:
    """
    Mount the tide model bucket for manual access.
    
    Args:
        mount_path: Path where to mount the bucket (default: /home/walter_littor_al/tide_model)
        
    Returns:
        True if mount successful, False otherwise
    """
    bucket = "aviso-fes2022"
    
    print(f"Mounting tide bucket: {bucket} -> {mount_path}")
    
    try:
        # Unmount if already mounted
        unmount_cmd = f"fusermount -u {mount_path}"
        subprocess.run(unmount_cmd, shell=True, capture_output=True, text=True)
        
        # Create mount directory
        os.makedirs(mount_path, exist_ok=True)
        
        # Mount bucket with special flags for AVISO data
        cmd = f"gcsfuse --implicit-dirs --dir-mode 777 --file-mode 777 {bucket} {mount_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully mounted {bucket} to {mount_path}")
            
            # Verify mount
            try:
                contents = os.listdir(mount_path)
                print(f"   Found {len(contents)} items in bucket")
                if contents:
                    print(f"   Sample items: {contents[:5]}")
                return True
            except Exception as e:
                print(f"❌ Mount appears successful but cannot list contents: {e}")
                return False
        else:
            print(f"❌ Failed to mount bucket")
            print(f"   Command: {cmd}")
            print(f"   Error: {result.stderr.strip() if result.stderr else 'No error details'}")
            return False
            
    except Exception as e:
        print(f"❌ Error during mount: {e}")
        return False


def unmount_geotools(mount_path: str = "/home/walter_littor_al/geotools_sites") -> bool:
    """
    Unmount the geotools bucket.
    
    Args:
        mount_path: Path where the bucket is mounted
        
    Returns:
        True if unmount successful, False otherwise
    """
    return _unmount_path(mount_path, "geotools")


def unmount_tide(mount_path: str = "/home/walter_littor_al/tide_model") -> bool:
    """
    Unmount the tide model bucket.
    
    Args:
        mount_path: Path where the bucket is mounted
        
    Returns:
        True if unmount successful, False otherwise
    """
    return _unmount_path(mount_path, "tide")


def unmount_all():
    """Unmount both geotools and tide buckets."""
    print("Unmounting all buckets...")
    
    geotools_success = unmount_geotools()
    tide_success = unmount_tide()
    
    if geotools_success and tide_success:
        print("✅ All buckets unmounted successfully")
    else:
        print("⚠️  Some unmount operations may have failed")


def _unmount_path(mount_path: str, bucket_name: str) -> bool:
    """
    Helper function to unmount a specific path.
    
    Args:
        mount_path: Path to unmount
        bucket_name: Name of bucket for logging
        
    Returns:
        True if unmount successful, False otherwise
    """
    print(f"Unmounting {bucket_name} bucket from {mount_path}")
    
    try:
        cmd = f"fusermount -u {mount_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully unmounted {mount_path}")
            return True
        else:
            # Return code 1 usually means "not mounted" which is fine
            if result.returncode == 1:
                print(f"✅ {mount_path} was not mounted")
                return True
            else:
                print(f"❌ Failed to unmount {mount_path}")
                print(f"   Error: {result.stderr.strip() if result.stderr else 'No error details'}")
                return False
                
    except Exception as e:
        print(f"❌ Error during unmount: {e}")
        return False


def status():
    """Check the status of both mount points."""
    print("Checking mount status...")
    
    geotools_path = "/home/walter_littor_al/geotools_sites"
    tide_path = "/home/walter_littor_al/tide_model"
    
    for path, name in [(geotools_path, "Geotools"), (tide_path, "Tide")]:
        try:
            if os.path.exists(path):
                contents = os.listdir(path)
                if contents:
                    print(f"✅ {name}: MOUNTED ({len(contents)} items)")
                else:
                    print(f"❌ {name}: NOT MOUNTED (empty directory)")
            else:
                print(f"❌ {name}: NOT MOUNTED (directory doesn't exist)")
        except Exception as e:
            print(f"❌ {name}: ERROR ({e})")


def main():
    """Command line interface for mount utilities."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Utility for mounting/unmounting cloud storage buckets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python mount_utils.py mount geotools     # Mount geotools bucket
    python mount_utils.py mount tide         # Mount tide bucket  
    python mount_utils.py mount all          # Mount both buckets
    python mount_utils.py unmount all        # Unmount all buckets
    python mount_utils.py status             # Check mount status
        """
    )
    
    parser.add_argument(
        'action', 
        choices=['mount', 'unmount', 'status'],
        help='Action to perform'
    )
    
    parser.add_argument(
        'target',
        nargs='?',
        choices=['geotools', 'tide', 'all'],
        default='all',
        help='Which bucket(s) to target (default: all)'
    )
    
    args = parser.parse_args()
    
    if args.action == 'status':
        status()
    elif args.action == 'mount':
        if args.target == 'geotools' or args.target == 'all':
            mount_geotools()
        if args.target == 'tide' or args.target == 'all':
            mount_tide()
    elif args.action == 'unmount':
        if args.target == 'geotools' or args.target == 'all':
            unmount_geotools()
        if args.target == 'tide' or args.target == 'all':
            unmount_tide()


if __name__ == "__main__":
    main()