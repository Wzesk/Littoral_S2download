#!/usr/bin/env python3
"""
Example usage of the Littoral Pipeline Test Framework

This script demonstrates how to run tests and analyze results programmatically.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_example_test():
    """Run an example test and analyze results."""
    
    # Ensure we're in the correct directory
    os.chdir('/home/walter_littor_al/Littoral_S2download')
    
    print("🧪 Littoral Pipeline Test Framework Example")
    print("=" * 50)
    
    # 1. List available configurations
    print("\n1. Available test configurations:")
    result = subprocess.run([
        'python', 'test_framework/test_runner.py', '--list-configs'
    ], capture_output=True, text=True)
    print(result.stdout)
    
    # 2. Run a quick smoke test
    print("\n2. Running quick smoke test...")
    result = subprocess.run([
        'conda', 'run', '-n', 'littoral_pipeline',
        'python', 'test_framework/test_runner.py', 
        '--config', 'test_framework/config/quick_smoke_test.yaml',
        '--log-level', 'INFO'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Test completed successfully!")
        print("Output:", result.stdout[-500:])  # Show last 500 characters
    else:
        print("❌ Test failed!")
        print("Error:", result.stderr[-500:])
        return False
    
    # 3. Check if results were generated
    results_dir = Path('test_framework/results')
    if results_dir.exists():
        result_files = list(results_dir.glob('*.csv'))
        print(f"\n3. Generated {len(result_files)} result files:")
        for f in result_files[-3:]:  # Show last 3 files
            print(f"   📄 {f.name}")
    
    # 4. Generate performance report (if results exist)
    if result_files:
        print("\n4. Generating performance report...")
        result = subprocess.run([
            'python', 'test_framework/visualize_results.py', 
            '--generate-report',
            '--results-dir', 'test_framework/results'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Performance report generated!")
            print("Check test_framework/results/ for HTML reports and plots")
        else:
            print("⚠️  Report generation failed (this is normal if dependencies are missing)")
    
    print("\n" + "=" * 50)
    print("✅ Example completed!")
    print("\nNext steps:")
    print("1. Check test_framework/results/ for detailed results")
    print("2. Modify test_framework/config/ files for custom tests")
    print("3. Run 'python test_framework/test_runner.py --help' for more options")
    
    return True

def verify_prerequisites():
    """Verify that prerequisites are met."""
    print("🔍 Verifying prerequisites...")
    
    # Check if we're in the right directory
    if not os.path.exists('test_framework'):
        print("❌ Please run this script from the Littoral_S2download directory")
        return False
    
    # Check conda environment
    result = subprocess.run(['conda', 'info', '--envs'], capture_output=True, text=True)
    if 'littoral_pipeline' not in result.stdout:
        print("❌ littoral_pipeline conda environment not found")
        return False
    
    # Check mounts
    result = subprocess.run([
        'python', 'pipeline/mount_utils.py', 'status'
    ], capture_output=True, text=True)
    
    if 'MOUNTED' not in result.stdout:
        print("⚠️  Cloud storage not mounted - attempting to mount...")
        mount_result = subprocess.run([
            'python', 'pipeline/mount_utils.py', 'mount', 'all'
        ], capture_output=True, text=True)
        
        if mount_result.returncode != 0:
            print("❌ Failed to mount cloud storage")
            print("Please run: python pipeline/mount_utils.py mount all")
            return False
    
    print("✅ Prerequisites verified!")
    return True

if __name__ == "__main__":
    if verify_prerequisites():
        run_example_test()
    else:
        print("❌ Prerequisites not met. Please fix the issues above and try again.")
        sys.exit(1)