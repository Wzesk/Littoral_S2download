#!/usr/bin/env python3
"""
Test runner for the Littoral Pipeline test suite.

This script runs all tests in the tests/ directory and provides
a summary of results.
"""

import sys
import os
import unittest
import argparse

# Add parent directory to path to import pipeline modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def discover_and_run_tests(test_dir='tests', pattern='test_*.py', verbosity=2):
    """
    Discover and run all tests in the specified directory.
    
    Args:
        test_dir: Directory containing test files
        pattern: Pattern to match test files
        verbosity: Test output verbosity level
    
    Returns:
        True if all tests pass, False otherwise
    """
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(script_dir, test_dir)
    
    # Discover tests
    loader = unittest.TestLoader()
    start_dir = test_path
    suite = loader.discover(start_dir, pattern=pattern)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nResult: {'✅ ALL TESTS PASSED' if success else '❌ SOME TESTS FAILED'}")
    
    return success


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="Run tests for the Littoral Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_tests.py                    # Run all tests
    python run_tests.py --pattern test_mounting.py  # Run specific test file
    python run_tests.py --verbose          # Run with maximum verbosity
    python run_tests.py --quiet            # Run with minimal output
        """
    )
    
    parser.add_argument(
        '--pattern',
        default='test_*.py',
        help='Pattern to match test files (default: test_*.py)'
    )
    
    parser.add_argument(
        '--test-dir',
        default='tests',
        help='Directory containing test files (default: tests)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_const',
        const=2,
        dest='verbosity',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_const',
        const=0,
        dest='verbosity',
        help='Quiet output'
    )
    
    parser.set_defaults(verbosity=1)
    
    args = parser.parse_args()
    
    print("Littoral Pipeline Test Runner")
    print("=" * 60)
    print(f"Test directory: {args.test_dir}")
    print(f"Test pattern: {args.pattern}")
    print(f"Verbosity: {args.verbosity}")
    print("=" * 60)
    
    success = discover_and_run_tests(
        test_dir=args.test_dir,
        pattern=args.pattern,
        verbosity=args.verbosity
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()