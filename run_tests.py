#!/usr/bin/env python3
"""
Test runner for the Voice Assistant.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --quick      # Run quick tests (skip slow model tests)
    python run_tests.py --audio      # Run audio tests only
    python run_tests.py --stt        # Run STT tests only
"""
import unittest
import sys
import argparse
import os

# Suppress pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'


def run_tests(test_modules=None, verbosity=2):
    """Run specified test modules."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    if test_modules is None:
        # Discover all tests
        suite = loader.discover('tests', pattern='test_*.py')
    else:
        for module in test_modules:
            try:
                suite.addTests(loader.loadTestsFromName(module))
            except Exception as e:
                print(f"❌ Could not load {module}: {e}")
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description="Run Voice Assistant tests")
    parser.add_argument('--quick', action='store_true', 
                       help='Skip slow tests (model loading)')
    parser.add_argument('--audio', action='store_true',
                       help='Run audio tests only')
    parser.add_argument('--stt', action='store_true',
                       help='Run STT tests only')
    parser.add_argument('--llm', action='store_true',
                       help='Run LLM tests only')
    parser.add_argument('--integration', action='store_true',
                       help='Run integration tests only')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧪 Voice Assistant Test Suite")
    print("=" * 60)
    
    verbosity = 2 if args.verbose else 1
    
    # Determine which tests to run
    if args.audio:
        modules = ['tests.test_audio']
    elif args.stt:
        modules = ['tests.test_stt']
    elif args.llm:
        modules = ['tests.test_llm']
    elif args.integration:
        modules = ['tests.test_integration']
    elif args.quick:
        modules = ['tests.test_audio']  # Skip STT (slow model loading)
    else:
        modules = None  # Run all
    
    success = run_tests(modules, verbosity)
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    print("=" * 60)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

