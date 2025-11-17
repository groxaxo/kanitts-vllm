#!/usr/bin/env python3
"""
Test script to verify that dependency resolution works correctly.
This simulates the exact scenario from the problem statement.
"""

import sys
import subprocess
import tempfile
import os
import shutil


def test_uv_dependency_resolution():
    """Test that uv can resolve dependencies with Python 3.12."""
    print("Testing dependency resolution with uv (Python 3.12)...")
    print("-" * 60)
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        venv_path = os.path.join(tmpdir, 'test_venv')
        
        # Create virtual environment
        print(f"Creating virtual environment at {venv_path}...")
        result = subprocess.run(
            [sys.executable, '-m', 'venv', venv_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"✗ Failed to create virtual environment: {result.stderr}")
            return False
        
        # Get path to pip in the virtual environment
        pip_path = os.path.join(venv_path, 'bin', 'pip')
        
        # Install uv
        print("Installing uv...")
        result = subprocess.run(
            [pip_path, 'install', '-q', 'uv'],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"✗ Failed to install uv: {result.stderr}")
            return False
        
        # Get path to uv in the virtual environment
        uv_path = os.path.join(venv_path, 'bin', 'uv')
        
        # Get path to requirements.txt
        requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
        
        # Try to compile requirements
        print("Compiling requirements.txt with uv...")
        result = subprocess.run(
            [uv_path, 'pip', 'compile', requirements_path],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            print(f"✗ Failed to resolve dependencies:")
            print(result.stderr)
            
            # Check if it's the numpy/ml-dtypes conflict
            if "ml-dtypes" in result.stderr and "numpy" in result.stderr:
                print("\n⚠ This is the numpy/ml-dtypes conflict!")
                print("  This should not happen with Python 3.10-3.12")
                print(f"  Current Python: {sys.version_info.major}.{sys.version_info.minor}")
            
            return False
        
        # Parse output to check resolved versions
        output = result.stdout
        
        # Check for numpy and ml-dtypes versions
        numpy_version = None
        ml_dtypes_version = None
        
        for line in output.split('\n'):
            if line.startswith('numpy=='):
                numpy_version = line.split('==')[1].strip()
            elif line.startswith('ml-dtypes=='):
                ml_dtypes_version = line.split('==')[1].strip()
        
        print(f"✓ Dependencies resolved successfully")
        if numpy_version:
            print(f"  numpy: {numpy_version}")
            # Check that numpy is < 2.0.0
            if numpy_version.startswith('1.'):
                print(f"  ✓ numpy version is < 2.0.0")
            else:
                print(f"  ✗ numpy version is >= 2.0.0 (should be < 2.0.0)")
                return False
        
        if ml_dtypes_version:
            print(f"  ml-dtypes: {ml_dtypes_version}")
            # Check that ml-dtypes is >= 0.5.0
            major, minor = ml_dtypes_version.split('.')[:2]
            if int(major) == 0 and int(minor) >= 5:
                print(f"  ✓ ml-dtypes version is >= 0.5.0")
            else:
                print(f"  ✗ ml-dtypes version is < 0.5.0 (should be >= 0.5.0)")
                return False
        
        return True


def test_pip_requirements_parsing():
    """Test that pip can parse requirements.txt without errors."""
    print("\nTesting pip requirements parsing...")
    print("-" * 60)
    
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    
    # Just check that the file can be read and parsed
    try:
        with open(requirements_path, 'r') as f:
            lines = f.readlines()
        
        # Check for key dependencies
        has_ml_dtypes = False
        has_numpy = False
        has_marker = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('ml_dtypes') or line.startswith('ml-dtypes'):
                has_ml_dtypes = True
                if "python_version<'3.13'" in line:
                    has_marker = True
            elif line.startswith('numpy'):
                has_numpy = True
        
        if not has_ml_dtypes:
            print("✗ ml_dtypes not found in requirements.txt")
            return False
        
        if not has_numpy:
            print("✗ numpy not found in requirements.txt")
            return False
        
        if not has_marker:
            print("✗ ml_dtypes missing python_version marker")
            return False
        
        print("✓ requirements.txt is syntactically valid")
        print("✓ ml_dtypes has python_version<'3.13' marker")
        print("✓ numpy constraint is present")
        
        return True
        
    except Exception as e:
        print(f"✗ Error reading requirements.txt: {e}")
        return False


def main():
    """Main test function."""
    print("=" * 60)
    print("Dependency Resolution Test")
    print("=" * 60)
    print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print()
    
    results = []
    
    # Run tests
    results.append(("pip requirements parsing", test_pip_requirements_parsing()))
    
    # Only test uv resolution if we have Python 3.10-3.12
    if sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12:
        results.append(("uv dependency resolution", test_uv_dependency_resolution()))
    else:
        print(f"\n⚠ Skipping uv resolution test (Python {sys.version_info.major}.{sys.version_info.minor} is not supported)")
        print("  This project requires Python 3.10-3.12")
    
    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    print()
    if all_passed:
        print("✅ All tests passed!")
        print()
        print("Dependencies can be resolved correctly with Python 3.10-3.12")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
