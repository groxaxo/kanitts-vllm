#!/usr/bin/env python3
"""
Test script to verify Python version requirements and dependency resolution.
This test ensures that the project correctly handles Python version constraints.
"""

import sys
import subprocess
import os


def test_setup_py_python_requires():
    """Test that setup.py has the correct python_requires field."""
    print("Testing setup.py python_requires field...")
    print("-" * 60)
    
    setup_py_path = os.path.join(os.path.dirname(__file__), 'setup.py')
    
    if not os.path.exists(setup_py_path):
        print("✗ setup.py not found")
        return False
    
    with open(setup_py_path, 'r') as f:
        content = f.read()
    
    # Check for correct python_requires
    if 'python_requires=">=3.10,<3.13"' in content:
        print("✓ setup.py has correct python_requires: >=3.10,<3.13")
    else:
        print("✗ setup.py missing or incorrect python_requires")
        return False
    
    # Check for version check code
    if 'sys.version_info >= (3, 13)' in content:
        print("✓ setup.py has Python 3.13 version check")
    else:
        print("✗ setup.py missing Python 3.13 version check")
        return False
    
    return True


def test_requirements_txt_ml_dtypes():
    """Test that requirements.txt has environment marker for ml_dtypes."""
    print("\nTesting requirements.txt ml_dtypes constraint...")
    print("-" * 60)
    
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    
    if not os.path.exists(requirements_path):
        print("✗ requirements.txt not found")
        return False
    
    with open(requirements_path, 'r') as f:
        content = f.read()
    
    # Check for ml_dtypes with environment marker
    if "ml_dtypes>=0.5.0; python_version<'3.13'" in content:
        print("✓ requirements.txt has ml_dtypes with python_version<'3.13' marker")
    elif "ml_dtypes>=0.5.0" in content:
        print("⚠ requirements.txt has ml_dtypes but without environment marker")
        print("  This may cause issues on Python 3.13+")
        return False
    else:
        print("✗ requirements.txt missing ml_dtypes requirement")
        return False
    
    # Check for numpy constraint
    if "numpy>=1.24.0,<2.0.0" in content:
        print("✓ requirements.txt has correct numpy constraint: >=1.24.0,<2.0.0")
    else:
        print("✗ requirements.txt missing or incorrect numpy constraint")
        return False
    
    # Check for Python version comment
    if "Python 3.13+ is NOT supported" in content or "PYTHON VERSION REQUIREMENT" in content:
        print("✓ requirements.txt has Python version warning comment")
    else:
        print("⚠ requirements.txt missing Python version warning")
    
    return True


def test_current_python_version():
    """Test the current Python version is supported."""
    print("\nTesting current Python version...")
    print("-" * 60)
    
    version = sys.version_info
    print(f"Current Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and 10 <= version.minor <= 12:
        print(f"✓ Python {version.major}.{version.minor} is supported")
        return True
    elif version.major == 3 and version.minor >= 13:
        print(f"✗ Python {version.major}.{version.minor} is NOT supported")
        print(f"  This project requires Python 3.10, 3.11, or 3.12")
        print(f"  ml-dtypes>=0.5.0 requires numpy>=2.1.0 on Python 3.13+")
        print(f"  But nemo-toolkit requires numpy<2.0.0")
        return False
    else:
        print(f"⚠ Python {version.major}.{version.minor} is not in the supported range (3.10-3.12)")
        return False


def test_documentation():
    """Test that documentation mentions Python version requirements."""
    print("\nTesting documentation for Python version requirements...")
    print("-" * 60)
    
    docs_to_check = {
        'README.md': ['Python 3.10, 3.11, or 3.12', 'Python 3.13'],
        'INSTALL.md': ['Python 3.13', 'numpy<2.0.0'],
    }
    
    all_ok = True
    for doc_name, keywords in docs_to_check.items():
        doc_path = os.path.join(os.path.dirname(__file__), doc_name)
        if not os.path.exists(doc_path):
            print(f"⚠ {doc_name} not found")
            continue
        
        with open(doc_path, 'r') as f:
            content = f.read()
        
        found_all = all(keyword in content for keyword in keywords)
        if found_all:
            print(f"✓ {doc_name} mentions Python version requirements")
        else:
            print(f"⚠ {doc_name} may be missing Python version information")
            missing = [k for k in keywords if k not in content]
            print(f"  Missing keywords: {missing}")
            all_ok = False
    
    return all_ok


def main():
    """Main test function."""
    print("=" * 60)
    print("Python Version Requirements Test")
    print("=" * 60)
    print()
    
    results = []
    
    # Run all tests
    results.append(("setup.py python_requires", test_setup_py_python_requires()))
    results.append(("requirements.txt constraints", test_requirements_txt_ml_dtypes()))
    results.append(("Current Python version", test_current_python_version()))
    results.append(("Documentation", test_documentation()))
    
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
        print("The project correctly enforces Python 3.10-3.12 requirements")
        print("and will prevent installation on Python 3.13+.")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
