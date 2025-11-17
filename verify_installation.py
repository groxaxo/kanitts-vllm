#!/usr/bin/env python3
"""
Verification script for KaniTTS-vLLM installation.
Run this script to verify all dependencies are correctly installed.
"""

import sys


def check_package(package_name, expected_version=None, import_name=None):
    """Check if a package is installed and optionally verify its version."""
    if import_name is None:
        import_name = package_name.replace("-", "_")
    
    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "unknown")
        
        if expected_version:
            if version == expected_version:
                print(f"✓ {package_name} {version}")
                return True
            else:
                print(f"✗ {package_name} {version} (expected {expected_version})")
                return False
        else:
            print(f"✓ {package_name} {version}")
            return True
            
    except ImportError as e:
        print(f"✗ {package_name}: Not installed ({e})")
        return False


def main():
    """Main verification function."""
    print("=" * 60)
    print("KaniTTS-vLLM Installation Verification")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # Check critical packages with specific versions
    print("Checking critical dependencies with version requirements:")
    print("-" * 60)
    all_ok &= check_package("transformers", "4.53.2")
    all_ok &= check_package("vllm", "0.10.0")
    all_ok &= check_package("nemo-toolkit", import_name="nemo")
    all_ok &= check_package("bitsandbytes", "0.46.0")
    
    print()
    print("Checking PyTorch ecosystem:")
    print("-" * 60)
    all_ok &= check_package("torch")
    all_ok &= check_package("numpy")
    all_ok &= check_package("scipy")
    all_ok &= check_package("ml_dtypes")
    
    # Check numpy version constraint
    try:
        import numpy
        major_version = int(numpy.__version__.split('.')[0])
        if major_version >= 2:
            print(f"✗ numpy version {numpy.__version__} is >= 2.0.0 (nemo-toolkit requires <2.0.0)")
            all_ok = False
    except Exception as e:
        print(f"✗ Could not verify numpy version: {e}")
        all_ok = False
    
    # Check ml_dtypes compatibility with onnx
    try:
        import ml_dtypes
        if not hasattr(ml_dtypes, 'float4_e2m1fn'):
            print(f"✗ ml_dtypes {ml_dtypes.__version__} is missing float4_e2m1fn (requires >=0.5.0)")
            all_ok = False
        else:
            print(f"✓ ml_dtypes {ml_dtypes.__version__} has required float4_e2m1fn support")
    except Exception as e:
        print(f"✗ Could not verify ml_dtypes compatibility: {e}")
        all_ok = False
    
    print()
    print("Checking web server dependencies:")
    print("-" * 60)
    all_ok &= check_package("fastapi")
    all_ok &= check_package("uvicorn")
    all_ok &= check_package("pydantic")
    
    print()
    print("Checking additional utilities:")
    print("-" * 60)
    all_ok &= check_package("langdetect")
    
    print()
    print("=" * 60)
    if all_ok:
        print("✅ All dependencies are correctly installed!")
        print()
        print("You can now start the server with:")
        print("  python server.py")
        return 0
    else:
        print("❌ Some dependencies are missing or have incorrect versions.")
        print()
        print("Please reinstall with:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
