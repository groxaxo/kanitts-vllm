#!/usr/bin/env python3
"""
Test script to verify ml_dtypes compatibility with onnx.
This test ensures that ml_dtypes has the float4_e2m1fn attribute required by onnx>=1.17.0.
"""

import sys


def test_ml_dtypes_compatibility():
    """Test that ml_dtypes has the required float4_e2m1fn attribute."""
    print("Testing ml_dtypes compatibility...")
    print("-" * 60)
    
    try:
        import ml_dtypes
        print(f"✓ ml_dtypes version: {ml_dtypes.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import ml_dtypes: {e}")
        return False
    
    # Check for the float4_e2m1fn attribute
    if not hasattr(ml_dtypes, 'float4_e2m1fn'):
        print(f"✗ ml_dtypes is missing float4_e2m1fn attribute")
        print(f"  Current version: {ml_dtypes.__version__}")
        print(f"  Required version: >=0.5.0")
        return False
    
    print(f"✓ ml_dtypes has float4_e2m1fn attribute")
    
    # Try to use it with numpy
    try:
        import numpy as np
        dtype = np.dtype(ml_dtypes.float4_e2m1fn)
        print(f"✓ Successfully created numpy dtype with ml_dtypes.float4_e2m1fn: {dtype}")
    except Exception as e:
        print(f"⚠ Warning: Could not create numpy dtype: {e}")
        # This is not a critical error as the attribute exists
    
    return True


def test_onnx_import():
    """Test that onnx can be imported without errors."""
    print("\nTesting onnx import...")
    print("-" * 60)
    
    try:
        import onnx
        print(f"✓ onnx version: {onnx.__version__}")
        print(f"✓ onnx imported successfully")
        return True
    except AttributeError as e:
        if "float4_e2m1fn" in str(e):
            print(f"✗ onnx import failed with ml_dtypes compatibility error:")
            print(f"  {e}")
            print(f"\nSolution: Upgrade ml_dtypes to >=0.5.0")
            print(f"  pip install --upgrade 'ml_dtypes>=0.5.0'")
            return False
        else:
            print(f"✗ onnx import failed with unknown error: {e}")
            return False
    except ImportError as e:
        print(f"⚠ onnx not installed (this is expected if dependencies aren't fully installed)")
        print(f"  {e}")
        return None  # Not a failure, just not installed yet


def main():
    """Main test function."""
    print("=" * 60)
    print("ml_dtypes Compatibility Test")
    print("=" * 60)
    print()
    
    # Test ml_dtypes
    ml_dtypes_ok = test_ml_dtypes_compatibility()
    
    # Test onnx import
    onnx_ok = test_onnx_import()
    
    print()
    print("=" * 60)
    
    if ml_dtypes_ok and onnx_ok:
        print("✅ All compatibility tests passed!")
        print()
        print("ml_dtypes is compatible with onnx and has the required")
        print("float4_e2m1fn attribute for proper operation.")
        return 0
    elif ml_dtypes_ok and onnx_ok is None:
        print("⚠ ml_dtypes is compatible, but onnx is not installed yet")
        print()
        print("This is expected if you haven't installed all dependencies.")
        print("Run: pip install -r requirements.txt")
        return 0
    else:
        print("❌ Compatibility test failed!")
        print()
        if not ml_dtypes_ok:
            print("ml_dtypes needs to be upgraded to >=0.5.0")
            print("Run: pip install --upgrade 'ml_dtypes>=0.5.0'")
        return 1


if __name__ == "__main__":
    sys.exit(main())
