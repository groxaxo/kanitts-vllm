# Troubleshooting: ml_dtypes Compatibility Issue

## Error Message

```
AttributeError: module 'ml_dtypes' has no attribute 'float4_e2m1fn'. Did you mean: 'float8_e4m3fn'?
```

## What Causes This Error?

This error occurs when you have an incompatible version of `ml_dtypes` installed. The error typically appears when:

1. Installing the project dependencies with Python 3.13 or newer
2. The dependency resolver installs `ml_dtypes==0.4.1` (or older)
3. The `onnx` package (version 1.17.0 or newer) tries to use `ml_dtypes.float4_e2m1fn`
4. This attribute doesn't exist in ml_dtypes versions older than 0.5.0

## Why Does This Happen?

The `onnx` package (used internally by `nemo-toolkit`) added support for the `float4_e2m1fn` data type in version 1.17.0. This type is provided by the `ml_dtypes` package, but only in version 0.5.0 and later.

### Dependency Chain
```
kanitts-vllm
  → nemo-toolkit[tts]==2.5.3
    → onnx==1.19.0
      → ml_dtypes>=0.5.0  (REQUIRED)
```

If the dependency resolver installs an older version of `ml_dtypes` (e.g., 0.4.1), the import chain breaks.

## Solution

### Quick Fix

If you've already installed dependencies and encounter this error:

```bash
# Upgrade ml_dtypes to a compatible version
pip install --upgrade 'ml_dtypes>=0.5.0'
```

### Fresh Installation

If you're installing from scratch, the fix is already included in `requirements.txt`:

```bash
# Clone the repository
git clone https://github.com/groxaxo/kanitts-vllm.git
cd kanitts-vllm

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (ml_dtypes>=0.5.0 is pinned in requirements.txt)
pip install -r requirements.txt
```

### Using uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/bin/env

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

## Verification

After installing or upgrading, verify the fix:

```bash
# Run the compatibility test
python test_ml_dtypes_compatibility.py

# Or use the full verification script
python verify_installation.py
```

Expected output:
```
✓ ml_dtypes version: 0.5.4 (or higher)
✓ ml_dtypes has float4_e2m1fn attribute
✓ onnx version: 1.19.0
✓ onnx imported successfully
```

## Technical Details

### The float4_e2m1fn Type

`float4_e2m1fn` is a 4-bit floating point format with:
- 1 sign bit
- 2 exponent bits
- 1 mantissa bit
- 1 fraction bit (fn = fraction normal)

This format is used in machine learning for extreme quantization scenarios.

### Version Requirements

| Package | Minimum Version | Reason |
|---------|----------------|--------|
| ml_dtypes | 0.5.0 | First version with float4_e2m1fn support |
| onnx | 1.17.0 | Started using float4_e2m1fn type |
| nemo-toolkit | 2.5.3 | Dependencies include onnx>=1.17.0 |

## Related Issues

- If you see errors about other missing attributes in ml_dtypes, the solution is the same
- Python 3.13 users may see this more frequently due to how newer Python versions resolve dependencies
- This issue affects any project using `nemo-toolkit[tts]>=2.5.0` with older ml_dtypes

## Prevention

To prevent this issue in future installations:

1. Always use the provided `requirements.txt` which pins ml_dtypes
2. Use a fresh virtual environment for each installation
3. Run `verify_installation.py` after installing to catch version issues early

## Additional Resources

- [ml_dtypes GitHub Repository](https://github.com/jax-ml/ml_dtypes)
- [ONNX GitHub Repository](https://github.com/onnx/onnx)
- [NeMo Toolkit Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/)

## Still Having Issues?

If upgrading ml_dtypes doesn't resolve the issue:

1. Check your installed versions:
   ```bash
   pip list | grep -E "(ml.dtypes|onnx|nemo.toolkit)"
   ```

2. Try a clean reinstall:
   ```bash
   pip uninstall ml_dtypes onnx nemo-toolkit -y
   pip install -r requirements.txt
   ```

3. Report the issue on GitHub with:
   - Your Python version (`python --version`)
   - Your OS and architecture
   - Output of `pip list`
   - Full error traceback
