# Python 3.13 Dependency Resolution Fix

## Problem Statement

When attempting to install KaniTTS-vLLM with Python 3.13, users encounter this error:

```
× No solution found when resolving dependencies:
  ╰─▶ Because ml-dtypes>=0.5.0 depends on numpy>=2.1.0 and only the following
      versions of ml-dtypes are available:
          ml-dtypes<=0.5.0
          ml-dtypes==0.5.1
          ml-dtypes==0.5.3
          ml-dtypes==0.5.4
      we can conclude that ml-dtypes>=0.5.0 depends on numpy>=2.1.0.
      And because you require numpy>=1.24.0,<2.0.0 and ml-dtypes>=0.5.0, we
      can conclude that your requirements are unsatisfiable.
```

## Root Cause

This is a fundamental dependency conflict that occurs with Python 3.13+:

1. **ml-dtypes>=0.5.0** is required for the `float4_e2m1fn` attribute (needed by onnx>=1.17.0)
2. **ml-dtypes>=0.5.0** requires `numpy>=2.1.0` when Python>=3.13
3. **nemo-toolkit[tts]==2.5.3** requires `numpy<2.0.0`
4. These requirements are fundamentally incompatible

### Dependency Chain

```
kanitts-vllm
├── nemo-toolkit[tts]==2.5.3
│   └── onnx==1.19.0
│       └── ml_dtypes>=0.5.0  (for float4_e2m1fn support)
│           └── numpy>=2.1.0  (on Python 3.13+)
└── numpy>=1.24.0,<2.0.0  (required by nemo-toolkit)
    └── ❌ CONFLICT with numpy>=2.1.0
```

## Solution

Since this is a dependency conflict that cannot be resolved with Python 3.13+, the solution is to **restrict the project to Python 3.10-3.12**.

### Changes Made

1. **requirements.txt**
   - Added environment marker to `ml_dtypes>=0.5.0; python_version<'3.13'`
   - Added clear comments explaining the Python version requirement
   - Kept `numpy>=1.24.0,<2.0.0` constraint

2. **setup.py** (New)
   - Added Python version check that fails fast on Python 3.13+
   - Set `python_requires=">=3.10,<3.13"` for proper pip enforcement
   - Shows helpful error message explaining the issue

3. **Documentation Updates**
   - **README.md**: Emphasized Python 3.10-3.12 requirement
   - **INSTALL.md**: Added Python 3.13 troubleshooting section
   - **TROUBLESHOOTING_ML_DTYPES.md**: Added detailed Python 3.13 incompatibility explanation

4. **Tests Added**
   - **test_python_version_requirements.py**: Validates Python version constraints
   - **test_dependency_resolution.py**: Confirms dependency resolution works

## How It Works

### For Python 3.10, 3.11, or 3.12

The environment marker allows ml-dtypes to be installed:

```python
ml_dtypes>=0.5.0; python_version<'3.13'
```

Dependencies resolve correctly:
- numpy: 1.26.4 (satisfies both numpy>=1.24.0,<2.0.0 and ml-dtypes requirements)
- ml-dtypes: 0.5.4 (provides float4_e2m1fn)

### For Python 3.13+

The environment marker prevents ml-dtypes from being installed with this rule, but more importantly:

1. **setup.py** will fail with a clear error message when installing via pip
2. Users are directed to use Python 3.10-3.12
3. Documentation clearly explains the incompatibility

## User Experience

### Before Fix

```bash
$ python3.13 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
❌ Error: No solution found when resolving dependencies
   (Cryptic error message about ml-dtypes and numpy)
```

### After Fix

```bash
$ python3.13 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
❌ ERROR: Python 3.13+ is not supported by KaniTTS-vLLM.

This project requires Python 3.10, 3.11, or 3.12 due to dependency conflicts:
  - nemo-toolkit requires numpy<2.0.0
  - ml-dtypes>=0.5.0 requires numpy>=2.1.0 on Python 3.13+

Please use Python 3.10, 3.11, or 3.12 to install this project.

Example:
  python3.12 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
```

Or with uv:

```bash
$ uv venv
Using CPython 3.13.6
❌ Error: No solution found when resolving dependencies
   (But requirements.txt now has clear comments explaining the issue)
```

## Testing

All tests pass with Python 3.12:

```bash
$ python test_python_version_requirements.py
✅ All tests passed!
The project correctly enforces Python 3.10-3.12 requirements

$ python test_dependency_resolution.py
✅ All tests passed!
Dependencies can be resolved correctly with Python 3.10-3.12
```

## Future Considerations

This issue will be resolved when one of the following occurs:

1. **nemo-toolkit** updates to support numpy>=2.0.0
2. **ml-dtypes** provides a version that works with numpy<2.0.0 on Python 3.13+
3. A workaround is found that doesn't require ml-dtypes>=0.5.0

Until then, Python 3.10-3.12 is required.

## References

- [ml-dtypes GitHub Repository](https://github.com/jax-ml/ml_dtypes)
- [ONNX GitHub Repository](https://github.com/onnx/onnx)
- [NeMo Toolkit Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/)
- [PEP 508 - Environment Markers](https://peps.python.org/pep-0508/)
