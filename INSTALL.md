# Installation Guide - KaniTTS-vLLM (Ultra Low VRAM Edition)

## Overview

KaniTTS-vLLM is optimized to run on **just 2GB VRAM** while delivering real-time text-to-speech performance. This guide walks you through the complete setup process.

## System Requirements

- **OS**: Linux (Ubuntu 20.04+, CentOS 8+, or similar)
- **Python**: 3.10 - 3.12
- **GPU**: NVIDIA GPU with CUDA 12.8+ support
- **VRAM**: Minimum 2GB (tested on RTX 3060 8GB)
- **RAM**: 8GB+ recommended
- **Storage**: 5GB free space for models

## Quick Install (Recommended)

### 1. Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip curl git ffmpeg

# CentOS/RHEL
sudo yum update
sudo yum install -y python3.10 python3-pip curl git ffmpeg
```

### 2. Install uv (Fast Python Package Manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version
```

### 3. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-username/kanitts-vllm.git
cd kanitts-vllm

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Test CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test vLLM installation
python -c "import vllm; print('vLLM installed successfully')"
```

### 5. Start the Server

```bash
python server.py
```

The server will start on `http://localhost:32855` and automatically download models on first run.

## Manual Installation (Alternative)

If you prefer not to use uv, you can install with pip:

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies in order
pip install fastapi uvicorn
pip install "nemo-toolkit[tts]==2.4.0"
pip install vllm --torch-backend=auto
pip install "transformers==4.57.1"
pip install torch numpy scipy pydantic
```

## Configuration for Low VRAM

The default configuration is optimized for minimal VRAM usage:

```python
# server.py (lines 69-74)
generator = VLLMTTSGenerator(
    tensor_parallel_size=1,        # Single GPU
    gpu_memory_utilization=0.15,   # Only 15% of GPU memory
    max_model_len=512,             # Limited sequence length
    quantization=None              # No quantization needed
)
```

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors, try reducing memory usage further:

```python
# In server.py, reduce these values:
gpu_memory_utilization=0.10  # Lower from 0.15
max_model_len=256           # Lower from 512
```

### CUDA Issues

1. **Check CUDA installation**:
   ```bash
   nvidia-smi
   nvcc --version
   ```

2. **Install PyTorch with CUDA support**:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Verify GPU access**:
   ```bash
   python -c "import torch; print(torch.cuda.get_device_name(0))"
   ```

### Model Download Issues

If model downloads fail:

1. **Check internet connection**
2. **Set HuggingFace cache directory**:
   ```bash
   export HF_HOME=~/.cache/huggingface
   ```
3. **Manual download** (if needed):
   ```bash
   python -c "
   from transformers import AutoTokenizer
   tokenizer = AutoTokenizer.from_pretrained('nineninesix/kani-tts-400m-en')
   print('Model downloaded successfully')
   "
   ```

### Permission Errors

If you encounter permission issues:

```bash
# Fix permissions for the project directory
sudo chown -R $USER:$USER /path/to/kanitts-vllm
chmod +x /path/to/kanitts-vllm/*.py
```

## Performance Verification

After installation, verify performance:

```bash
# Run RTF test
python test_rtf.py

# Expected output on RTX 3060:
# RTF: ~0.37x (faster than real-time)
# VRAM Usage: ~2GB
```

## Integration with Open-WebUI

1. Start KaniTTS server:
   ```bash
   python server.py
   ```

2. In Open-WebUI Settings → Audio:
   - TTS Engine: "OpenAI"
   - API Base URL: `http://localhost:32855/v1`
   - Voice: "andrew" or "katie"

3. Test by typing a message and clicking the speaker icon.

## Next Steps

- Read the main [README.md](README.md) for usage examples
- Check [API Reference](README.md#api-reference) for endpoint details
- Visit the [Discord community](https://discord.gg/NzP3rjB4SB) for support

## Support

For installation issues:
1. Check this guide first
2. Search existing [GitHub issues](https://github.com/your-username/kanitts-vllm/issues)
3. Join our [Discord](https://discord.gg/NzP3rjB4SB) for live help
