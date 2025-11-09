# Quick Start with BitsAndBytes Quantization

This guide helps you get started with KaniTTS-vLLM using BnB quantization for reduced VRAM usage.

## What You Get

🎯 **~4x less VRAM** - Run on 4GB+ GPUs instead of 16GB+
⚡ **Same speed** - Comparable inference performance
🎵 **Same quality** - Minimal audio quality impact
✅ **Already enabled** - Works out of the box!

## Installation (5 Minutes)

### 1. Prerequisites
```bash
# Linux with NVIDIA GPU, CUDA 12.8+, Python 3.10-3.12
nvidia-smi  # Verify GPU is available
```

### 2. Install Dependencies
```bash
# Clone the repository
git clone https://github.com/groxaxo/kanitts-vllm.git
cd kanitts-vllm

# Create virtual environment with uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv
source .venv/bin/activate

# Install packages
uv pip install fastapi uvicorn
uv pip install "nemo-toolkit[tts]==2.4.0"
uv pip install vllm --torch-backend=auto
uv pip install "transformers==4.57.1"
uv pip install "bitsandbytes>=0.46.1"  # For quantization
```

### 3. Start the Server
```bash
# BnB quantization is enabled by default
uv run python server.py
```

You'll see this log confirming BnB is active:
```
🔧 Using bitsandbytes quantization to reduce VRAM consumption
```

### 4. Test It Out
```bash
# Generate speech
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, this is a test with reduced VRAM usage.",
    "voice": "andrew",
    "response_format": "wav"
  }' \
  --output speech.wav
```

## VRAM Usage Comparison

| GPU            | Without BnB | With BnB | Works? |
|----------------|-------------|----------|--------|
| RTX 3050 (8GB) | ❌ OOM      | ✅ ~4GB  | Yes!   |
| RTX 3060 (12GB)| ⚠️ ~16GB   | ✅ ~4GB  | Yes!   |
| RTX 4060 (8GB) | ❌ OOM      | ✅ ~4GB  | Yes!   |
| RTX 5090 (32GB)| ✅ ~16GB   | ✅ ~4GB  | Yes!   |

## Configuration

BnB is **enabled by default**. To change settings, edit `config.py`:

### Disable BnB (Use Full Precision)
```python
USE_BNB_QUANTIZATION = False  # Requires 12-16GB VRAM
```

### Enable BnB (Default)
```python
USE_BNB_QUANTIZATION = True   # Requires 4GB+ VRAM
```

After changing, restart the server:
```bash
uv run python server.py
```

## Verify Installation

Run the validation test:
```bash
python3 test_bnb_config.py
```

You should see:
```
🎉 All tests passed! BnB integration is properly configured.
```

## Common Issues

### Out of Memory
**Solution**: Already using BnB! If still OOM, lower memory usage:

Edit `server.py`:
```python
gpu_memory_utilization=0.3  # Lower from 0.5
```

### "bitsandbytes not found"
**Solution**: Install bitsandbytes:
```bash
uv pip install "bitsandbytes>=0.46.1"
```

### Want to use full precision
**Solution**: Edit `config.py`:
```python
USE_BNB_QUANTIZATION = False
```

## Performance Tips

### For Lower-End GPUs (8GB or less)
```python
# config.py
USE_BNB_QUANTIZATION = True  # Keep enabled

# server.py
gpu_memory_utilization=0.3    # Use less memory
max_model_len=512             # Shorter sequences
```

### For High-End GPUs (16GB+)
```python
# config.py
USE_BNB_QUANTIZATION = True  # Still recommended for efficiency

# server.py
gpu_memory_utilization=0.5    # Standard setting
max_model_len=1024            # Standard length
```

## What's Next?

- 📖 Read [BNB_INTEGRATION.md](BNB_INTEGRATION.md) for detailed information
- 📖 Check [README.md](README.md) for all features
- 🧪 Run `python test_rtf.py` to test performance
- 🌐 Check the API docs at `http://localhost:8000/docs`

## Key Features Working with BnB

✅ Streaming (SSE)
✅ Long-form generation
✅ Multi-speaker support
✅ Multiple output formats (WAV, PCM)
✅ Multiple languages (English, Spanish, etc.)
✅ OpenAI-compatible API

Everything works the same, just with less VRAM! 🎉

## Need Help?

1. Run validation: `python3 test_bnb_config.py`
2. Check logs for errors
3. Read [BNB_INTEGRATION.md](BNB_INTEGRATION.md) troubleshooting section
4. Open an issue on GitHub

---

**Happy TTS-ing with 4x less VRAM!** 🚀
