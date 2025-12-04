# KaniTTS-vLLM: Ultra Low VRAM Text-to-Speech

[![](https://dcbadge.limes.pink/api/server/https://discord.gg/NzP3rjB4SB?style=flat)](https://discord.gg/NzP3rjB4SB) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

🚀 **Amazing TTS that consumes only 2GB VRAM** while delivering real-time performance!

KaniTTS-vLLM is a revolutionary text-to-speech system optimized for minimal GPU memory usage. Through careful engineering and vLLM optimization, we've achieved **real-time speech generation (RTF 0.37x) on just 2GB VRAM** - making high-quality TTS accessible to everyone with an NVIDIA GPU.

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **🎯 Ultra Low VRAM** | Only **2GB VRAM** per model - works on RTX 3060 and budget GPUs |
| **⚡ Real-Time Performance** | RTF 0.37x - generates speech faster than you can listen |
| **🌍 Multi-Language** | Automatic language detection for English and Spanish |
| **🌐 OpenAI Compatible** | Drop-in replacement for OpenAI's TTS API |
| **💬 Open-WebUI Ready** | Perfect integration with chat interfaces |
| **📡 Streaming** | Real-time audio streaming with SSE |
| **🎭 Multiple Voices** | Different speaker voices per language |
| **📝 Long-Form** | Automatic text chunking for lengthy content |

## 🏆 Performance Benchmarks

| Mode | VRAM | RTF | Best For |
|------|------|-----|----------|
| **Low VRAM** | **2GB** | **0.37x** | RTX 3060, limited memory |
| **Balanced** | 6GB | 0.25x | RTX 4060, good balance |
| **High Performance** | 16GB | 0.19x | RTX 5090, maximum speed |
| **4-bit Quantization** | 1GB | 0.45x | Very limited memory |

*RTF < 1.0 = faster than real-time. Lower is better.*

## 🚀 Quick Start

### Prerequisites

- Linux system
- **Python 3.10, 3.11, or 3.12** (Python 3.13+ not supported)
- NVIDIA GPU with CUDA 12.8+
- **Only 2GB VRAM required!**

### Installation

```bash
# Install system dependencies
sudo apt install python3.12 python3.12-venv curl git ffmpeg

# Install uv (fast package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Clone and setup
git clone https://github.com/groxaxo/kanitts-vllm.git
cd kanitts-vllm
uv venv --python 3.12 && source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Start server
python server.py
```

Server runs on `http://localhost:32855`

## 🎤 Usage Examples

### Generate Speech

```bash
curl -X POST http://localhost:32855/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello! This is amazing low-VRAM text-to-speech.", "voice": "andrew"}' \
  --output hello.wav
```

### Stream in Real-Time

```bash
curl -X POST http://localhost:32855/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "This streams in real-time!", "voice": "katie", "stream_format": "sse"}'
```

### Multi-Language (Auto-Detected)

```bash
# English - automatically routed to English model
curl -X POST http://localhost:32855/v1/audio/speech \
  -d '{"input": "Hello world!", "voice": "andrew"}' --output english.wav

# Spanish - automatically routed to Spanish model  
curl -X POST http://localhost:32855/v1/audio/speech \
  -d '{"input": "¡Hola mundo!", "voice": "nova"}' --output spanish.wav
```

## 🌍 Multi-Language Support

Multi-language mode is **enabled by default**, running both English and Spanish models with automatic language detection.

### Available Voices

| Language | Model | Voices |
|----------|-------|--------|
| **English** | `nineninesix/kani-tts-400m-en` | `andrew`, `katie` |
| **Spanish** | `nineninesix/kani-tts-400m-es` | `nova`, `ballad`, `ash` |

### Configuration

Edit `config.py` to customize:

```python
# Enable/disable multi-language mode
MULTI_LANGUAGE_MODE = True

# Select languages to load (affects VRAM usage)
ENABLED_LANGUAGES = ["en", "es"]  # Both: ~4GB VRAM
# ENABLED_LANGUAGES = ["en"]      # English only: ~2GB VRAM

# Voice preferences for each language
VOICE_PREFERENCES = {
    "en": "andrew",
    "es": "nova"
}
```

## ⚙️ Performance Modes

Choose your mode in `config.py`:

```python
PERFORMANCE_MODE = "low_vram"  # Options: low_vram, balanced, high_performance, bnb_4bit
```

| Mode | `gpu_memory_utilization` | `max_model_len` | Use Case |
|------|--------------------------|-----------------|----------|
| `low_vram` | 0.15 | 1536 | Default, minimal VRAM |
| `balanced` | 0.5 | 1536 | More headroom |
| `high_performance` | 0.9 | 2048 | Maximum speed |
| `bnb_4bit` | 0.3 | 1536 | 4-bit quantization |

## 💬 Open-WebUI Integration

1. Start KaniTTS: `python server.py`
2. In Open-WebUI **Settings → Audio**:
   - TTS Engine: `OpenAI`
   - API Base URL: `http://localhost:32855/v1`
   - Voice: `andrew` or `katie`
3. Click the speaker icon in any chat!

## 📚 API Reference

### POST `/v1/audio/speech`

OpenAI-compatible endpoint for text-to-speech generation.

```json
{
  "input": "Text to convert to speech",
  "voice": "andrew",
  "response_format": "wav",
  "stream_format": null,
  "max_chunk_duration": 12.0,
  "silence_duration": 0.2
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string | required | Text to synthesize |
| `voice` | string | `"andrew"` | Voice name (`andrew`, `katie`, `nova`, `ballad`, `ash`) |
| `response_format` | string | `"wav"` | `"wav"` or `"pcm"` |
| `stream_format` | string | `null` | `"sse"` for streaming, `null` for complete file |
| `max_chunk_duration` | float | `12.0` | Max duration per chunk for long-form |
| `silence_duration` | float | `0.2` | Silence between chunks |

### GET `/health`

Returns server status:

```json
{
  "status": "healthy",
  "mode": "multi-language",
  "languages_initialized": ["en", "es"],
  "tts_initialized": true
}
```

## 📝 Long-Form Generation

For texts >15 seconds, the system automatically:
1. Splits text into sentence-based chunks (~12 seconds each)
2. Generates each chunk with voice consistency
3. Concatenates with configurable silence
4. Returns seamless combined audio

## 🔧 Configuration Reference

Key parameters in [config.py](config.py):

```python
# Generation Parameters
TEMPERATURE = 0.6          # Lower = more deterministic (0.3-0.8)
TOP_P = 0.95              # Nucleus sampling threshold
MAX_TOKENS = 1200         # ~96 seconds max audio

# Long-Form Settings
LONG_FORM_THRESHOLD_SECONDS = 15.0  # Auto-enable threshold
LONG_FORM_CHUNK_DURATION = 12.0     # Target chunk duration
```

### PCM Output with Custom Processing

```bash
curl -X POST http://localhost:32855/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Raw audio for processing",
    "voice": "andrew",
    "response_format": "pcm"
  }' \
  --output speech.pcm

# Headers will include:
# X-Sample-Rate: 22050
# X-Channels: 1
# X-Bit-Depth: 16
```

## 🔧 Troubleshooting

### Out of Memory (OOM)

The default configuration uses only ~2GB VRAM. If still experiencing OOM:

1. Reduce `ENABLED_LANGUAGES` to one language in `config.py`
2. Lower `gpu_memory_utilization` to `0.10`
3. Check GPU availability: `nvidia-smi`

### Python Version Error

If you see dependency conflicts with Python 3.13+:

```bash
# Use Python 3.10-3.12 instead
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### ml_dtypes Error

```bash
pip install --upgrade 'ml_dtypes>=0.5.0'
```

## 🐳 Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy project files
COPY requirements.txt .
COPY . .

# Create virtual environment and install dependencies
RUN uv venv --python 3.12 && \
    . .venv/bin/activate && \
    uv pip install -r requirements.txt

# Expose port
EXPOSE 32855

# Run server
CMD [".venv/bin/python", "server.py"]
```

Build and run:

```bash
docker build -t kanitts-vllm .
docker run --gpus all -p 32855:32855 kanitts-vllm
```

## 📜 License

Apache 2.0. See [LICENSE](LICENSE) for details.

## 🔗 Links

- [vLLM Documentation](https://docs.vllm.ai/)
- [OpenAI Audio API](https://platform.openai.com/docs/api-reference/audio)
- [Discord Community](https://discord.gg/NzP3rjB4SB)
- [HuggingFace Models](https://huggingface.co/nineninesix)
