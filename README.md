```
 ╦╔═┌─┐┌┐┌┬╔╦╗╔╦╗╔═╗   ┬  ┬╦  ╦  ╔╦╗
 ╠╩╗├─┤││││ ║  ║ ╚═╗───└┐┌┘║  ║  ║║║
 ╩ ╩┴ ┴┘└┘┴ ╩  ╩ ╚═╝    └┘ ╩═╝╩═╝╩ ╩
 Ultra Low VRAM Text-to-Speech Engine
```

[![](https://dcbadge.limes.pink/api/server/https://discord.gg/NzP3rjB4SB?style=flat)](https://discord.gg/NzP3rjB4SB) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

🚀 **High-quality text-to-speech that runs on just 2GB VRAM** with real-time streaming performance!

**KaniTTS-vLLM** is an advanced text-to-speech engine powered by vLLM, delivering studio-quality voice synthesis with unprecedented efficiency. Optimized for consumer hardware, it achieves **real-time speech generation (RTF 0.37x) using only 2GB of VRAM** - making professional-grade TTS accessible to anyone with an NVIDIA GPU.

### What Makes KaniTTS-vLLM Special?

This system combines the power of neural TTS models with vLLM's efficient inference engine to deliver:
- **Minimal Resource Usage**: Run high-quality TTS on budget GPUs (RTX 3060 and up)
- **Real-Time Performance**: Generate speech faster than playback speed
- **Multi-Language Intelligence**: Automatic language detection routes text to the right model
- **Production Ready**: OpenAI-compatible API for seamless integration
- **Streaming Support**: Real-time audio delivery for responsive applications

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

- Linux system (Ubuntu 22.04+ recommended)
- **Python 3.10, 3.11, or 3.12** (Python 3.13+ not supported)
- NVIDIA GPU with CUDA 12.8+ support
- **Minimum 2GB VRAM** (4GB for dual language)

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
```

### Choose Your Language Setup

**Before starting the server**, edit `config.py` to select your language configuration:

**For English + Spanish with auto-detection (recommended for Open-WebUI):**
```python
MULTI_LANGUAGE_MODE = True
ENABLED_LANGUAGES = ["en", "es"]  # Requires ~4GB VRAM
```

**For English only:**
```python
MULTI_LANGUAGE_MODE = True
ENABLED_LANGUAGES = ["en"]  # Requires ~2GB VRAM
```

**For Spanish only:**
```python
MULTI_LANGUAGE_MODE = True
ENABLED_LANGUAGES = ["es"]  # Requires ~2GB VRAM
```

### Start the Server

```bash
# Start TTS server on http://localhost:32855
python server.py
```

The server will automatically:
- Load the selected language model(s)
- Enable language auto-detection if both languages are configured
- Initialize the OpenAI-compatible API endpoint

## 🎤 Usage Examples

### Basic Speech Generation

```bash
# Generate English speech
curl -X POST http://localhost:32855/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello! This is amazing low-VRAM text-to-speech.", "voice": "andrew"}' \
  --output hello.wav

# Generate Spanish speech (auto-detected when both languages enabled)
curl -X POST http://localhost:32855/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "¡Hola! Este es un increíble sistema de texto a voz.", "voice": "nova"}' \
  --output hola.wav
```

### Real-Time Streaming

Stream audio as it's generated using Server-Sent Events (SSE):

```bash
curl -X POST http://localhost:32855/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "This streams in real-time!", "voice": "katie", "stream_format": "sse"}'
```

### Multi-Language Auto-Detection

When `ENABLED_LANGUAGES = ["en", "es"]` in config.py, the server automatically detects the language:

```bash
# English text → Automatically routed to English model
curl -X POST http://localhost:32855/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world! How are you doing today?", "voice": "andrew"}' \
  --output english.wav

# Spanish text → Automatically routed to Spanish model  
curl -X POST http://localhost:32855/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "¡Hola mundo! ¿Cómo estás hoy?", "voice": "nova"}' \
  --output spanish.wav

# Mixed content → Detects the dominant language
curl -X POST http://localhost:32855/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Let me say this in Spanish: ¡Hola amigos!", "voice": "andrew"}' \
  --output mixed.wav
```

**Note**: Language detection analyzes the entire input text. For best results with mixed-language content, make separate requests per language.

## 🌍 Language Configuration

KaniTTS-vLLM supports flexible language deployment - choose what works best for your use case:

### Quick Language Setup

Edit `config.py` before starting the server to configure your language setup:

```python
# Option 1: Both languages with auto-detection (default, ~4GB VRAM)
MULTI_LANGUAGE_MODE = True
ENABLED_LANGUAGES = ["en", "es"]

# Option 2: English only (~2GB VRAM)
MULTI_LANGUAGE_MODE = True
ENABLED_LANGUAGES = ["en"]

# Option 3: Spanish only (~2GB VRAM)
MULTI_LANGUAGE_MODE = True
ENABLED_LANGUAGES = ["es"]

# Option 4: Single language mode (legacy, ~2GB VRAM)
MULTI_LANGUAGE_MODE = False
MODEL_NAME = "nineninesix/kani-tts-400m-en"  # or "...-es" for Spanish
```

### How Language Auto-Detection Works

When both languages are enabled (`ENABLED_LANGUAGES = ["en", "es"]`):

1. **Automatic Detection**: The server analyzes each request to detect the language
2. **Smart Routing**: Text is automatically sent to the appropriate model (English or Spanish)
3. **Voice Selection**: Uses language-specific voices or falls back to defaults
4. **Seamless Experience**: Works transparently with OpenAI API and Open-WebUI

**Example with Auto-Detection:**
```bash
# English text → routed to English model automatically
curl -X POST http://localhost:32855/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world!", "voice": "andrew"}' --output hello.wav

# Spanish text → routed to Spanish model automatically  
curl -X POST http://localhost:32855/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "¡Hola mundo!", "voice": "nova"}' --output hola.wav
```

### Available Voices by Language

| Language | Model | Voices | VRAM Usage |
|----------|-------|--------|------------|
| **English** | `nineninesix/kani-tts-400m-en` | `andrew`, `katie` | ~2GB |
| **Spanish** | `nineninesix/kani-tts-400m-es` | `nova`, `ballad`, `ash` | ~2GB |
| **Both** | Both models loaded | All voices available | ~4GB |

### Customizing Voice Preferences

Set default voices for each language in `config.py`:

```python
VOICE_PREFERENCES = {
    "en": "andrew",  # Default English voice
    "es": "nova"     # Default Spanish voice
}
```

When a user doesn't specify a voice or uses `"voice": "random"`, these defaults are used for their respective languages.

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

KaniTTS-vLLM works seamlessly with Open-WebUI, providing real-time text-to-speech with automatic language detection.

### Setup Instructions

1. **Configure Language Support** (in `config.py`):
   ```python
   # Recommended: Enable both languages for auto-detection
   MULTI_LANGUAGE_MODE = True
   ENABLED_LANGUAGES = ["en", "es"]
   ```

2. **Start KaniTTS Server**:
   ```bash
   python server.py
   # Server runs on http://localhost:32855
   ```

3. **Configure Open-WebUI**:
   - Go to **Settings → Audio → Text-to-Speech**
   - Set **TTS Engine**: `OpenAI`
   - Set **API Base URL**: `http://localhost:32855/v1`
   - Set **Voice**: Choose from available voices:
     - English: `andrew`, `katie`
     - Spanish: `nova`, `ballad`, `ash`

4. **Use TTS in Chat**:
   - Type your message in English or Spanish
   - Click the speaker icon to hear the response
   - **Language auto-detection works automatically!**

### How Auto-Detection Works with Open-WebUI

When both languages are enabled:

1. **User types in English or Spanish** in the chat
2. **KaniTTS automatically detects** the language of each message
3. **Routes to the appropriate model** (English or Spanish)
4. **Streams audio in real-time** using the correct voice
5. **No manual language selection needed!**

**Example:**
- English message: "Hello, how are you?" → Uses English model + `andrew` voice
- Spanish message: "Hola, ¿cómo estás?" → Uses Spanish model + `nova` voice

### Streaming Performance

The streaming mode (`stream_format: "sse"`) is optimized for Open-WebUI:
- **Ultra-low latency**: Audio starts playing within milliseconds
- **Real-time delivery**: Server-Sent Events (SSE) for continuous streaming
- **Automatic chunking**: Long texts are split intelligently
- **Seamless transitions**: Smooth audio playback across chunks

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

Returns server status and language configuration:

```json
{
  "status": "healthy",
  "mode": "multi-language",
  "languages_initialized": ["en", "es"],
  "tts_initialized": true
}
```

**Response Fields:**
- `status`: Server health status (`"healthy"` when ready)
- `mode`: `"multi-language"` or `"single-language"`
- `languages_initialized`: List of loaded language models (e.g., `["en"]`, `["es"]`, or `["en", "es"]`)
- `tts_initialized`: Whether TTS engine is ready to process requests

**Example:**
```bash
curl http://localhost:32855/health
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

The default configuration uses only ~2GB VRAM per language model. If experiencing OOM:

1. **Reduce to single language** in `config.py`:
   ```python
   ENABLED_LANGUAGES = ["en"]  # or ["es"] - reduces VRAM to ~2GB
   ```

2. **Lower GPU memory utilization** in `config.py`:
   ```python
   PERFORMANCE_MODE = "low_vram"  # Uses only 15% GPU memory
   ```

3. **Check GPU availability**:
   ```bash
   nvidia-smi  # Verify VRAM usage and availability
   ```

### Language Detection Issues

If the wrong language model is being used:

1. **Check which languages are initialized**:
   ```bash
   curl http://localhost:32855/health
   # Returns: {"languages_initialized": ["en", "es"], ...}
   ```

2. **Ensure text is clearly in one language**: Mixed-language text may be detected as the dominant language

3. **Manually specify language** by using language-specific voices:
   - English voices: `andrew`, `katie`
   - Spanish voices: `nova`, `ballad`, `ash`

4. **Single language mode**: If you only need one language, configure it explicitly:
   ```python
   ENABLED_LANGUAGES = ["en"]  # Forces English only
   ```

### Voice Not Available

If you get an error about voice not being available:

1. **Check which languages are loaded** via `/health` endpoint
2. **Use language-appropriate voices**:
   - English models only support: `andrew`, `katie`
   - Spanish models only support: `nova`, `ballad`, `ash`
3. **Verify voice spelling** in your request (case-sensitive)

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

Deploy KaniTTS-vLLM with Docker for easy containerization.

### Create Dockerfile

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

### Build and Run

**Basic deployment (default: both languages):**
```bash
docker build -t kanitts-vllm .
docker run --gpus all -p 32855:32855 kanitts-vllm
```

**Custom language configuration:**

1. Edit `config.py` before building to set your preferred language(s):
   ```python
   ENABLED_LANGUAGES = ["en"]  # or ["es"] or ["en", "es"]
   ```

2. Build and run:
   ```bash
   docker build -t kanitts-vllm .
   docker run --gpus all -p 32855:32855 kanitts-vllm
   ```

**Or use environment-based configuration:**

Modify the Dockerfile to accept language configuration as environment variables, or mount a custom `config.py`:

```bash
docker run --gpus all -p 32855:32855 \
  -v $(pwd)/config.py:/app/config.py \
  kanitts-vllm
```

## 📜 License

Apache 2.0. See [LICENSE](LICENSE) for details.

## 🔗 Links

- [vLLM Documentation](https://docs.vllm.ai/)
- [OpenAI Audio API](https://platform.openai.com/docs/api-reference/audio)
- [Discord Community](https://discord.gg/NzP3rjB4SB)
- [HuggingFace Models](https://huggingface.co/nineninesix)
