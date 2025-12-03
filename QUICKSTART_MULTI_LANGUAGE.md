# Quick Start: Multi-Language TTS

Get started with multi-language TTS in 5 minutes!

## Prerequisites

- Linux system
- Python 3.10-3.12
- NVIDIA GPU with CUDA 12.8+
- **6GB+ VRAM recommended** (4GB minimum for low VRAM mode with dual languages)

## Installation

```bash
# 1. Install system dependencies
sudo apt install python3.10 python3.10-venv curl git ffmpeg

# 2. Install uv (fast package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 3. Clone repository
git clone https://github.com/groxaxo/kanitts-vllm.git
cd kanitts-vllm

# 4. Create virtual environment
uv venv && source .venv/bin/activate

# 5. Install dependencies
uv pip install -r requirements.txt
```

## Configuration (Optional)

Multi-language mode is **enabled by default**. To customize:

```bash
# Edit config.py
nano config.py
```

Key settings:
```python
# Enable/disable multi-language (default: True)
MULTI_LANGUAGE_MODE = True

# Select which languages to load (default: both)
# Use ["en"] for English only, ["es"] for Spanish only
ENABLED_LANGUAGES = ["en", "es"]

# Set voice preferences
VOICE_PREFERENCES = {
    "en": "andrew",  # English: andrew or katie
    "es": "nova"     # Spanish: nova, ballad, or ash
}

# Performance mode (default: low_vram)
PERFORMANCE_MODE = "low_vram"  # Options: low_vram, balanced, high_performance
```

## Start Server

```bash
python server.py
```

Wait for:
```
✅ Multi-language TTS initialized successfully with 2 languages!
```

Server runs on: `http://localhost:32855`

## Test It!

### Quick Test - English

```bash
curl -X POST http://localhost:32855/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello! This is an amazing multi-language TTS system.",
    "voice": "andrew"
  }' \
  --output test_en.wav

# Play it
ffplay test_en.wav
```

### Quick Test - Spanish

```bash
curl -X POST http://localhost:32855/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "¡Hola! Este es un sistema increíble de texto a voz.",
    "voice": "nova"
  }' \
  --output test_es.wav

# Play it
ffplay test_es.wav
```

### Run All Examples

Test all voices and features:

```bash
# Bash version
./examples_multi_language.sh

# OR Python version
python example_multi_language.py
```

## How It Works

1. **Send text** → System automatically detects language (English or Spanish)
2. **Routes to correct model** → Uses English or Spanish TTS model
3. **Generates audio** → Returns high-quality audio in requested voice
4. **Instant switching** → Both models loaded and ready

## Voice Options

### English Voices
- `andrew` - Male voice (default)
- `katie` - Female voice

### Spanish Voices
- `nova` - Voice 1 (default)
- `ballad` - Voice 2
- `ash` - Voice 3

## Performance

| GPU | Mode | VRAM | Languages |
|-----|------|------|-----------|
| RTX 3060 | Low VRAM | 2GB | EN only or ES only |
| RTX 3060 | Low VRAM | 4GB | EN + ES |
| RTX 4060 | Balanced | 12GB | EN + ES |
| RTX 5090 | High Perf | 32GB | EN + ES |

**Note**: Single-language (`ENABLED_LANGUAGES = ["en"]` or `["es"]`) uses ~50% less VRAM.

## Common Issues

### "Out of Memory"

**Solution**: Reduce VRAM usage by enabling only one language:

```python
# config.py
PERFORMANCE_MODE = "low_vram"  # Use lowest VRAM mode
ENABLED_LANGUAGES = ["en"]     # Load only English (~2GB)

# OR disable multi-language mode entirely
MULTI_LANGUAGE_MODE = False
MODEL_NAME = "nineninesix/kani-tts-400m-en"
```

### "Server Not Starting"

**Check**:
1. GPU available: `nvidia-smi`
2. CUDA installed: `nvcc --version`
3. Dependencies installed: `pip list | grep -E "vllm|torch"`

### "Wrong Language Detected"

**For very short texts**, language detection may be unreliable. Use longer sentences or specify voice explicitly.

## Next Steps

- Read [MULTI_LANGUAGE_GUIDE.md](MULTI_LANGUAGE_GUIDE.md) for detailed documentation
- Check [README.md](README.md) for full feature documentation
- Join [Discord](https://discord.gg/NzP3rjB4SB) for support

## API Examples

### Python

```python
import requests

response = requests.post(
    "http://localhost:32855/v1/audio/speech",
    json={
        "input": "Your text here",
        "voice": "andrew",  # or katie, nova, ballad, ash
        "response_format": "wav"
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

### JavaScript

```javascript
fetch('http://localhost:32855/v1/audio/speech', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    input: 'Your text here',
    voice: 'andrew',
    response_format: 'wav'
  })
})
.then(res => res.blob())
.then(blob => {
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  audio.play();
});
```

## Integration with Open-WebUI

1. Start KaniTTS server: `python server.py`
2. In Open-WebUI Settings → Audio:
   - TTS Engine: "OpenAI"
   - API Base URL: `http://localhost:32855/v1`
   - Voice: Choose from andrew, katie, nova, ballad, ash
3. Click speaker icon to generate speech!

The system automatically detects if your chat message is in English or Spanish and uses the appropriate model.

---

**Happy TTSing! 🎤🌍**
