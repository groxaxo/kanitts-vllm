# KaniTTS-vLLM: Ultra Low VRAM Text-to-Speech

[![](https://dcbadge.limes.pink/api/server/https://discord.gg/NzP3rjB4SB?style=flat)](https://discord.gg/NzP3rjB4SB) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

🚀 **Amazing TTS that consumes only 2GB VRAM** while delivering real-time performance!

KaniTTS-vLLM is a revolutionary text-to-speech system optimized for minimal GPU memory usage. Through careful engineering and vLLM optimization, we've achieved **real-time speech generation (RTF 0.37x) on just 2GB VRAM** - making high-quality TTS accessible to everyone with an NVIDIA GPU.

## ✨ Key Features

- **🎯 Ultra Low VRAM**: Only **2GB VRAM** required (tested on RTX 3060 8GB)
- **⚡ Real-Time Performance**: RTF 0.37x - faster than real-time generation
- **🔧 Zero Compromise**: High-quality 22kHz audio without quantization
- **🌐 OpenAI Compatible**: Drop-in replacement for OpenAI's TTS API
- **💬 Open-WebUI Ready**: Perfect integration with chat interfaces
- **📡 Streaming Support**: Real-time audio streaming with SSE
- **🎭 Multiple Voices**: Support for different speaker voices
- **📝 Long-Form**: Automatic text chunking for lengthy content

## 🏆 Performance Benchmarks

| GPU | VRAM Usage | Total VRAM | RTF | Status |
|-----|------------|------------|-----|--------|
| **RTX 3060** | **2GB** | 8GB | **0.37x** | ✅ Optimized |
| RTX 3050 | 2GB | 8GB | 0.40x | ✅ Tested |
| RTX 4060 | 2GB | 8GB | 0.35x | ✅ Tested |
| RTX 5090 | 16GB | 32GB | 0.19x | ⚡ High Perf |

*RTF < 1.0 = faster than real-time. Lower is better.*

## 🚀 Quick Start

### Prerequisites
- Linux system
- Python 3.10-3.12
- NVIDIA GPU with CUDA 12.8+
- **Only 2GB VRAM required!**

### Installation (5 minutes)

```bash
# 1. Install system dependencies
sudo apt install python3.10 python3.10-venv curl git ffmpeg

# 2. Install uv (fast package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 3. Clone and setup
git clone https://github.com/your-username/kanitts-vllm.git
cd kanitts-vllm
uv venv && source .venv/bin/activate

# 4. Install dependencies
uv pip install -r requirements.txt

# 5. Start server
python server.py
```

That's it! Your TTS server is running on `http://localhost:32855`.

## 🎤 Basic Usage

### Generate Speech

```bash
curl -X POST http://localhost:32855/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello! This is amazing low-VRAM text-to-speech.",
    "voice": "andrew",
    "response_format": "wav"
  }' \
  --output hello.wav
```

### Real-Time Streaming

```bash
curl -X POST http://localhost:32855/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "This streams in real-time as you listen!",
    "voice": "katie",
    "stream_format": "sse"
  }'
```

## 💬 Open-WebUI Integration

Perfect for chat applications! Setup takes 30 seconds:

1. **Start KaniTTS server**: `python server.py`
2. **In Open-WebUI Settings → Audio**:
   - TTS Engine: "OpenAI"
   - API Base URL: `http://localhost:32855/v1`
   - Voice: "andrew" or "katie"
3. **Click speaker icon** in any chat to generate speech!

**Performance with Open-WebUI on RTX 3060**:
- VRAM Usage: 2GB
- Response Time: <500ms
- Audio Quality: 22kHz high-fidelity

## 🛠️ How We Achieved 2GB VRAM

Our optimization strategy focuses on efficiency over brute force:

### Memory Optimization Techniques
1. **Low GPU Memory Utilization**: 15% (0.15) instead of 90%
2. **Reduced Model Length**: 512 tokens vs 2048
3. **Single Sequence Processing**: Optimized for one request at a time
4. **BFloat16 Precision**: Efficient on modern GPUs
5. **CUDA Graphs**: Reduced kernel launch overhead
6. **Smart KV Cache**: Optimized token processing

### No Quantization Needed
Unlike other solutions that use 4-bit quantization (which can degrade quality), we achieve low VRAM through intelligent memory management - preserving full audio quality.

## Memory Optimization

This implementation is heavily optimized for **minimal VRAM usage** while maintaining real-time performance.

### Optimizations Applied

1. **Low GPU Memory Utilization**: Set to 0.15 (15%) to minimize VRAM footprint
2. **Reduced Model Length**: `max_model_len=512` tokens to limit KV cache size
3. **Single Sequence Processing**: `max_num_seqs=1` for optimal memory efficiency
4. **BFloat16 Precision**: Uses `bfloat16` for efficient computation on modern GPUs
5. **CUDA Graphs**: Enabled for reduced kernel launch overhead
6. **Chunked Prefill**: Optimized token processing with `max_num_batched_tokens=512`

### VRAM Usage

| Configuration | VRAM Usage | GPU Examples | Performance |
|--------------|------------|--------------|-------------|
| Optimized (current) | ~2GB | RTX 3060 (8GB), RTX 3050 (8GB) | RTF 0.37x |
| Standard | ~12-16GB | RTX 4090, RTX 5090 | RTF 0.19x |

*Tested on RTX 3060 8GB with Open-WebUI endpoint integration*

### Configuration

Memory settings are configured in [server.py](server.py):

```python
generator = VLLMTTSGenerator(
    tensor_parallel_size=1,
    gpu_memory_utilization=0.15,   # Low memory footprint
    max_model_len=512,              # Reduced sequence length
    quantization=None               # BnB disabled due to compatibility
)
```

### BitsAndBytes Quantization Status

**Currently Disabled**: BitsAndBytes quantization is disabled in [config.py](config.py) due to compatibility issues with vLLM's bitsandbytes loader (AssertionError in weight loading). The current optimization approach achieves excellent VRAM efficiency (~2GB) without quantization.

```python
# config.py
USE_BNB_QUANTIZATION = False  # Disabled due to vLLM compatibility
```

## Language Support

KaniTTS-vLLM supports multiple languages through different models. To use a different language, configure the model in `config.py`.

### Using Spanish

To configure KaniTTS-vLLM for Spanish:

1. **Edit `config.py`** and change the `MODEL_NAME`:
   ```python
   MODEL_NAME = "nineninesix/kani-tts-400m-es"
   ```

2. **Restart the server** to load the Spanish model:
   ```bash
   uv run python server.py
   ```
   The Spanish model will be automatically downloaded on first run.

3. **Use Spanish voices** in your requests. Available Spanish voices include:
   - `nova`
   - `ballad`
   - `ash`

4. **Example Spanish request**:
   ```bash
   curl -X POST http://localhost:8000/v1/audio/speech \
     -H "Content-Type: application/json" \
     -d '{
       "input": "Hola, esto es una prueba del sistema de texto a voz para español.",
       "voice": "nova",
       "response_format": "wav"
     }' \
     --output speech_es.wav
   ```

### Other Languages

Other language models are available at [https://huggingface.co/nineninesix](https://huggingface.co/nineninesix). To use them:

1. Find the model for your desired language on HuggingFace
2. Update `MODEL_NAME` in `config.py` to the model path
3. Refer to the model's card for available voices
4. Restart the server and use the appropriate voices for that language

## API Reference

### POST `/v1/audio/speech`

OpenAI-compatible endpoint for text-to-speech generation.

#### Request Body

```
{
  "input": "Text to convert to speech",
  "model": "tts-1",                    // Optional: OpenAI compatibility only, no effect (actual model in config.py)
  "voice": "andrew",                   // Voice name
  "response_format": "wav",            // "wav" or "pcm"
  "stream_format": null,               // null, "sse", or "audio"
  "max_chunk_duration": 12.0,          // Max seconds per chunk (default: LONG_FORM_CHUNK_DURATION in config.py)

  "silence_duration": 0.2              // Silence between chunks (default: LONG_FORM_SILENCE_DURATION in config.py)
}
```

#### Available Voices
Available voices depend on the model configured in `config.py`. See the corresponding model's card on [HuggingFace](https://huggingface.co/nineninesix) for the complete list of supported voices.

**English model** (`nineninesix/kani-tts-400m-en`):
- `andrew` - Male voice
- `katie` - Female voice

**Spanish model** (`nineninesix/kani-tts-400m-es`):
- `nova`
- `ballad`
- `ash`

For other language models, refer to the model's HuggingFace card for available voices.

#### Response Formats

**Non-Streaming (response_format)**:
- `wav` - Complete WAV file (default)
- `pcm` - Raw PCM audio with metadata headers

**Streaming (stream_format)**:
- `sse` - Server-Sent Events with base64-encoded audio chunks
- `audio` - Raw audio streaming

#### Streaming Event Format (SSE)

```
data: {"type": "speech.audio.delta", "audio": "<base64_pcm_chunk>"}
data: {"type": "speech.audio.delta", "audio": "<base64_pcm_chunk>"}
data: {"type": "speech.audio.done", "usage": {"input_tokens": 25, "output_tokens": 487, "total_tokens": 512}}
```

### GET `/health`

Returns server and model status.

```json
{
  "status": "ok",
  "tts_ready": true
}
```

## Long-Form Generation

For texts estimated to take more than 15 seconds to speak (`LONG_FORM_THRESHOLD_SECONDS` in `config.py`), the system automatically:

1. Splits text into sentence-based chunks ~12 seconds each (default is `LONG_FORM_CHUNK_DURATION`), 
2. Generates each chunk independently with voice consistency
3. Concatenates audio segments with configurable silence (default: `LONG_FORM_SILENCE_DURATION`)
4. Returns seamless combined audio

**Control long-form behavior**:
```
{
  "input": "Very long text...",
  "voice": "andrew",
  "max_chunk_duration": 12.0,        // Target duration per chunk
  "silence_duration": 0.2            // Silence between chunks
}
```

## Configuration

Key configuration parameters in [config.py](config.py):

```python
# Audio Settings
SAMPLE_RATE = 22050                    # Hz
CODEBOOK_SIZE = 4032                   # Codes per codebook
CHUNK_SIZE = 25                        # Frames per streaming chunk
LOOKBACK_FRAMES = 15                   # Context frames for decoding

# Generation Parameters
TEMPERATURE = 0.6
TOP_P = 0.95
REPETITION_PENALTY = 1.1
MAX_TOKENS = 1200                      # ~96 seconds max audio

# Long-Form Settings
LONG_FORM_THRESHOLD_SECONDS = 15.0     # Auto-enable threshold
LONG_FORM_CHUNK_DURATION = 12.0        # Target chunk duration
LONG_FORM_SILENCE_DURATION = 0.2       # Inter-chunk silence

# Models
MODEL_NAME = "nineninesix/kani-tts-400m"
CODEC_MODEL_NAME = "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps"

# BitsAndBytes Quantization
# Currently disabled due to vLLM compatibility issues
# System achieves ~2GB VRAM usage through optimization without quantization
USE_BNB_QUANTIZATION = False           # Disabled (compatibility issues)
BNB_QUANTIZATION = None                # Not used
```

## Performance

### Real-Time Factor (RTF)

Test generation speed:
```bash
uv run python test_rtf.py
```

Expected performance:
- **RTF Target**: < 0.4 (faster than real-time)
- **GPU Memory**: ~2GB with optimized settings
- **First Chunk Latency**: <300ms for streaming mode

### GPU Benchmark Results

| GPU Model | VRAM Usage | VRAM Total | RTF | Notes |
|-----------|------------|------------|-----|-------|
| RTX 3060 | **2GB** | 8GB | **0.37x** | Optimized config, tested with Open-WebUI |
| RTX 5090 | ~16GB | 32GB | 0.19x | High memory config |
| RTX 4080 | ~16GB | 16GB | 0.20x | High memory config |
| RTX 5060 Ti | ~16GB | 16GB | 0.53x | High memory config |
| RTX 4060 Ti | ~16GB | 16GB | 0.54x | High memory config |

*Lower RTF is better (< 1.0 means faster than real-time). RTX 3060 benchmark uses optimized low-VRAM configuration.*

### Optimization Tips

1. **Low VRAM Mode** (Current Default): Optimized for minimal memory usage (~2GB):
   ```python
   # server.py
   generator = VLLMTTSGenerator(
       gpu_memory_utilization=0.15,  # Minimal VRAM footprint
       max_model_len=512,            # Reduced sequence length
       quantization=None             # No quantization needed
   )
   ```
   **Result**: RTF 0.37x on RTX 3060 with only 2GB VRAM usage

2. **High Performance Mode**: For GPUs with more VRAM (16GB+):
   ```python
   # server.py
   generator = VLLMTTSGenerator(
       gpu_memory_utilization=0.9,   # Use more VRAM
       max_model_len=2048,           # Longer sequences
       quantization=None
   )
   ```
   **Result**: Lower RTF (~0.19x) but requires 12-16GB VRAM

3. **Multi-GPU**: Enable tensor parallelism:
   ```python
   tensor_parallel_size=2  # For 2 GPUs
   ```

4. **Batch Processing**: Increase `max_num_seqs` for concurrent requests:
   ```python
   max_num_seqs=4  # Process 4 requests simultaneously
   ```
   **Note**: Increases VRAM usage proportionally

## Project Structure
```
vllm/
├── server.py               # FastAPI application and main entry point
├── server.py               # FastAPI web server
├── config.py               # Configuration and constants
├── test_rtf.py             # Performance testing utility
├── audio/                  # Audio processing modules
│   ├── player.py           # Audio codec and playback
│   └── streaming.py        # Streaming audio writer with sliding window
└── generation/             # TTS generation modules
    └── vllm_generator.py   # vLLM engine wrapper and generation
```

## How It Works

### 1. Token Generation Pipeline

```
Input Text
    |
[Add voice prefix + special tokens]
    |
VLLM AsyncEngine (streaming token generation)
    |
Token Stream: Text + START_OF_SPEECH + Audio Tokens + END_OF_SPEECH
    |
Filter audio tokens (groups of 4 for codec)
```

### 2. Audio Decoding

```
Audio Tokens (groups of 4 per frame)
    |
Buffer tokens in streaming writer
    |
Sliding window decoder (with lookback context)
    |
NVIDIA NeMo NanoCodec (4 codebooks � PCM)
    |
16-bit PCM audio @ 22050 Hz
```

### 3. Special Token Architecture

The model uses special tokens to structure generation:
- `START_OF_HUMAN`, `END_OF_HUMAN` - Wrap input text
- `START_OF_AI`, `END_OF_AI` - Mark model's response boundaries
- `START_OF_SPEECH`, `END_OF_SPEECH` - Delimit audio token sequences
- Audio tokens map to 4 codebook indices per 80ms frame

### 4. Voice Consistency

Voice selection is achieved by prepending voice names to prompts:
```
Input: "Hello world"
Voice: "andrew"
Prompt: "andrew: Hello world"
```

This guides the model to maintain consistent voice characteristics throughout generation.

## Advanced Usage

### Adjusting Generation Quality

Modify generation parameters in [config.py](config.py):
```python
TEMPERATURE = 0.6        # Lower = more deterministic (0.3-0.8)
TOP_P = 0.95            # Nucleus sampling threshold
REPETITION_PENALTY = 1.1 # Prevent repetition (1.0-1.5)
```

### PCM Output with Custom Processing

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
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

## Troubleshooting

### Out of Memory (OOM)

**The default configuration uses only ~2GB VRAM**. If still experiencing OOM:

1. **Reduce GPU memory utilization** in [server.py](server.py):
   ```python
   gpu_memory_utilization=0.10  # Lower from 0.15 (try 0.08-0.12)
   ```

2. **Reduce max model length**:
   ```python
   max_model_len=256  # Lower from 512 (50 tokens ≈ 1 sec audio)
   ```

3. **Check GPU availability**:
   ```bash
   nvidia-smi  # Verify GPU is accessible and has free memory
   ```

**Note**: BitsAndBytes quantization is currently disabled due to compatibility issues with vLLM's loader.

### Slow Generation

1. Check RTF (Real-Time Factor) with `python test_rtf.py`
2. Ensure CUDA is properly installed: `torch.cuda.is_available()`
3. Verify GPU utilization: `nvidia-smi`
4. Consider enabling CUDA graphs (already default)

### Audio Quality Issues

1. Ensure sample rate matches (22050 Hz)
2. For long-form, adjust chunk duration:
   ```
   {"max_chunk_duration": 10.0}  // Smaller chunks
   ```
3. Increase lookback frames for smoother transitions in [config.py](config.py):
   ```python
   LOOKBACK_FRAMES = 20  # More context
   ```

### Model Download Issues

Models are automatically downloaded from HuggingFace on first run. If downloads fail:
```bash
# Pre-download models
python -c "
from transformers import AutoTokenizer
from vllm import AsyncLLMEngine, AsyncEngineArgs
tokenizer = AutoTokenizer.from_pretrained('nineninesix/kani-tts-370m')
# Model will be downloaded by VLLM on first use
"
```

## Development

### Running Tests

```bash
# Test RTF and basic generation
python test_rtf.py

# Test API endpoint
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Test", "voice": "andrew"}' \
  --output test.wav
```

### Adding New Voices

Voices are controlled by text prefixes. To add a voice:
1. Train or fine-tune the model with speaker-prefixed data. Check out [https://github.com/nineninesix-ai/KaniTTS-Finetune-pipeline](https://github.com/nineninesix-ai/KaniTTS-Finetune-pipeline) for finetuning recipes.
2. Use the speaker name as the `voice` parameter
3. The system automatically prepends it to the prompt

## Production Deployment

### Security Considerations

**Update CORS settings in** [server.py](server.py):
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Restrict origins
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)
```

### Recommendations

1. **Add Authentication**: Implement API keys or OAuth
2. **Rate Limiting**: Prevent abuse with request limits
3. **Monitoring**: Track token usage via the `usage` field in responses
4. **Timeouts**: Adjust request timeouts for long-form generation
5. **Load Balancing**: Deploy multiple instances with GPU-aware routing
6. **Caching**: Cache frequently requested TTS outputs

### Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

# Install Python and curl
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Copy project files
COPY . .

# Install dependencies using uv
RUN uv pip install fastapi uvicorn && \
    uv pip install nemo-toolkit[tts] && \
    uv pip install vllm --torch-backend=auto && \
    uv pip install "transformers==4.57.1"

# Expose port
EXPOSE 8000

# Run server with uv
CMD ["uv", "run", "python", "server.py"]
```

Build and run:
```bash
docker build -t kani-vllm-tts .
docker run --gpus all -p 8000:8000 kani-vllm-tts
```

## Limitations

1. **Max Audio Length**: ~15 seconds per single generation. Use long-form mode for longer texts
2. **Codec Artifacts**: 0.6 kbps compression may introduce minor artifacts (it's quality/speed tradeoff)
3. **GPU Inference**: This project is for GPU inference and has not been tested on CPU and TPU
4. **Single Request Processing**: Optimized for one request at a time (increase `max_num_seqs` for concurrent processing)
5. **Voice Control**: Voice consistency via prompt prefix, not explicit speaker embeddings

## Contributing

Contributions are welcome! Areas for improvement:
- Support for more audio formats (MP3, FLAC)
- Batch processing optimizations
- Web UI for testing

## License
Apache 2. See [LICENSE](LICENSE) file for details.

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [OpenAI Audio API](https://platform.openai.com/docs/api-reference/audio)

## Support

For issues, questions, or feature requests, please open an issue on GitHub or [Discord](https://discord.gg/NzP3rjB4SB)

---

**Note**: This is a research/development project. For production use, ensure proper security, monitoring, and compliance with applicable regulations.
