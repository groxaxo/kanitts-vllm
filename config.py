"""Configuration and constants for Kani TTS"""

# Tokenizer configuration
TOKENIZER_LENGTH = 64400

# Special tokens
START_OF_TEXT = 1
END_OF_TEXT = 2
START_OF_SPEECH = TOKENIZER_LENGTH + 1
END_OF_SPEECH = TOKENIZER_LENGTH + 2
START_OF_HUMAN = TOKENIZER_LENGTH + 3
END_OF_HUMAN = TOKENIZER_LENGTH + 4
START_OF_AI = TOKENIZER_LENGTH + 5
END_OF_AI = TOKENIZER_LENGTH + 6
PAD_TOKEN = TOKENIZER_LENGTH + 7
AUDIO_TOKENS_START = TOKENIZER_LENGTH + 10

# Audio configuration
CODEBOOK_SIZE = 4032
SAMPLE_RATE = 22050

# Streaming configuration
CHUNK_SIZE = 25  # Number of new frames to output per iteration
LOOKBACK_FRAMES = 15  # Number of frames to include from previous context

# Generation configuration
TEMPERATURE = 0.6
TOP_P = 0.95
REPETITION_PENALTY = 1.1
REPETITION_CONTEXT_SIZE = 20
MAX_TOKENS = 1200

# Long-form generation configuration
LONG_FORM_THRESHOLD_SECONDS = 15.0  # Auto-enable chunking for texts estimated >15s
LONG_FORM_CHUNK_DURATION = 12.0     # Target duration per chunk (stay within 5-15s training distribution)
LONG_FORM_SILENCE_DURATION = 0.2    # Silence between chunks in seconds


# Model paths
# Change MODEL_NAME to use a different language:
# - English: "nineninesix/kani-tts-400m-en" (voices: andrew, katie)
# - Spanish: "nineninesix/kani-tts-400m-es" (voices: nova, ballad, ash)
# See https://huggingface.co/nineninesix for more language models
MODEL_NAME = "nineninesix/kani-tts-400m-en"
CODEC_MODEL_NAME = "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps"

# Performance Mode Configuration
# Choose between different performance modes based on your hardware and needs
# Options: "low_vram", "balanced", "high_performance"
PERFORMANCE_MODE = "low_vram"  # Default mode for minimal VRAM usage

# BitsAndBytes Quantization Configuration
# Enable BnB quantization to reduce VRAM consumption by ~4x
# Options: None (disabled), "bitsandbytes" (4-bit quantization)
# NOTE: May have compatibility issues with some vLLM versions
USE_BNB_QUANTIZATION = False  # Set to True to enable 4-bit quantization
BNB_QUANTIZATION = "bitsandbytes" if USE_BNB_QUANTIZATION else None

# Precision Configuration
# Options: "bfloat16", "float16", "float32"
# bfloat16: Best for RTX 30xx+ GPUs, good performance
# float16: Compatible with more GPUs, slightly less efficient
# float32: Maximum quality, highest VRAM usage
PRECISION = "bfloat16"  # Default precision

# Performance Mode Presets
def get_performance_config(mode):
    """Get configuration based on performance mode"""
    configs = {
        "low_vram": {
            "gpu_memory_utilization": 0.15,
            "max_model_len": 512,
            "quantization": None,
            "precision": "bfloat16"
        },
        "balanced": {
            "gpu_memory_utilization": 0.5,
            "max_model_len": 1024,
            "quantization": None,
            "precision": "bfloat16"
        },
        "high_performance": {
            "gpu_memory_utilization": 0.9,
            "max_model_len": 2048,
            "quantization": None,
            "precision": "bfloat16"
        },
        "bnb_4bit": {
            "gpu_memory_utilization": 0.3,
            "max_model_len": 1024,
            "quantization": "bitsandbytes",
            "precision": "bfloat16"
        }
    }
    return configs.get(mode, configs["low_vram"])

# Get current performance configuration
PERFORMANCE_CONFIG = get_performance_config(PERFORMANCE_MODE)
