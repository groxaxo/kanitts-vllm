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


# ============================================================================
# LANGUAGE CONFIGURATION - Choose Your Deployment Mode
# ============================================================================
#
# KaniTTS-vLLM supports flexible language deployment:
#
# 🌍 OPTION 1: Both Languages with Auto-Detection (Recommended for Open-WebUI)
#    - MULTI_LANGUAGE_MODE = True
#    - ENABLED_LANGUAGES = ["en", "es"]
#    - VRAM Required: ~4GB
#    - Features: Automatic language detection, all voices available
#
# 🇬🇧 OPTION 2: English Only
#    - MULTI_LANGUAGE_MODE = True
#    - ENABLED_LANGUAGES = ["en"]
#    - VRAM Required: ~2GB
#    - Voices: andrew, katie
#
# 🇪🇸 OPTION 3: Spanish Only
#    - MULTI_LANGUAGE_MODE = True
#    - ENABLED_LANGUAGES = ["es"]
#    - VRAM Required: ~2GB
#    - Voices: nova, ballad, ash
#
# 📦 OPTION 4: Legacy Single Language Mode
#    - MULTI_LANGUAGE_MODE = False
#    - Uses MODEL_NAME setting below
#    - VRAM Required: ~2GB
#
# ============================================================================

# Multi-Language Mode (recommended)
MULTI_LANGUAGE_MODE = True  # Set to False for legacy single language mode

# Select which languages to load when MULTI_LANGUAGE_MODE = True
# Change this setting to deploy English only, Spanish only, or both together
ENABLED_LANGUAGES = ["en", "es"]  # Options: ["en"], ["es"], or ["en", "es"]

# Legacy single language model path (only used when MULTI_LANGUAGE_MODE = False)
# Options:
# - English: "nineninesix/kani-tts-400m-en" (voices: andrew, katie)
# - Spanish: "nineninesix/kani-tts-400m-es" (voices: nova, ballad, ash)
# See https://huggingface.co/nineninesix for more language models
MODEL_NAME = "nineninesix/kani-tts-400m-en"
CODEC_MODEL_NAME = "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps"

# Language-specific model configurations
# These define the models and voices available for each language
LANGUAGE_MODELS = {
    "en": {
        "model_name": "nineninesix/kani-tts-400m-en",
        "default_voice": "andrew",
        "available_voices": ["andrew", "katie"]
    },
    "es": {
        "model_name": "nineninesix/kani-tts-400m-es",
        "default_voice": "nova",
        "available_voices": ["nova", "ballad", "ash"]
    }
}

# Voice Preferences
# When using multi-language mode with auto-detection, these are the default
# voices used for each language when the user doesn't specify a voice or uses
# "voice": "random" in their API request.
VOICE_PREFERENCES = {
    "en": "andrew",  # Default English voice (options: andrew, katie)
    "es": "nova"     # Default Spanish voice (options: nova, ballad, ash)
}

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
            "max_model_len": 1536,  # Must be >= MAX_TOKENS (1200) + prompt tokens (~300)
            "quantization": None,
            "precision": "bfloat16"
        },
        "balanced": {
            "gpu_memory_utilization": 0.5,
            "max_model_len": 1536,  # Must be >= MAX_TOKENS (1200) + prompt tokens (~300)
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
            "max_model_len": 1536,  # Must be >= MAX_TOKENS (1200) + prompt tokens (~300)
            "quantization": "bitsandbytes",
            "precision": "bfloat16"
        }
    }
    return configs.get(mode, configs["low_vram"])

# Get current performance configuration
PERFORMANCE_CONFIG = get_performance_config(PERFORMANCE_MODE)
