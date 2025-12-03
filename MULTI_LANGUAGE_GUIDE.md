# Multi-Language TTS Implementation Guide

This guide explains the new multi-language feature in KaniTTS-vLLM, which enables automatic language detection and parallel model execution for English and Spanish.

## Overview

The multi-language feature allows KaniTTS-vLLM to:
- **Automatically detect** whether input text is English or Spanish
- **Run both models in parallel** for instant language switching
- **Route requests automatically** to the appropriate model
- **Support voice preferences** for each language
- **Maintain backward compatibility** with single-language mode

## Architecture

### Components

1. **LanguageDetector** (`generation/language_detection.py`)
   - Uses the `langdetect` library to identify text language
   - Supports English and Spanish detection
   - Falls back to default language if detection fails

2. **MultiLanguageGenerator** (`generation/multi_language_generator.py`)
   - Manages multiple VLLMTTSGenerator instances (one per language)
   - Initializes both English and Spanish models in parallel
   - Routes text to appropriate model based on detected language
   - Handles voice selection and preferences

3. **Updated VLLMTTSGenerator** (`generation/vllm_generator.py`)
   - Now accepts `model_name` parameter for flexible model loading
   - Supports `dtype` parameter for precision configuration
   - Maintains backward compatibility with existing code

4. **Enhanced Server** (`server.py`)
   - Detects multi-language mode from config
   - Initializes appropriate generators based on mode
   - Routes API requests to correct language model
   - Handles voice selection per language

### Configuration

All configuration is in `config.py`:

```python
# Enable/disable multi-language mode
MULTI_LANGUAGE_MODE = True  # Default: True for multi-language support

# Select which languages to load in multi-language mode
# Options:
#   ["en", "es"] - Load both English and Spanish (default, ~4GB VRAM)
#   ["en"]       - Load only English (~2GB VRAM)
#   ["es"]       - Load only Spanish (~2GB VRAM)
ENABLED_LANGUAGES = ["en", "es"]

# Language-specific model configurations
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

# User voice preferences (configurable)
VOICE_PREFERENCES = {
    "en": "andrew",  # Your preferred English voice
    "es": "nova"     # Your preferred Spanish voice
}
```

## Request Flow

### Multi-Language Mode

```
1. Client sends request with text
          ↓
2. Server receives request at /v1/audio/speech
          ↓
3. MultiLanguageGenerator.detect_and_route(text)
          ↓
4. LanguageDetector.detect_language(text)
          ↓
5. Returns "en" or "es"
          ↓
6. Get appropriate generator and player
          ↓
7. Determine voice (requested or preference)
          ↓
8. Generate audio with selected model
          ↓
9. Return audio to client
```

### Single-Language Mode

```
1. Client sends request with text
          ↓
2. Server receives request at /v1/audio/speech
          ↓
3. Use single generator instance
          ↓
4. Generate audio with configured model
          ↓
5. Return audio to client
```

## API Usage

The API remains OpenAI-compatible and requires no changes for basic usage:

### Automatic Language Detection

```bash
# English text - automatically routed to English model
curl -X POST http://localhost:32855/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, this is a test."
  }'

# Spanish text - automatically routed to Spanish model
curl -X POST http://localhost:32855/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hola, esto es una prueba."
  }'
```

### Specifying Voices

```bash
# English with specific voice
curl -X POST http://localhost:32855/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, this is Katie speaking.",
    "voice": "katie"
  }'

# Spanish with specific voice
curl -X POST http://localhost:32855/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hola, yo soy Ballad.",
    "voice": "ballad"
  }'
```

### Voice Fallback Behavior

The system intelligently handles voice selection:

1. If you specify a voice that's valid for the detected language → uses that voice
2. If you specify a voice that's NOT valid for the detected language → uses language preference
3. If you don't specify a voice → uses language preference from config

Example:
```python
# English text with Spanish voice specified
# System detects English, sees "nova" isn't an English voice
# Falls back to English preference (andrew)
{
  "input": "Hello world",
  "voice": "nova"  # Spanish voice requested for English text
}
# Result: Uses "andrew" (English preference)
```

## VRAM Requirements

VRAM usage depends on which languages you enable:

| Languages Enabled | Configuration | VRAM Usage | Notes |
|------------------|--------------|------------|-------|
| One language (`["en"]` or `["es"]`) | Low VRAM | ~2GB | Single model loaded |
| Both languages (`["en", "es"]`) | Low VRAM | ~4GB | Both EN + ES models |
| One language | Balanced | ~6GB | Single model loaded |
| Both languages | Balanced | ~12GB | Both EN + ES models |
| One language | High Perf | ~16GB | Single model loaded |
| Both languages | High Perf | ~32GB | Both EN + ES models |

**Recommendation**: For GPUs with less than 6GB VRAM, enable only one language:
```python
# config.py
MULTI_LANGUAGE_MODE = True
ENABLED_LANGUAGES = ["en"]  # or ["es"] for Spanish only
```

Or use single-language mode:
```python
# config.py
MULTI_LANGUAGE_MODE = False
```

## Testing

### Unit Tests

Run language detection tests:
```bash
python test_language_detection.py
```

Expected output:
```
Testing Language Detection
============================================================
✓ 'Hello, how are you today?' -> en (expected: en)
✓ 'This is a test of the text-to-speech system.' -> en (expected: en)
...
✓ 'Buenos días, me llamo María y vivo en España.' -> es (expected: es)
============================================================
Results: 10 passed, 0 failed out of 10 tests
✅ All language detection tests passed!
```

### Integration Tests

Run example scripts (requires server running):

**Bash:**
```bash
./examples_multi_language.sh
```

**Python:**
```bash
python example_multi_language.py
```

These scripts test:
- All English voices (andrew, katie)
- All Spanish voices (nova, ballad, ash)
- Voice preferences
- Long-form generation
- Automatic language detection

## Troubleshooting

### Issue: "Language 'xx' not initialized"

**Cause**: The language model wasn't loaded during startup.

**Solution**: Check server logs during startup. Ensure both models are successfully loaded:
```
✅ Multi-language TTS initialized successfully with 2 languages!
```

### Issue: Wrong language detected

**Cause**: Short texts or ambiguous content may be misdetected.

**Solution**: 
1. Use more text for better detection accuracy
2. Or temporarily disable multi-language mode for single-language use
3. Or explicitly set voice to force a language (system will use preference)

### Issue: Out of memory with multi-language mode

**Cause**: Loading both models requires ~2x VRAM compared to single model.

**Solution**: Switch to single-language mode:
```python
# config.py
MULTI_LANGUAGE_MODE = False
MODEL_NAME = "nineninesix/kani-tts-400m-en"  # or -es for Spanish
```

### Issue: Voice not working for detected language

**Cause**: Voice specified doesn't exist for the detected language.

**Solution**: The system automatically falls back to the language preference. Check your voice names:
- English: andrew, katie
- Spanish: nova, ballad, ash

## Future Enhancements

Possible future improvements:
1. **More Languages**: Add French, German, etc.
2. **Mixed Language**: Support code-switching within a single request
3. **Custom Language Detection**: Allow users to specify language explicitly
4. **Dynamic Model Loading**: Load models on-demand instead of at startup
5. **Language-Specific Parameters**: Different temperature/top_p per language

## Migration Guide

### Upgrading from Single-Language

If you're upgrading from a previous single-language installation:

1. **Install new dependency**:
   ```bash
   pip install langdetect
   ```

2. **Update config.py** (optional):
   ```python
   # Keep your existing MODEL_NAME for single-language mode
   MULTI_LANGUAGE_MODE = False  # Disable multi-language
   ```
   
   OR
   
   ```python
   # Enable multi-language mode
   MULTI_LANGUAGE_MODE = True
   # Configure your voice preferences
   VOICE_PREFERENCES = {
       "en": "andrew",
       "es": "nova"
   }
   ```

3. **Restart server**:
   ```bash
   python server.py
   ```

### API Compatibility

✅ **Full backward compatibility**: Existing API calls work without modification.

## Security Notes

- ✅ CodeQL security scan: 0 vulnerabilities found
- ✅ All input validation preserved
- ✅ No new external network dependencies (langdetect is offline)
- ✅ Model files loaded from HuggingFace with standard security practices

## Support

For issues or questions:
1. Check [GitHub Issues](https://github.com/groxaxo/kanitts-vllm/issues)
2. Join [Discord](https://discord.gg/NzP3rjB4SB)
3. Review the main [README.md](README.md)
