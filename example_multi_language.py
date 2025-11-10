#!/usr/bin/env python3
"""
Multi-Language TTS Example Script

This script demonstrates how to use the multi-language TTS API
with automatic language detection.

Usage:
    python example_multi_language.py
"""

import requests
import json
from pathlib import Path

# Server configuration
SERVER_URL = "http://localhost:32855"
OUTPUT_DIR = Path("/tmp/tts_examples")

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)


def test_tts(text: str, voice: str = None, language: str = None, filename: str = None):
    """Test TTS with given parameters
    
    Args:
        text: Input text to synthesize
        voice: Voice name (optional, uses preference if not specified)
        language: Expected language (for display only)
        filename: Output filename
    """
    # Prepare request
    payload = {
        "input": text,
        "response_format": "wav"
    }
    
    if voice:
        payload["voice"] = voice
    
    # Display info
    lang_display = f"({language})" if language else ""
    voice_display = f"with voice '{voice}'" if voice else "with default preference"
    print(f"\n🎤 Testing {lang_display} {voice_display}")
    print(f"   Text: {text[:60]}{'...' if len(text) > 60 else ''}")
    
    # Make request
    try:
        response = requests.post(
            f"{SERVER_URL}/v1/audio/speech",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            # Save audio file
            output_path = OUTPUT_DIR / filename
            output_path.write_bytes(response.content)
            print(f"   ✅ Success! Saved to {output_path}")
        else:
            print(f"   ❌ Error: {response.status_code} - {response.text}")
    
    except Exception as e:
        print(f"   ❌ Exception: {e}")


def check_health():
    """Check server health and display status"""
    print("\n🏥 Checking server health...")
    try:
        response = requests.get(f"{SERVER_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Server is healthy")
            print(f"   Mode: {data.get('mode', 'unknown')}")
            
            if data.get('mode') == 'multi-language':
                langs = data.get('languages_initialized', [])
                print(f"   Languages: {', '.join(langs)}")
            
            return True
        else:
            print(f"   ❌ Server returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Cannot reach server: {e}")
        print(f"   Make sure server is running: python server.py")
        return False


def main():
    """Run all multi-language examples"""
    print("=" * 70)
    print("🌍 Multi-Language TTS Examples")
    print("=" * 70)
    
    # Check server health first
    if not check_health():
        print("\n❌ Server is not available. Exiting.")
        return
    
    print("\n" + "=" * 70)
    print("📝 ENGLISH EXAMPLES")
    print("=" * 70)
    
    # English examples with different voices
    test_tts(
        "Hello! This is an amazing text-to-speech system with automatic language detection.",
        voice="andrew",
        language="English",
        filename="english_andrew.wav"
    )
    
    test_tts(
        "Welcome to the multi-language text-to-speech system. I am Katie.",
        voice="katie",
        language="English",
        filename="english_katie.wav"
    )
    
    # English with voice preference
    test_tts(
        "This text will use your configured English voice preference.",
        language="English",
        filename="english_preference.wav"
    )
    
    # Long-form English
    test_tts(
        "The multi-language text-to-speech system is designed to automatically detect "
        "the language of your input text and route it to the appropriate model. "
        "This enables seamless bilingual support without requiring users to manually "
        "specify which language they are using. The system currently supports English "
        "and Spanish, with both models loaded in parallel for instant response times.",
        voice="andrew",
        language="English",
        filename="english_longform.wav"
    )
    
    print("\n" + "=" * 70)
    print("📝 SPANISH EXAMPLES")
    print("=" * 70)
    
    # Spanish examples with different voices
    test_tts(
        "¡Hola! Este es un sistema increíble de texto a voz con detección automática de idioma.",
        voice="nova",
        language="Spanish",
        filename="spanish_nova.wav"
    )
    
    test_tts(
        "Bienvenido al sistema de texto a voz multilingüe. Yo soy Ballad.",
        voice="ballad",
        language="Spanish",
        filename="spanish_ballad.wav"
    )
    
    test_tts(
        "Buenos días. Me llamo Ash y puedo hablar en español con claridad.",
        voice="ash",
        language="Spanish",
        filename="spanish_ash.wav"
    )
    
    # Spanish with voice preference
    test_tts(
        "Este texto utilizará tu voz preferida configurada para español.",
        language="Spanish",
        filename="spanish_preference.wav"
    )
    
    # Long-form Spanish
    test_tts(
        "El sistema de texto a voz multilingüe está diseñado para detectar automáticamente "
        "el idioma de su texto de entrada y enrutarlo al modelo apropiado. "
        "Esto permite un soporte bilingüe sin problemas sin requerir que los usuarios "
        "especifiquen manualmente qué idioma están usando. El sistema actualmente admite "
        "inglés y español, con ambos modelos cargados en paralelo para tiempos de respuesta instantáneos.",
        voice="nova",
        language="Spanish",
        filename="spanish_longform.wav"
    )
    
    print("\n" + "=" * 70)
    print("✅ All examples completed!")
    print("=" * 70)
    print(f"\n📁 Audio files saved to: {OUTPUT_DIR}")
    print(f"   Play with: ffplay {OUTPUT_DIR}/english_andrew.wav")
    print()


if __name__ == "__main__":
    main()
