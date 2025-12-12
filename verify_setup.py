import requests
import json
import time

# Configuration
ROUTER_URL = "http://localhost:8000/v1/audio/speech"
WHISPER_URL = "http://localhost:8887/v1/audio/transcriptions"
OUTPUT_DIR = "verification_output"

import os
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def test_voice(text, voice, language_code, filename):
    print(f"\n--- Testing Voice: {voice} ({language_code}) ---")
    print(f"Input Text: {text}")

    # 1. Generate Audio
    print("Generating audio...")
    start_time = time.time()
    try:
        response = requests.post(
            ROUTER_URL,
            json={
                "model": "tts-1",
                "input": text,
                "voice": voice,
                "response_format": "wav"
            },
            timeout=60
        )
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Audio Generation Failed: {e}")
        return False
    
    generation_time = time.time() - start_time
    audio_path = os.path.join(OUTPUT_DIR, filename)
    with open(audio_path, "wb") as f:
        f.write(response.content)
    print(f"✅ Audio generated in {generation_time:.2f}s: {audio_path}")

    # 2. Transcribe Audio (Verification)
    print("Transcribing with Whisper (port 8887)...")
    try:
        with open(audio_path, "rb") as audio_file:
            files = {
                "file": (filename, audio_file, "audio/wav"),
                "model": (None, "whisper-1"), # Standard OpenAI generic model name
            }
            # Note: standard OpenAI transcription endpoint is /v1/audio/transcriptions
            # We assume the server at 8887 follows this or similar.
            # If 8887 is a raw whisper server it might be different, but user said "check their endpoints... adapt server".
            # The prompt implies keeping this server compatible with standard OpenAI, 
            # and verify using whisper. Let's try standard OpenAI format first.
            
            response = requests.post(
                WHISPER_URL,
                files=files,
                timeout=60
            )
            
            # Print raw response for debugging if it fails
            if response.status_code != 200:
                print(f"Server Response: {response.text}")
            
            response.raise_for_status()
            result = response.json()
            transcription = result.get("text", "")
            
            print(f"📝 Transcription: {transcription}")
            
            # Simple validation
            # We don't do strict string matching because Whisper might hallucinate slightly or adds punctuation
            if len(transcription) < 5: 
                 print("⚠️ Transcription too short/empty. Possible gibberish or silence.")
                 return False
            
            print("✅ Verification cycle complete.")
            return True

    except Exception as e:
        print(f"❌ Transcription Failed: {e}")
        return False

if __name__ == "__main__":
    # Test English
    test_voice(
        "The quick brown fox jumps over the lazy dog.",
        "andrew",
        "en",
        "test_en.wav"
    )

    # Test Spanish
    # "Hola, esta es una prueba del sistema de texto a voz."
    test_voice(
        "Hola, esta es una prueba del sistema de texto a voz.",
        "nova",
        "es",
        "test_es.wav"
    )
