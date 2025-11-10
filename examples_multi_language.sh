#!/bin/bash
# Multi-Language TTS Usage Examples
# Make sure the server is running: python server.py

echo "🌍 Multi-Language TTS Examples"
echo "================================"
echo ""

# Server URL
SERVER_URL="http://localhost:32855"

echo "1️⃣  Testing English with andrew voice..."
curl -X POST "$SERVER_URL/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello! This is an amazing text-to-speech system with automatic language detection.",
    "voice": "andrew",
    "response_format": "wav"
  }' \
  --output /tmp/english_andrew.wav --silent
echo "   ✓ Saved to /tmp/english_andrew.wav"
echo ""

echo "2️⃣  Testing English with katie voice..."
curl -X POST "$SERVER_URL/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Welcome to the multi-language text-to-speech system. I am Katie.",
    "voice": "katie",
    "response_format": "wav"
  }' \
  --output /tmp/english_katie.wav --silent
echo "   ✓ Saved to /tmp/english_katie.wav"
echo ""

echo "3️⃣  Testing Spanish with nova voice..."
curl -X POST "$SERVER_URL/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "¡Hola! Este es un sistema increíble de texto a voz con detección automática de idioma.",
    "voice": "nova",
    "response_format": "wav"
  }' \
  --output /tmp/spanish_nova.wav --silent
echo "   ✓ Saved to /tmp/spanish_nova.wav"
echo ""

echo "4️⃣  Testing Spanish with ballad voice..."
curl -X POST "$SERVER_URL/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Bienvenido al sistema de texto a voz multilingüe. Yo soy Ballad.",
    "voice": "ballad",
    "response_format": "wav"
  }' \
  --output /tmp/spanish_ballad.wav --silent
echo "   ✓ Saved to /tmp/spanish_ballad.wav"
echo ""

echo "5️⃣  Testing Spanish with ash voice..."
curl -X POST "$SERVER_URL/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Buenos días. Me llamo Ash y puedo hablar en español con claridad.",
    "voice": "ash",
    "response_format": "wav"
  }' \
  --output /tmp/spanish_ash.wav --silent
echo "   ✓ Saved to /tmp/spanish_ash.wav"
echo ""

echo "6️⃣  Testing automatic voice preference (English)..."
curl -X POST "$SERVER_URL/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "This will use the configured English voice preference."
  }' \
  --output /tmp/english_preference.wav --silent
echo "   ✓ Saved to /tmp/english_preference.wav"
echo ""

echo "7️⃣  Testing automatic voice preference (Spanish)..."
curl -X POST "$SERVER_URL/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Esto utilizará la voz preferida configurada para español."
  }' \
  --output /tmp/spanish_preference.wav --silent
echo "   ✓ Saved to /tmp/spanish_preference.wav"
echo ""

echo "8️⃣  Testing long-form English..."
curl -X POST "$SERVER_URL/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "The multi-language text-to-speech system is designed to automatically detect the language of your input text and route it to the appropriate model. This enables seamless bilingual support without requiring users to manually specify which language they are using. The system currently supports English and Spanish, with both models loaded in parallel for instant response times.",
    "voice": "andrew",
    "response_format": "wav"
  }' \
  --output /tmp/english_longform.wav --silent
echo "   ✓ Saved to /tmp/english_longform.wav"
echo ""

echo "9️⃣  Testing long-form Spanish..."
curl -X POST "$SERVER_URL/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "El sistema de texto a voz multilingüe está diseñado para detectar automáticamente el idioma de su texto de entrada y enrutarlo al modelo apropiado. Esto permite un soporte bilingüe sin problemas sin requerir que los usuarios especifiquen manualmente qué idioma están usando. El sistema actualmente admite inglés y español, con ambos modelos cargados en paralelo para tiempos de respuesta instantáneos.",
    "voice": "nova",
    "response_format": "wav"
  }' \
  --output /tmp/spanish_longform.wav --silent
echo "   ✓ Saved to /tmp/spanish_longform.wav"
echo ""

echo "🔟  Testing health endpoint..."
curl -X GET "$SERVER_URL/health" \
  -H "Content-Type: application/json" --silent | python -m json.tool
echo ""

echo "================================"
echo "✅ All examples completed!"
echo "   Audio files saved to /tmp/"
echo "   Play them with: ffplay /tmp/english_andrew.wav"
