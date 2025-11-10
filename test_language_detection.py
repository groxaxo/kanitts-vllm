"""Test script for language detection functionality"""

import sys
import os

# Add parent directory to path to allow direct import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Direct import to avoid loading heavy dependencies
from langdetect import detect, LangDetectException
from typing import Literal


class LanguageDetector:
    """Detects language of input text for routing to appropriate TTS model"""
    
    SUPPORTED_LANGUAGES = {"en", "es"}
    
    def __init__(self, default_language: str = "en"):
        """Initialize language detector
        
        Args:
            default_language: Language to use when detection fails (default: "en")
        """
        if default_language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"Default language must be one of {self.SUPPORTED_LANGUAGES}")
        self.default_language = default_language
    
    def detect_language(self, text: str) -> Literal["en", "es"]:
        """Detect language of input text
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code: "en" for English, "es" for Spanish
        """
        if not text or not text.strip():
            return self.default_language
        
        try:
            detected = detect(text)
            
            # Map detected language to supported languages
            if detected == "en":
                return "en"
            elif detected == "es":
                return "es"
            else:
                # For unsupported languages, use default
                print(f"[LanguageDetector] Detected unsupported language '{detected}', using default '{self.default_language}'")
                return self.default_language
                
        except LangDetectException as e:
            print(f"[LanguageDetector] Detection failed: {e}, using default '{self.default_language}'")
            return self.default_language


def test_language_detection():
    """Test language detection with various inputs"""
    detector = LanguageDetector(default_language="en")
    
    # Test cases
    test_cases = [
        # English texts
        ("Hello, how are you today?", "en"),
        ("This is a test of the text-to-speech system.", "en"),
        ("The quick brown fox jumps over the lazy dog.", "en"),
        
        # Spanish texts
        ("Hola, ¿cómo estás hoy?", "es"),
        ("Esta es una prueba del sistema de texto a voz.", "es"),
        ("Buenos días, me llamo María y vivo en España.", "es"),
        ("El rápido zorro marrón salta sobre el perro perezoso.", "es"),
        
        # Edge cases
        ("", "en"),  # Empty string should use default
        ("   ", "en"),  # Whitespace only should use default
        ("123456", "en"),  # Numbers only may use default
    ]
    
    print("Testing Language Detection")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for text, expected_lang in test_cases:
        detected_lang = detector.detect_language(text)
        status = "✓" if detected_lang == expected_lang else "✗"
        
        if detected_lang == expected_lang:
            passed += 1
        else:
            failed += 1
        
        # Truncate text for display
        display_text = text[:50] + "..." if len(text) > 50 else text
        print(f"{status} '{display_text}' -> {detected_lang} (expected: {expected_lang})")
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    
    if failed == 0:
        print("✅ All language detection tests passed!")
    else:
        print(f"⚠️  {failed} test(s) failed")
    
    return failed == 0


if __name__ == "__main__":
    success = test_language_detection()
    exit(0 if success else 1)
