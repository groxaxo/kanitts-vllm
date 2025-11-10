"""Language detection utilities for multi-language TTS support"""

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
