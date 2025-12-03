"""Multi-language TTS generator that manages multiple language models in parallel"""

from typing import Dict, Optional, Literal
from generation.vllm_generator import VLLMTTSGenerator
from generation.language_detection import LanguageDetector
from audio import LLMAudioPlayer


class MultiLanguageGenerator:
    """Manages multiple TTS generators for different languages with automatic routing"""
    
    def __init__(
        self,
        language_configs: Dict[str, Dict],
        voice_preferences: Dict[str, str],
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.15,
        max_model_len: int = 512,
        quantization: Optional[str] = None,
        dtype: str = "bfloat16"
    ):
        """Initialize multi-language generator
        
        Args:
            language_configs: Dictionary mapping language codes to model configurations
            voice_preferences: Dictionary mapping language codes to preferred voices
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization per model
            max_model_len: Maximum sequence length
            quantization: Quantization method (e.g., "bitsandbytes")
            dtype: Model precision (e.g., "bfloat16")
        """
        self.language_configs = language_configs
        self.voice_preferences = voice_preferences
        self.generators: Dict[str, VLLMTTSGenerator] = {}
        self.players: Dict[str, LLMAudioPlayer] = {}
        self.language_detector = LanguageDetector(default_language="en")
        
        # Store initialization parameters for lazy loading
        self.init_params = {
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "quantization": quantization,
            "dtype": dtype
        }
        
        print(f"[MultiLanguageGenerator] Initialized with languages: {list(language_configs.keys())}")
    
    async def initialize_language(self, language: str):
        """Initialize a specific language model
        
        Args:
            language: Language code (e.g., "en", "es")
        """
        if language in self.generators:
            print(f"[MultiLanguageGenerator] Language '{language}' already initialized")
            return
        
        if language not in self.language_configs:
            raise ValueError(f"Language '{language}' not configured")
        
        config = self.language_configs[language]
        model_name = config["model_name"]
        
        print(f"[MultiLanguageGenerator] Initializing {language} model: {model_name}")
        
        # Create generator for this language
        generator = VLLMTTSGenerator(
            model_name=model_name,
            **self.init_params
        )
        
        # Initialize the engine
        await generator.initialize_engine()
        
        # Create player for this language
        player = LLMAudioPlayer(generator.tokenizer)
        
        # Store generator and player
        self.generators[language] = generator
        self.players[language] = player
        
        print(f"[MultiLanguageGenerator] {language} model initialized successfully")
    
    async def initialize_all_languages(self):
        """Initialize all configured language models"""
        print(f"[MultiLanguageGenerator] Initializing all {len(self.language_configs)} language models...")
        
        for language in self.language_configs.keys():
            await self.initialize_language(language)
        
        print(f"[MultiLanguageGenerator] All language models initialized!")
    
    def get_generator(self, language: str) -> VLLMTTSGenerator:
        """Get generator for specific language
        
        Args:
            language: Language code
            
        Returns:
            VLLMTTSGenerator instance for the language
            
        Raises:
            ValueError: If language not initialized
        """
        if language not in self.generators:
            raise ValueError(f"Language '{language}' not initialized. Call initialize_language() first.")
        return self.generators[language]
    
    def get_player(self, language: str) -> LLMAudioPlayer:
        """Get audio player for specific language
        
        Args:
            language: Language code
            
        Returns:
            LLMAudioPlayer instance for the language
            
        Raises:
            ValueError: If language not initialized
        """
        if language not in self.players:
            raise ValueError(f"Language '{language}' not initialized. Call initialize_language() first.")
        return self.players[language]
    
    def detect_and_route(self, text: str) -> Literal["en", "es"]:
        """Detect language of text and return appropriate language code
        
        Args:
            text: Input text to analyze
            
        Returns:
            Language code for routing (must be an initialized language)
        """
        detected_language = self.language_detector.detect_language(text)
        
        # If detected language is not initialized, use the first available language
        if detected_language not in self.generators:
            available_languages = list(self.generators.keys())
            fallback_language = available_languages[0] if available_languages else "en"
            print(f"[MultiLanguageGenerator] Detected '{detected_language}' but not initialized, "
                  f"falling back to '{fallback_language}'")
            return fallback_language
        
        print(f"[MultiLanguageGenerator] Detected language: {detected_language}")
        return detected_language
    
    def get_voice_for_language(self, language: str, requested_voice: Optional[str] = None) -> str:
        """Get appropriate voice for language
        
        Args:
            language: Language code
            requested_voice: User-requested voice (optional)
            
        Returns:
            Voice name to use
        """
        config = self.language_configs.get(language)
        if not config:
            raise ValueError(f"Language '{language}' not configured")
        
        # If user requested a specific voice and it's available for this language, use it
        if requested_voice and requested_voice in config["available_voices"]:
            return requested_voice
        
        # Otherwise, use the voice preference for this language
        if language in self.voice_preferences:
            return self.voice_preferences[language]
        
        # Fallback to default voice for the language
        return config["default_voice"]
    
    def get_available_voices(self, language: str) -> list:
        """Get list of available voices for a language
        
        Args:
            language: Language code
            
        Returns:
            List of available voice names
        """
        config = self.language_configs.get(language)
        if not config:
            return []
        return config["available_voices"]
