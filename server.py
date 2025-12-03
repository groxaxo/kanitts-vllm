"""FastAPI server for Kani TTS with streaming support"""

import io
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field
from typing import Optional, Literal
import numpy as np
from scipy.io.wavfile import write as wav_write
import base64
import json

from audio import LLMAudioPlayer, StreamingAudioWriter
from generation.vllm_generator import VLLMTTSGenerator
from generation.multi_language_generator import MultiLanguageGenerator
from config import (CHUNK_SIZE, LOOKBACK_FRAMES, TEMPERATURE, TOP_P, MAX_TOKENS, 
                    LONG_FORM_THRESHOLD_SECONDS, LONG_FORM_SILENCE_DURATION, 
                    LONG_FORM_CHUNK_DURATION, PERFORMANCE_CONFIG, PERFORMANCE_MODE,
                    MULTI_LANGUAGE_MODE, LANGUAGE_MODELS, VOICE_PREFERENCES,
                    ENABLED_LANGUAGES)

from nemo.utils.nemo_logging import Logger

nemo_logger = Logger()
nemo_logger.remove_stream_handlers()


app = FastAPI(title="Kani TTS API", version="1.0.0")

# Add CORS middleware to allow client.html to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialized on startup)
generator = None
player = None
multi_language_generator = None  # Used when MULTI_LANGUAGE_MODE is enabled


class TTSRequest(BaseModel):
    text: str
    temperature: Optional[float] = TEMPERATURE
    max_tokens: Optional[int] = MAX_TOKENS
    top_p: Optional[float] = TOP_P
    chunk_size: Optional[int] = CHUNK_SIZE
    lookback_frames: Optional[int] = LOOKBACK_FRAMES


class OpenAISpeechRequest(BaseModel):
    """OpenAI-compatible speech request model"""
    input: str = Field(..., description="Text to convert to speech")
    model: Literal["tts-1", "tts-1-hd", "gpt-4o-mini-tts"] = Field(default="tts-1", description="TTS model to use")
    voice: str = Field(default="andrew", description="Voice to use (available voices depend on the model configured in config.py; use 'random' to omit voice prefix)")
    response_format: Literal["wav", "pcm"] = Field(default="wav", description="Audio format: wav or pcm")
    stream_format: Optional[Literal["sse", "audio"]] = Field(default=None, description="Use 'sse' for Server-Sent Events streaming")
    # Long-form generation parameters
    enable_long_form: Optional[bool] = Field(default=True, description="Auto-detect and use long-form generation for texts >15s")
    max_chunk_duration: Optional[float] = Field(default=12.0, description="Max duration per chunk in long-form mode (seconds)")
    silence_duration: Optional[float] = Field(default=0.2, description="Silence between chunks in long-form mode (seconds)")


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global generator, player, multi_language_generator
    
    print(f"🚀 Initializing VLLM TTS models in '{PERFORMANCE_MODE}' mode...")
    print(f"📊 Configuration: {PERFORMANCE_CONFIG}")

    if MULTI_LANGUAGE_MODE:
        # Filter LANGUAGE_MODELS to only include enabled languages
        enabled_language_models = {
            lang: config for lang, config in LANGUAGE_MODELS.items() 
            if lang in ENABLED_LANGUAGES
        }
        
        if not enabled_language_models:
            raise ValueError(f"No valid languages in ENABLED_LANGUAGES: {ENABLED_LANGUAGES}. "
                           f"Available languages: {list(LANGUAGE_MODELS.keys())}")
        
        # Initialize multi-language generator with selected language models
        languages_str = ", ".join(enabled_language_models.keys())
        print(f"🌍 Multi-language mode enabled - initializing {languages_str} model(s)...")
        print(f"📝 Voice preferences: {VOICE_PREFERENCES}")
        
        multi_language_generator = MultiLanguageGenerator(
            language_configs=enabled_language_models,
            voice_preferences=VOICE_PREFERENCES,
            tensor_parallel_size=1,
            gpu_memory_utilization=PERFORMANCE_CONFIG["gpu_memory_utilization"],
            max_model_len=PERFORMANCE_CONFIG["max_model_len"],
            quantization=PERFORMANCE_CONFIG["quantization"],
            dtype=PERFORMANCE_CONFIG["precision"]
        )
        
        # Initialize enabled language models
        await multi_language_generator.initialize_all_languages()
        
        print(f"✅ Multi-language TTS initialized successfully with {len(enabled_language_models)} language(s)!")
    else:
        # Single language mode (legacy behavior)
        print("🔤 Single language mode - initializing default model...")
        
        generator = VLLMTTSGenerator(
            tensor_parallel_size=1,
            gpu_memory_utilization=PERFORMANCE_CONFIG["gpu_memory_utilization"],
            max_model_len=PERFORMANCE_CONFIG["max_model_len"],
            quantization=PERFORMANCE_CONFIG["quantization"],
            dtype=PERFORMANCE_CONFIG["precision"]
        )

        # Initialize the async engine during startup to avoid lazy loading on first request
        await generator.initialize_engine()

        player = LLMAudioPlayer(generator.tokenizer)
        print(f"✅ VLLM TTS models initialized successfully in '{PERFORMANCE_MODE}' mode!")


@app.get("/health")
async def health_check():
    """Check if server is ready"""
    if MULTI_LANGUAGE_MODE:
        return {
            "status": "healthy",
            "mode": "multi-language",
            "languages_initialized": list(multi_language_generator.generators.keys()) if multi_language_generator else [],
            "tts_initialized": multi_language_generator is not None
        }
    else:
        return {
            "status": "healthy",
            "mode": "single-language",
            "tts_initialized": generator is not None and player is not None
        }


@app.post("/v1/audio/speech")
async def openai_speech(request: OpenAISpeechRequest):
    """OpenAI-compatible speech generation endpoint with multi-language support

    Supports both streaming (SSE) and non-streaming modes:
    - Without stream_format: Returns complete audio file (WAV or PCM)
    - With stream_format="sse": Returns Server-Sent Events with audio chunks
    
    In multi-language mode:
    - Automatically detects language of input text
    - Routes to appropriate English or Spanish model
    - Uses language-specific voice preferences
    """
    global generator, player  # Declare global variables
    
    # Check initialization based on mode
    if MULTI_LANGUAGE_MODE:
        if not multi_language_generator:
            raise HTTPException(status_code=503, detail="Multi-language TTS models not initialized")
        
        # Detect language and select appropriate generator/player
        detected_language = multi_language_generator.detect_and_route(request.input)
        generator = multi_language_generator.get_generator(detected_language)
        player = multi_language_generator.get_player(detected_language)
        
        # Determine voice to use for this language
        # If user didn't specify "random", check if their requested voice is valid for this language
        if request.voice != "random":
            voice_to_use = multi_language_generator.get_voice_for_language(detected_language, request.voice)
        else:
            voice_to_use = request.voice
        
        print(f"[Server] Language: {detected_language}, Voice: {voice_to_use}")
    else:
        # Single language mode
        if not generator or not player:
            raise HTTPException(status_code=503, detail="TTS models not initialized")
        voice_to_use = request.voice

    # Prepare prompt text with voice prefix (unless voice is "random")
    if voice_to_use == "random":
        prompt_text = request.input
    else:
        prompt_text = f"{voice_to_use}: {request.input}"

    # Streaming mode (SSE)
    if request.stream_format == "sse":
        async def sse_generator():
            """Generate Server-Sent Events with audio chunks"""
            import asyncio
            import queue as thread_queue
            from generation.chunking import estimate_duration, split_into_sentences

            chunk_queue = thread_queue.Queue()

            # Estimate duration to determine if we need long-form generation
            estimated_duration = estimate_duration(request.input)
            # In multi-language mode, use language-specific default voice
            if MULTI_LANGUAGE_MODE and voice_to_use == "random":
                voice_for_generation = multi_language_generator.voice_preferences.get(detected_language, "andrew")
            else:
                voice_for_generation = voice_to_use if voice_to_use != "random" else "andrew"
            use_long_form = estimated_duration > LONG_FORM_THRESHOLD_SECONDS

            # Track token counts for usage reporting
            input_token_count = 0
            output_token_count = 0

            if use_long_form:
                # Long-form streaming: stream each sentence chunk as it's generated
                print(f"[Server] Using long-form SSE streaming (estimated {estimated_duration:.1f}s)")

                async def generate_async_long_form():
                    nonlocal input_token_count, output_token_count
                    try:
                        # Split into chunks
                        chunks = split_into_sentences(request.input, max_duration_seconds=request.max_chunk_duration or LONG_FORM_CHUNK_DURATION)
                        total_chunks = len(chunks)

                        for i, text_chunk in enumerate(chunks):
                            # Custom list wrapper that pushes chunks to queue
                            class ChunkList(list):
                                def append(self, chunk):
                                    super().append(chunk)
                                    chunk_queue.put(("chunk", chunk))

                            audio_writer = StreamingAudioWriter(
                                player,
                                output_file=None,
                                chunk_size=CHUNK_SIZE,
                                lookback_frames=LOOKBACK_FRAMES
                            )
                            audio_writer.audio_chunks = ChunkList()
                            audio_writer.start()

                            # Generate with voice prefix
                            chunk_prompt = f"{voice_for_generation}: {text_chunk}"
                            result = await generator._generate_async(
                                chunk_prompt,
                                audio_writer,
                                max_tokens=MAX_TOKENS
                            )
                            audio_writer.finalize()

                            # Track tokens
                            input_token_count += len(generator.prepare_input(chunk_prompt))
                            output_token_count += len(result.get('all_token_ids', []))

                            # Add silence between chunks (except after last chunk)
                            if i < total_chunks - 1:
                                silence_samples = int((request.silence_duration or LONG_FORM_SILENCE_DURATION) * 22050)
                                silence = np.zeros(silence_samples, dtype=np.float32)
                                chunk_queue.put(("chunk", silence))

                        chunk_queue.put(("done", {"input": input_token_count, "output": output_token_count}))
                    except Exception as e:
                        print(f"Generation error: {e}")
                        import traceback
                        traceback.print_exc()
                        chunk_queue.put(("error", str(e)))

                gen_task = asyncio.create_task(generate_async_long_form())
            else:
                # Standard streaming for short texts
                print(f"[Server] Using standard SSE streaming (estimated {estimated_duration:.1f}s)")

                # Custom list wrapper that pushes chunks to queue
                class ChunkList(list):
                    def append(self, chunk):
                        super().append(chunk)
                        chunk_queue.put(("chunk", chunk))

                audio_writer = StreamingAudioWriter(
                    player,
                    output_file=None,
                    chunk_size=CHUNK_SIZE,
                    lookback_frames=LOOKBACK_FRAMES
                )
                audio_writer.audio_chunks = ChunkList()

                # Start generation in background task
                async def generate_async():
                    nonlocal input_token_count, output_token_count
                    try:
                        audio_writer.start()
                        result = await generator._generate_async(
                            prompt_text,
                            audio_writer,
                            max_tokens=MAX_TOKENS
                        )
                        audio_writer.finalize()

                        # Extract token counts from result
                        input_token_count = len(generator.prepare_input(prompt_text))
                        output_token_count = len(result.get('all_token_ids', []))

                        chunk_queue.put(("done", {"input": input_token_count, "output": output_token_count}))
                    except Exception as e:
                        print(f"Generation error: {e}")
                        import traceback
                        traceback.print_exc()
                        chunk_queue.put(("error", str(e)))

                # Start generation as async task
                gen_task = asyncio.create_task(generate_async())

            # Stream chunks as they arrive
            try:
                while True:
                    msg_type, data = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: chunk_queue.get(timeout=30)
                    )

                    if msg_type == "chunk":
                        # Convert numpy array to int16 PCM
                        pcm_data = (data * 32767).astype(np.int16)

                        # Encode as base64
                        audio_base64 = base64.b64encode(pcm_data.tobytes()).decode('utf-8')

                        # Send SSE event: speech.audio.delta
                        event_data = {
                            "type": "speech.audio.delta",
                            "audio": audio_base64
                        }
                        yield f"data: {json.dumps(event_data)}\n\n"

                    elif msg_type == "done":
                        # Send SSE event: speech.audio.done with usage stats
                        token_counts = data
                        event_data = {
                            "type": "speech.audio.done",
                            "usage": {
                                "input_tokens": token_counts["input"],
                                "output_tokens": token_counts["output"],
                                "total_tokens": token_counts["input"] + token_counts["output"]
                            }
                        }
                        yield f"data: {json.dumps(event_data)}\n\n"
                        break

                    elif msg_type == "error":
                        # Send error event
                        error_data = {
                            "type": "error",
                            "error": data
                        }
                        yield f"data: {json.dumps(error_data)}\n\n"
                        break

            finally:
                await gen_task

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    # Non-streaming mode (complete audio file)
    else:
        try:
            # Estimate duration to determine if we need long-form generation
            from generation.chunking import estimate_duration
            estimated_duration = estimate_duration(request.input)

            # Determine voice for long-form generation
            # In multi-language mode, use language-specific default voice
            if MULTI_LANGUAGE_MODE and voice_to_use == "random":
                voice_for_generation = multi_language_generator.voice_preferences.get(detected_language, "andrew")
            else:
                voice_for_generation = voice_to_use if voice_to_use != "random" else "andrew"

            # Use long-form generation for longer texts
            use_long_form = estimated_duration > 15.0

            if use_long_form:
                print(f"[Server] Using long-form generation (estimated {estimated_duration:.1f}s)")
                result = await generator.generate_long_form_async(
                    text=request.input,
                    voice=voice_for_generation,
                    player=player,
                    max_chunk_duration=request.max_chunk_duration or LONG_FORM_CHUNK_DURATION,
                    silence_duration=request.silence_duration or LONG_FORM_SILENCE_DURATION,
                    max_tokens=MAX_TOKENS
                )
                full_audio = result['audio']
            else:
                # Standard generation for short texts
                print(f"[Server] Using standard generation (estimated {estimated_duration:.1f}s)")
                audio_writer = StreamingAudioWriter(
                    player,
                    output_file=None,
                    chunk_size=CHUNK_SIZE,
                    lookback_frames=LOOKBACK_FRAMES
                )
                audio_writer.start()

                # Generate speech
                result = await generator._generate_async(
                    prompt_text,
                    audio_writer,
                    max_tokens=MAX_TOKENS
                )

                # Finalize and get audio
                audio_writer.finalize()

                if not audio_writer.audio_chunks:
                    raise HTTPException(status_code=500, detail="No audio generated")

                # Concatenate all chunks
                full_audio = np.concatenate(audio_writer.audio_chunks)

            # Return based on response_format
            if request.response_format == "pcm":
                # Return raw PCM (int16)
                pcm_data = (full_audio * 32767).astype(np.int16)
                return Response(
                    content=pcm_data.tobytes(),
                    media_type="application/octet-stream",
                    headers={
                        "Content-Type": "application/octet-stream",
                        "X-Sample-Rate": "22050",
                        "X-Channels": "1",
                        "X-Bit-Depth": "16"
                    }
                )
            else:  # wav
                # Convert to WAV bytes (convert float to int16)
                wav_buffer = io.BytesIO()
                # Convert float audio to int16 PCM
                pcm_audio = (full_audio * 32767).astype(np.int16)
                wav_write(wav_buffer, 22050, pcm_audio)
                wav_buffer.seek(0)

                return Response(
                    content=wav_buffer.read(),
                    media_type="audio/wav"
                )

        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Kani TTS API",
        "version": "1.0.0",
        "endpoints": {
            "/v1/audio/speech": "POST - OpenAI-compatible speech generation",
            "/health": "GET - Health check"
        }
    }


if __name__ == "__main__":
    import uvicorn
    print("🎤 Starting Kani TTS Server...")
    uvicorn.run(app, host="0.0.0.0", port=32855)
