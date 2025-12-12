"""Spanish-only TTS server (port 8001)"""

import io
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field
from typing import Optional, Literal
import numpy as np
from scipy.io.wavfile import write as wav_write

from audio import LLMAudioPlayer, StreamingAudioWriter
from generation.vllm_generator import VLLMTTSGenerator
from config import CHUNK_SIZE, LOOKBACK_FRAMES, TEMPERATURE, TOP_P, MAX_TOKENS

from nemo.utils.nemo_logging import Logger

nemo_logger = Logger()
nemo_logger.remove_stream_handlers()

ES_MODEL = "nineninesix/kani-tts-400m-es"

app = FastAPI(title="Kani TTS API - Spanish", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

generator = None
player = None


class SpeechRequest(BaseModel):
    model: str = Field(default="tts-1", description="Model name (OpenAI compatible)")
    input: str = Field(..., description="Text to convert to speech")
    voice: str = Field(default="nova", description="Spanish voice")
    response_format: str = Field(default="wav", description="Audio format")

    class Config:
        extra = "ignore"


@app.on_event("startup")
async def startup_event():
    global generator, player
    print(f"🇪🇸 Initializing Spanish TTS (port 8001)...")

    generator = VLLMTTSGenerator(
        model_name=ES_MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.2,
        max_model_len=2048,
    )

    await generator.initialize_engine()
    player = LLMAudioPlayer(generator.tokenizer)
    print("✅ Spanish TTS ready!")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "language": "es",
        "model": ES_MODEL,
        "voices": ["nova", "ballad", "ash"],
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "tts-1",
                "object": "model",
                "created": 1686935000,
                "owned_by": "kani-tts",
            }
        ],
    }


@app.post("/v1/audio/speech")
async def generate_speech(request: SpeechRequest):
    if not generator or not player:
        raise HTTPException(status_code=503, detail="TTS not initialized")

    prompt_text = request.input

    audio_writer = StreamingAudioWriter(
        player, output_file=None, chunk_size=CHUNK_SIZE, lookback_frames=LOOKBACK_FRAMES
    )
    audio_writer.start()

    result = await generator._generate_async(
        prompt_text, audio_writer, max_tokens=MAX_TOKENS
    )

    audio_writer.finalize()

    if not audio_writer.audio_chunks:
        raise HTTPException(status_code=500, detail="No audio generated")

    full_audio = np.concatenate(audio_writer.audio_chunks)

    if request.response_format == "pcm":
        pcm_data = (full_audio * 32767).astype(np.int16)
        return Response(
            content=pcm_data.tobytes(),
            media_type="application/octet-stream",
            headers={
                "Content-Type": "application/octet-stream",
                "X-Sample-Rate": "22050",
                "X-Channels": "1",
                "X-Bit-Depth": "16",
            },
        )
    else:
        wav_buffer = io.BytesIO()
        wav_write(wav_buffer, 22050, full_audio)
        wav_buffer.seek(0)
        return Response(content=wav_buffer.read(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    print("🇪🇸 Starting Spanish TTS Server on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
