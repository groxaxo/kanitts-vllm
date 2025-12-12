"""Router server that forwards requests to language-specific backends"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional, Literal
import httpx

app = FastAPI(title="Kani TTS Router", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Backend servers
SPANISH_SERVER = "http://localhost:8001"
ENGLISH_SERVER = "http://localhost:8002"

# Voice to language mapping
VOICE_TO_LANGUAGE = {
    "nova": "es",
    "ballad": "es",
    "ash": "es",
    "andrew": "en",
    "katie": "en",
    # Map standard OpenAI voices to defaults
    "alloy": "es",   # Default to Spanish as requested
    "echo": "es",
    "fable": "en",
    "onyx": "en",
    "shimmer": "es"
}


class OpenAISpeechRequest(BaseModel):
    input: str = Field(..., description="Text to convert to speech")
    model: str = Field(default="tts-1", description="Model name")
    voice: str = Field(default="nova", description="Voice name")
    response_format: Optional[str] = Field(default="wav", description="Audio format")

    class Config:
        extra = "ignore"


@app.get("/health")
async def health_check():
    """Check health of all backend servers"""
    spanish_ok = False
    english_ok = False
    
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{SPANISH_SERVER}/health", timeout=2.0)
            spanish_ok = resp.status_code == 200
        except:
            pass
        
        try:
            resp = await client.get(f"{ENGLISH_SERVER}/health", timeout=2.0)
            english_ok = resp.status_code == 200
        except:
            pass
    
    return {
        "status": "healthy" if (spanish_ok or english_ok) else "unhealthy",
        "backends": {
            "spanish": "healthy" if spanish_ok else "unhealthy",
            "english": "healthy" if english_ok else "unhealthy"
        },
        "available_voices": {
            "spanish": ["nova", "ballad", "ash"],
            "english": ["andrew", "katie"]
        }
    }


@app.post("/v1/audio/speech")
async def route_speech(request: OpenAISpeechRequest):
    """Route request to appropriate language server based on voice"""
    
    # Determine which server to use based on voice
    language = VOICE_TO_LANGUAGE.get(request.voice)
    
    if not language:
        raise HTTPException(status_code=400, detail=f"Unknown voice: {request.voice}")
    
    # Select backend server
    # Select backend server and map generic voices
    if language == "es":
        backend_url = f"{SPANISH_SERVER}/v1/audio/speech"
        # If voice is generic OpenAI one, map to nova
        if request.voice not in ["nova", "ballad", "ash"]:
            print(f"[Router] Mapping generic voice '{request.voice}' to 'nova'")
            request.voice = "nova"
        print(f"[Router] Routing '{request.voice}' → Spanish server (8001)")
    else:
        backend_url = f"{ENGLISH_SERVER}/v1/audio/speech"
        # If voice is generic OpenAI one, map to andrew
        if request.voice not in ["andrew", "katie"]:
            print(f"[Router] Mapping generic voice '{request.voice}' to 'andrew'")
            request.voice = "andrew"
        print(f"[Router] Routing '{request.voice}' → English server (8002)")
    
    # Forward request to backend
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                backend_url,
                json={
                    "input": request.input,
                    "voice": request.voice,
                    "response_format": request.response_format
                }
            )
            
            # Return the audio response
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.headers.get("content-type", "audio/wav")
            )
        
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Backend server timeout")
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail=f"{language.upper()} server not available")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/")
async def root():
    return {
        "name": "Kani TTS Router",
        "version": "1.0.0",
        "description": "Routes requests to language-specific TTS backends"
,
        "endpoints": {
            "/v1/audio/speech": "POST - OpenAI-compatible speech generation",
            "/health": "GET - Health check"
        },
        "voices": {
            "spanish": ["nova", "ballad", "ash"],
            "english": ["andrew", "katie"]
        }
    }


if __name__ == "__main__":
    import uvicorn
    print("🔀 Starting TTS Router on port 8000...")
    print("   Spanish (8001): nova, ballad, ash")
    print("   English (8002): andrew, katie")
    uvicorn.run(app, host="0.0.0.0", port=8000)
