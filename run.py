import os
import sys
import uvicorn
import base64
import soundfile as sf
import io
from fastapi import FastAPI, Body, Query, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add GPT_SoVITS to path
sys.path.append(os.path.join(os.path.dirname(__file__), "GPT_SoVITS"))

from GPT_SoVITS.inference_webui import get_tts_wav

app = FastAPI()

class TTSRequest(BaseModel):
    refAudio: str  # Base64 encoded audio
    refText: str
    refLang: str

@app.post("/api/gptsovits/tts")
async def tts_endpoint(
    text: str = Query(...),
    voice: str = Query(...),
    lang: str = Query(...),
    request: TTSRequest = Body(...)
):
    synthesis_result = get_tts_wav(
        ref_audio_fn=lambda: base64.b64decode(request.refAudio),
        ref_id=voice,
        prompt_text=request.refText,
        prompt_language=request.refLang,
        text=text,
        text_language=lang,
    )
    
    result_list = list(synthesis_result)

    if not result_list:
         raise HTTPException(status_code=500, detail="TTS generation failed (no output)")

    last_sampling_rate, last_audio_data = result_list[-1]

    out_buffer = io.BytesIO()
    sf.write(out_buffer, last_audio_data, last_sampling_rate, format='mp3')
    out_buffer.seek(0)

    return StreamingResponse(out_buffer, media_type="audio/mpeg")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=58606)
