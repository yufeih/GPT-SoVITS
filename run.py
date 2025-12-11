import os
import sys
import uvicorn
import base64
import soundfile as sf
import io
import torch
import onnxruntime
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

class CUDASupportResponse(BaseModel):
    torch: bool
    onnx: bool

@app.get("/api/gptsovits/cuda")
async def cuda_support_endpoint():
    return CUDASupportResponse(
        torch=torch.cuda.is_available(),
        onnx="CUDAExecutionProvider" in onnxruntime.get_available_providers()
    )

@app.post("/api/gptsovits/tts")
async def tts_endpoint(
    text: str = Query(...),
    voice: str = Query(...),
    request: TTSRequest = Body(...)
):
    synthesis_result = get_tts_wav(
        ref_audio_fn=lambda: base64.b64decode(request.refAudio),
        ref_id=voice,
        prompt_text=request.refText,
        prompt_language=request.refLang,
        text=text,
        text_language='中文',
        how_to_cut="按标点符号切"
    )
    
    
    def audio_generator():
        out_buffer = io.BytesIO()
        sf_file = None
        read_pos = 0

        try:
            for sampling_rate, audio_data in synthesis_result:
                if sf_file is None:
                    sf_file = sf.SoundFile(
                        out_buffer, 
                        mode='w', 
                        samplerate=sampling_rate, 
                        channels=1, 
                        format='mp3'
                    )
                
                sf_file.write(audio_data)
                sf_file.flush()

                # Read new data
                current_pos = out_buffer.tell()
                out_buffer.seek(read_pos)
                new_data = out_buffer.read()
                read_pos += len(new_data)
                
                # Reset to end for next write
                out_buffer.seek(0, 2)
                
                if new_data:
                    yield new_data
        finally:
            if sf_file:
                sf_file.close()
                # Yield any remaining data (footer/header updates)
                out_buffer.seek(read_pos)
                remaining = out_buffer.read()
                if remaining:
                    yield remaining

    return StreamingResponse(audio_generator(), media_type="audio/mpeg")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=58606)
