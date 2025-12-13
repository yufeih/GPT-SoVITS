import os
import sys
import uvicorn
import base64
import soundfile as sf
import io
import torch
import onnxruntime
import traceback
from fastapi import FastAPI, Body, Query, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import json

from inference_webui import get_tts_wav

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
    
    
    async def audio_generator():
        out_buffer = io.BytesIO()
        sf_file = None
        read_pos = 0

        try:
            async for sampling_rate, audio_data in synthesis_result:
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

@app.websocket("/api/gptsovits/tts/ws")
async def tts_websocket(websocket: WebSocket):
    await websocket.accept()
    
    tts_task = None
    tts_queue = asyncio.Queue()
    session_conf = {}

    try:
        while True:
            message = await websocket.receive_json()
            
            if "session" in message:
                if tts_task:
                    continue
                session_conf = message["session"]
                
                async def text_provider():
                    while True:
                        t = await tts_queue.get()
                        if t is None: break
                        yield t

                # Extract config
                ref_audio_b64 = session_conf.get("refAudio")
                
                def make_ref_provider(b64_data):
                    return lambda: base64.b64decode(b64_data) if b64_data else None

                voice = session_conf.get("voice")
                ref_text = session_conf.get("refText")
                ref_lang = session_conf.get("refLang")
                text_lang = session_conf.get("textLang", "中文")

                async def tts_runner(r_fn, v, rt, rl, tl):
                    try:
                        gen = get_tts_wav(
                            ref_audio_fn=r_fn,
                            ref_id=v,
                            prompt_text=rt,
                            prompt_language=rl,
                            text=text_provider(),
                            text_language=tl,
                            how_to_cut="按标点符号切"
                        )
                        
                        out_buffer = io.BytesIO()
                        sf_file = None
                        read_pos = 0

                        try:
                            async for sr, audio_data in gen:
                                if sf_file is None:
                                    sf_file = sf.SoundFile(
                                        out_buffer, 
                                        mode='w', 
                                        samplerate=sr, 
                                        channels=1, 
                                        format='mp3'
                                    )
                                
                                sf_file.write(audio_data)
                                sf_file.flush()

                                # Read new data
                                out_buffer.seek(read_pos)
                                new_data = out_buffer.read()
                                read_pos += len(new_data)
                                
                                # Reset to end for next write
                                out_buffer.seek(0, 2)
                                
                                if new_data:
                                    await websocket.send_bytes(new_data)
                        finally:
                            if sf_file:
                                sf_file.close()
                                # Yield any remaining data (footer/header updates)
                                out_buffer.seek(read_pos)
                                remaining = out_buffer.read()
                                if remaining:
                                    await websocket.send_bytes(remaining)
                        
                        await websocket.send_json({"done": True})
                    except Exception as e:
                        traceback.print_exc()
                        await websocket.send_json({"error": str(e)})

                tts_task = asyncio.create_task(tts_runner(
                    make_ref_provider(ref_audio_b64),
                    voice, 
                    ref_text, 
                    ref_lang, 
                    text_lang
                ))
                
            elif "text" in message:
                await tts_queue.put(message["text"])

            elif "done" in message:
                await tts_queue.put(None)
                
    except WebSocketDisconnect:
        if tts_task:
            tts_task.cancel()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=58606)
