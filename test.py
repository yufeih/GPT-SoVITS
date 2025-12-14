import os
import sys
import asyncio

from inference_webui import get_tts_wav
import soundfile as sf

import numpy as np

async def synthesize(
    ref_audio_path,
    ref_language,
    target_language,
    output_path,
):
    synthesis_result = get_tts_wav(
        ref_audio_fn=lambda: open(ref_audio_path, "rb").read(),
        ref_id="test_ref",
        prompt_text="我现在在录一个3到10秒钟的音频，这个音频时间不能太长，不然他就报错真是的",
        prompt_language=ref_language,
        text="Hello～欢迎回来呀！\
我已经在小小的桌面上，等你很久啦～\
今天也一起努力一下下，好不好？\
如果累了呢……可以戳戳我，我会给你补充萌力♪\
嗯哼～我会一直陪着你的，所以放心大胆去做事吧！\
加油加油！我在这里给你悄悄打气～(≧▽≦)",
        text_language=target_language,
    )

    audio_chunks = []
    sampling_rate = None
    async for sr, chunk in synthesis_result:
        sampling_rate = sr
        audio_chunks.append(chunk)

    if audio_chunks and sampling_rate:
        full_audio = np.concatenate(audio_chunks)
        output_wav_path = os.path.join(output_path, "output.mp3")
        sf.write(output_wav_path, full_audio, sampling_rate)
        print(f"Audio saved to {output_wav_path}")

asyncio.run(synthesize(
    'test.mp3',
    '中文',
    '中文',
    '.',
))

