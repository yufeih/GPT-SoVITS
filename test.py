import os
import soundfile as sf

from GPT_SoVITS.inference_webui import get_tts_wav

def synthesize(
    ref_audio_path,
    ref_text_path,
    ref_language,
    target_text_path,
    target_language,
    output_path,
):
    # Read reference text
    with open(ref_text_path, "r", encoding="utf-8") as file:
        ref_text = file.read()

    # Read target text
    with open(target_text_path, "r", encoding="utf-8") as file:
        target_text = file.read()

    # Synthesize audio
    synthesis_result = get_tts_wav(
        ref_wav_path=ref_audio_path,
        prompt_text=ref_text,
        prompt_language=ref_language,
        text=target_text,
        text_language=target_language,
        top_p=1,
        temperature=1,
    )

    result_list = list(synthesis_result)

    if result_list:
        last_sampling_rate, last_audio_data = result_list[-1]
        output_wav_path = os.path.join(output_path, "output.wav")
        sf.write(output_wav_path, last_audio_data, last_sampling_rate)
        print(f"Audio saved to {output_wav_path}")

synthesize(
    'test.m4a',
    'test.txt',
    '中文',
    'target.txt',
    '中文',
    '.',
)

