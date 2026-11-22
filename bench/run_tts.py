"""Text-to-speech benchmark: time to first playable audio, speed, and intelligibility after a
simulated telephone line (checked by transcribing the result with whisper.cpp).
    .venv/bin/python -m bench.run_tts"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from phone_assistant.conversation.prompt import GREETING
from phone_assistant.speech import stt as S
from phone_assistant.speech import tts as T
from phone_assistant.speech.telephony import resample, telephone

from .corpus import MODELS, wer

FIRST_PHRASES = ["Sure, I can help with that.", "Of course,", "I'm sorry, I don't have that information.",
                 "Thanks for calling,", "Let me make sure I understand,"]
SENTENCES = [
    GREETING,
    "Adham prefers to handle scheduling in writing, so please send him an email with the details.",
    "I can't share that, but I'm happy to pass along a message.",
    "Could you tell me your name and what the call is about?",
    "He will follow up with you by text or email as soon as he can.",
    "I'm an AI assistant, so I can't confirm appointments on his behalf.",
]


def engines():
    v = MODELS / "voices-v1.0.bin"
    yield lambda: T.KokoroTTS(MODELS / "kokoro-v1.0.int8.onnx", v, voice="af_heart")
    yield lambda: T.KokoroTTS(MODELS / "kokoro-v1.0.onnx", v, voice="af_heart")
    yield lambda: T.KokoroTTS(MODELS / "kokoro-v1.0.onnx", v, voice="am_michael")
    yield lambda: T.KokoroTTS(MODELS / "kokoro-v1.0.onnx", v, voice="bf_emma")
    yield lambda: T.PiperTTS(MODELS / "en_US-lessac-medium.onnx")


def main() -> None:
    judge = S.WhisperCpp(MODELS / "ggml-base.en.bin", fit_audio_ctx=False, vocabulary="Adham Elkhouly.")
    results = []
    for make in engines():
        try:
            t = time.perf_counter(); eng = make(); load_s = time.perf_counter() - t
            t = time.perf_counter(); eng.synthesize("Hello there."); first_call_s = time.perf_counter() - t
        except Exception as e:
            print("SKIP engine:", type(e).__name__, str(e)[:160], flush=True)
            continue
        first = []
        for _ in range(3):
            for p in FIRST_PHRASES:
                t = time.perf_counter(); eng.synthesize(p); first.append(time.perf_counter() - t)
        rtf, errs, words, errs_clean = [], 0, 0, 0
        for s in SENTENCES:
            t = time.perf_counter(); pcm = eng.synthesize(s); dt = time.perf_counter() - t
            rtf.append(dt / (len(pcm) / eng.sample_rate))
            a16 = resample(pcm, eng.sample_rate, 16000)
            e, n = wer(s, judge.transcribe(telephone(a16, 16000, noise_snr_db=20, seed=1)))
            errs += e; words += n
            errs_clean += wer(s, judge.transcribe(a16))[0]
        first.sort()
        row = {"engine": eng.name, "load_s": round(load_s, 2), "first_call_ms": round(1000 * first_call_s),
               "first_phrase_median_ms": round(1000 * statistics.median(first)),
               "first_phrase_p95_ms": round(1000 * first[int(0.95 * (len(first) - 1))]),
               "realtime_factor": round(statistics.median(rtf), 3),
               "phone_roundtrip_wer_pct": round(100 * errs / words, 1),
               "clean_roundtrip_wer_pct": round(100 * errs_clean / words, 1)}
        results.append(row)
        print(json.dumps(row), flush=True)
        del eng
    out = Path(__file__).parent / "results" / "tts.json"
    out.write_text(json.dumps({"results": results}, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
