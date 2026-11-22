"""Speech-recognition benchmark: accuracy (WER) and latency per engine, voice and line condition.
    .venv/bin/python -m bench.run_stt"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from phone_assistant.speech import stt as S
from phone_assistant.speech.telephony import telephone

from .corpus import KOKORO_VOICES, MODELS, UTTERANCES, say_voices_available, speak, wer

SMALL_EN = Path.home() / "Desktop/Claude/Personal Assistant/System/Models/ggml-small.en.bin"   # already on this Mac
VOCAB = "Adham Elkhouly."
CONDITIONS = {"clean": None, "phone": {}, "phone+noise15dB": {"noise_snr_db": 15}, "phone+noise5dB": {"noise_snr_db": 5}}


def engines():
    yield lambda: S.WhisperCpp(MODELS / "ggml-base.en.bin", fit_audio_ctx=True, vocabulary=VOCAB)
    yield lambda: S.WhisperCpp(MODELS / "ggml-base.en.bin", fit_audio_ctx=False, vocabulary=VOCAB)
    if SMALL_EN.exists():
        yield lambda: S.WhisperCpp(SMALL_EN, fit_audio_ctx=True, vocabulary=VOCAB)
    yield lambda: S.MlxWhisper("mlx-community/whisper-base.en-mlx")
    yield lambda: S.Moonshine("moonshine/base")
    yield lambda: S.Moonshine("moonshine/tiny")


def main() -> None:
    voices = KOKORO_VOICES + say_voices_available()
    print("voices:", voices, flush=True)
    clips = {(uid, v): speak(text, v) for uid, _, text in UTTERANCES for v in voices}
    kinds = {uid: kind for uid, kind, _ in UTTERANCES}
    refs = {uid: text for uid, _, text in UTTERANCES}
    results = []
    for make in engines():
        try:
            t = time.time(); eng = make(); load_s = time.time() - t
            eng.transcribe(clips[("recruiter", voices[0])])          # warm-up
        except Exception as e:
            print("SKIP engine:", type(e).__name__, str(e)[:120], flush=True)
            continue
        row = {"engine": eng.name, "load_s": round(load_s, 2), "conditions": {}, "by_voice": {}, "interjections_ok": None}
        inter_ok = inter_n = 0
        for cond, kw in CONDITIONS.items():
            errs = words = 0
            lat = {"short": [], "long": [], "interjection": []}
            for (uid, v), pcm in clips.items():
                audio = pcm if kw is None else telephone(pcm, 16000, seed=hash((uid, v)) % 1000, **kw)
                t = time.perf_counter(); hyp = eng.transcribe(audio); dt = time.perf_counter() - t
                e, n = wer(refs[uid], hyp)
                errs += e; words += n
                lat[kinds[uid]].append(dt)
                if cond == "phone":
                    be, bn = row["by_voice"].get(v, (0, 0)); row["by_voice"][v] = (be + e, bn + n)
                    if kinds[uid] == "interjection":
                        inter_n += 1; inter_ok += (e == 0)
            allv = sorted(lat["short"] + lat["long"] + lat["interjection"])
            row["conditions"][cond] = {
                "wer_pct": round(100 * errs / words, 1),
                "median_ms": round(1000 * statistics.median(allv)), "p95_ms": round(1000 * allv[int(0.95 * (len(allv) - 1))]),
                "median_ms_short": round(1000 * statistics.median(lat["short"] + lat["interjection"])),
                "median_ms_long": round(1000 * statistics.median(lat["long"]))}
        row["by_voice"] = {v: round(100 * e / n, 1) for v, (e, n) in row["by_voice"].items()}
        row["interjections_ok"] = f"{inter_ok}/{inter_n}"
        results.append(row)
        print(json.dumps(row), flush=True)
        del eng
    out = Path(__file__).parent / "results" / "stt.json"
    out.write_text(json.dumps({"voices": voices, "utterances": len(UTTERANCES), "results": results}, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
