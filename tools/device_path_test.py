"""Offline test of both call audio paths through the real virtual devices, with stand-ins for DIGITS.
No call, no default-device change, no microphone, nothing saved.

    AI voice  -> BlackHole 2ch  -> [stand-in for DIGITS' microphone: records, then Whisper checks the words]
    [stand-in for DIGITS' speaker: plays a synthetic caller] -> BlackHole 16ch -> the real pipeline

    .venv/bin/python tools/device_path_test.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.corpus import MODELS, speak, wer                                   # noqa: E402
from phone_assistant.audio import coreaudio as ca                             # noqa: E402
from phone_assistant.audio.io import DEVICE_RATE, CallAudioIO, virtual_device_index  # noqa: E402
from phone_assistant.conversation.prompt import GREETING, system_prompt       # noqa: E402
from phone_assistant.speech.audio_out import AudioOut                         # noqa: E402
from phone_assistant.speech.llm import OllamaLLM                              # noqa: E402
from phone_assistant.speech.pipeline import ConversationPipeline, PipelineConfig  # noqa: E402
from phone_assistant.speech.stt import WhisperCpp                             # noqa: E402
from phone_assistant.speech.telephony import resample                         # noqa: E402
from phone_assistant.speech.tts import KokoroTTS                              # noqa: E402
from phone_assistant.speech.vad import SileroVAD                              # noqa: E402

MIC, CAP = "BlackHole 2ch", "BlackHole 16ch"
CALLER_LINES = ["Hello, is he available right now?", "Okay, can you ask him to email me about the interview?", "Is Adham there?"]


async def main() -> int:
    import sounddevice as sd
    before = (ca.default_input().uid, ca.default_output().uid)
    stt = WhisperCpp(MODELS / "ggml-base.en.bin", vocabulary="Adham Elkhouly.")
    tts = KokoroTTS(MODELS / "kokoro-v1.0.int8.onnx", MODELS / "voices-v1.0.bin", voice="af_heart")
    vad = SileroVAD(MODELS / "silero_vad.onnx")
    llm = OllamaLLM("qwen3:8b")
    await llm.warm()
    out = AudioOut(DEVICE_RATE)
    io = CallAudioIO(CAP, MIC, out)
    heard_by_ai: list[str] = []
    pipe = ConversationPipeline(vad=vad, stt=stt, llm=llm, tts=tts, out=out, system_prompt=system_prompt(),
                                cfg=PipelineConfig(idle_prompt_s=1e9, idle_hangup_s=1e9))
    pipe.debug_text = lambda who, t: heard_by_ai.append(t) if who == "user" else None   # synthetic text only

    digits_mic: list[np.ndarray] = []            # what DIGITS would get as its microphone
    mic_times: list[float] = []
    rec = sd.InputStream(device=virtual_device_index(MIC, want_input=True), samplerate=DEVICE_RATE, channels=1,
                         dtype="float32", callback=lambda d, f, t, s: (digits_mic.append(d[:, 0].copy()), mic_times.append(time.monotonic())))
    play_buf = {"pcm": np.zeros(0, dtype=np.float32), "pos": 0, "t0": None}

    def digits_speaker(outdata, frames, t, s):   # what DIGITS would play: the caller
        chunk = play_buf["pcm"][play_buf["pos"]:play_buf["pos"] + frames]
        if len(chunk) and play_buf["t0"] is None:
            play_buf["t0"] = time.monotonic()
        outdata[:, :] = 0.0
        outdata[:len(chunk), 0] = chunk
        play_buf["pos"] += frames
    spk = sd.OutputStream(device=virtual_device_index(CAP, want_input=False), samplerate=DEVICE_RATE, channels=1,
                          dtype="float32", callback=digits_speaker)
    io.start(); rec.start(); spk.start()
    feeder = asyncio.get_running_loop().create_task(pipe.run(io.frames()))
    ok = True

    def rec_len() -> int:
        return sum(len(b) for b in digits_mic)

    async def wait_quiet(timeout=40.0):
        end, quiet = time.monotonic() + timeout, 0.0
        while time.monotonic() < end and quiet < 0.8:
            await asyncio.sleep(0.1)
            quiet = 0.0 if (pipe.assistant_active or out.audible) else quiet + 0.1

    def words(start: int) -> str:
        x = np.concatenate(digits_mic)[start:]
        return stt.transcribe(resample(x, DEVICE_RATE, 16000)) if len(x) > 8000 else ""

    # 1. Greeting: AI voice must arrive on the virtual microphone, intelligibly, and NOT in the AI's own ears.
    in_peak_before = io.in_peak
    await pipe.say(GREETING)
    await pipe.prime(GREETING)
    await wait_quiet()
    got = words(0)
    e, n = wer(GREETING, got)
    self_heard = sum(1 for _, name, _ in pipe.events if name == "speech_start")
    print(f"1. Greeting on '{MIC}': {n - e}/{n} words recognised after the device  ->  {'PASS' if e <= 2 else 'FAIL'}")
    print(f"   AI's own voice on its listening device '{CAP}': peak {max(io.in_peak, in_peak_before):.5f}, "
          f"speech detections {self_heard}  ->  {'PASS' if self_heard == 0 and io.in_peak < 0.001 else 'FAIL'}")
    ok &= e <= 2 and self_heard == 0 and io.in_peak < 0.001

    # 2. Conversation: synthetic caller in through one device, AI reply out through the other.
    for i, line in enumerate(CALLER_LINES, 1):
        clip = resample(speak(line, "am_michael"), 16000, DEVICE_RATE) * 0.8
        last_speech = int(np.flatnonzero(np.abs(clip) > 0.03 * float(np.max(np.abs(clip))))[-1])
        mark_blocks = len(digits_mic)
        mark = rec_len()
        play_buf["pcm"], play_buf["pos"], play_buf["t0"] = clip, 0, None
        while play_buf["t0"] is None:
            await asyncio.sleep(0.005)
        speech_end = play_buf["t0"] + last_speech / DEVICE_RATE
        first_audio = None
        while first_audio is None and time.monotonic() < speech_end + 10:
            await asyncio.sleep(0.02)
            for j in range(mark_blocks, len(digits_mic)):
                if mic_times[j] > speech_end and float(np.max(np.abs(digits_mic[j]))) > 0.01:
                    first_audio = mic_times[j]
                    break
        delay = (first_audio - speech_end) if first_audio else float("nan")
        await wait_quiet()
        reply = words(mark)
        understood = heard_by_ai[-1] if len(heard_by_ai) >= i else ""
        e, n = wer(line, understood)
        good = e <= 2 and len(reply.split()) >= 3 and delay < 3
        ok &= good
        print(f"2.{i} Caller line through '{CAP}': {n - e}/{n} words understood; reply of {len(reply.split())} words arrived on "
              f"'{MIC}' {delay:.2f} s after the caller's last sound  ->  {'PASS' if good else 'FAIL'}")
    mics = [d.name for d in ca.devices() if d.is_physical_input and d.running]
    after = (ca.default_input().uid, ca.default_output().uid)
    print(f"3. Physical microphones in use during the test: {mics or 'none'}  ->  {'PASS' if not mics else 'FAIL'}")
    print(f"4. Default devices untouched: {before == after}; dropped frames {io.dropped_frames}; stream warnings {io.status_events}; "
          f"gaps in AI audio {out.gap_events}")
    ok &= not mics and before == after
    feeder.cancel()
    spk.stop(); spk.close(); rec.stop(); rec.close(); io.stop()
    await pipe.close(); await llm.close()
    for b in digits_mic:
        b.fill(0.0)
    print("\nDEVICE PATH TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
