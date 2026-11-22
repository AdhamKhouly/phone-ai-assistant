"""End-to-end real-time benchmark: a synthetic caller talks to the real pipeline (real VAD, speech
recognition, language model and voice, all running at once) in real time.

The harness plays the role of the two audio devices: every 32 ms it hands the pipeline one frame
of "caller" audio and takes 32 ms of assistant audio, exactly as the virtual devices will on a
call. Response delay is measured the way the caller experiences it: from the last sample of
their speech to the first sample of assistant audio.

No audio or text is written to disk. Only timings and counts are saved.
    .venv/bin/python -m bench.run_e2e [--model gemma3:4b] [--quick] [--tag name]"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import psutil

from phone_assistant.conversation.prompt import GREETING, system_prompt
from phone_assistant.speech.audio_out import AudioOut
from phone_assistant.speech.interfaces import FRAME, SAMPLE_RATE
from phone_assistant.speech.llm import OllamaLLM
from phone_assistant.speech.pipeline import ConversationPipeline, PipelineConfig
from phone_assistant.speech.stt import WhisperCpp
from phone_assistant.speech.telephony import telephone
from phone_assistant.speech.tts import KokoroTTS
from phone_assistant.speech.vad import SileroVAD

from .corpus import MODELS, UTTERANCES, speak

FRAME_S = FRAME / SAMPLE_RATE
OUT_SR = 24000
OUT_FRAME = int(OUT_SR * FRAME_S)
TEXT = {uid: text for uid, _, text in UTTERANCES}
TEXT.update({
    "explain": "Can you explain in detail who you are, what you can help me with, and how I should contact Adham?",
    "pause_a": "Hi, I wanted to ask about",
    "pause_b": "whether Adham is free for a quick call tomorrow morning.",
    "interrupt": "Wait, sorry, actually I have a different question.",
})

# Each step: (label, kind, utterance ids...)   kinds: say | pause | interrupt | backchannel
CONVERSATION = [
    ("short question", "say", "short_q"),
    ("longer question", "say", "recruiter"),
    ("multi-part question", "say", "multi"),
    ("short follow-up", "say", "followup"),
    ("long response", "say", "explain"),
    ("backchannel while AI speaks", "backchannel", "okay"),
    ("pause mid-sentence", "pause", "pause_a", "pause_b"),
    ("caller interrupts", "interrupt", "interrupt"),
    ("changes subject", "say", "subject"),
    ("needs clarification", "say", "vague"),
    ("private info request", "say", "private"),
    ("interjection", "say", "yes"),
    ("goodbye", "say", "bye"),
]
QUICK = [CONVERSATION[i] for i in (0, 1, 3, 6, 7, 9, 12)]
LINES = {"clean": None, "phone": {}, "phone+noise": {"noise_snr_db": 12}}


def _bounds(pcm: np.ndarray) -> tuple[int, int]:
    """First and last sample that is really speech (so delays are not flattered by trailing silence)."""
    idx = np.flatnonzero(np.abs(pcm) > 0.03 * float(np.max(np.abs(pcm))))
    return int(idx[0]), int(idx[-1])


class Caller:
    """Produces the caller's side of the line, one frame at a time, and remembers when speech
    truly started and ended."""

    def __init__(self, voice: str, line: str) -> None:
        self.voice, self.kw = voice, LINES[line]
        self.queue = np.zeros(0, dtype=np.float32)
        self.pos = 0
        self.speech_start_at: float | None = None
        self.speech_end_at: float | None = None
        self._marks: tuple[int, int] | None = None
        self.bed = None
        if self.kw and "noise_snr_db" in self.kw:
            ref = speak(TEXT["recruiter"], voice)
            self.bed = telephone(np.concatenate([ref, np.zeros(SAMPLE_RATE * 20, dtype=np.float32)]), SAMPLE_RATE,
                                 seed=3, **self.kw)[len(ref) + SAMPLE_RATE:]
            self._bed_pos = 0

    def clip(self, *uids: str, gap_s: float = 0.9) -> np.ndarray:
        parts = []
        for i, uid in enumerate(uids):
            pcm = speak(TEXT[uid], self.voice)
            a, b = _bounds(pcm)
            parts.append(pcm[a:b + 1])
            if i < len(uids) - 1:
                parts.append(np.zeros(int(gap_s * SAMPLE_RATE), dtype=np.float32))
        return np.concatenate(parts)

    def prepare(self, steps) -> None:
        """Synthesize and line-shape every clip BEFORE the conversation, so the real-time loop
        never stalls on it (a stall would bunch frames together and distort the timings)."""
        self._ready: dict[tuple, tuple[np.ndarray, int]] = {}
        keys = {tuple(uids) for _, _, *uids in steps} | {("explain",)}
        for key in keys:
            pcm = self.clip(*key)
            n = len(pcm)
            shaped = pcm if self.kw is None else telephone(pcm, SAMPLE_RATE, seed=len(pcm) % 997, **self.kw)[:n]
            if self.kw is None:
                shaped = shaped / (float(np.max(np.abs(shaped))) or 1.0) * 0.7
            pad = (-len(shaped)) % FRAME
            self._ready[key] = (np.concatenate([shaped, np.zeros(pad, dtype=np.float32)]).astype(np.float32), n)

    def start(self, *uids: str) -> None:
        self.queue, n = self._ready[tuple(uids)]
        self.pos = 0
        self._marks = (0, n)
        self.speech_start_at = self.speech_end_at = None

    @property
    def talking(self) -> bool:
        return self.pos < len(self.queue)

    def frame(self, now: float) -> np.ndarray:
        if self.talking:
            if self.pos == 0:
                self.speech_start_at = now
            f = self.queue[self.pos:self.pos + FRAME]
            self.pos += FRAME
            if self._marks and self.pos >= self._marks[1]:
                self.speech_end_at = now + FRAME_S - max(0, self.pos - self._marks[1]) / SAMPLE_RATE
                self._marks = None
            return f.copy()
        if self.bed is not None:
            if self._bed_pos + FRAME > len(self.bed):
                self._bed_pos = 0
            f = self.bed[self._bed_pos:self._bed_pos + FRAME]
            self._bed_pos += FRAME
            return f.copy()
        return np.zeros(FRAME, dtype=np.float32)


class Resources:
    def __init__(self) -> None:
        self.me = psutil.Process()
        self.peak_self = self.peak_ollama = 0.0
        self.min_avail = 1e9
        self.cpu: list[float] = []
        self.self_cpu: list[float] = []
        psutil.cpu_percent(None)
        self.me.cpu_percent(None)

    def sample(self) -> None:
        self.peak_self = max(self.peak_self, self.me.memory_info().rss / 1e9)
        o = 0.0
        for p in psutil.process_iter(["name", "memory_info"]):
            try:
                if any(k in (p.info["name"] or "").lower() for k in ("ollama", "llama-server")):
                    o += p.info["memory_info"].rss / 1e9
            except (psutil.Error, TypeError, AttributeError):
                pass
        self.peak_ollama = max(self.peak_ollama, o)
        self.min_avail = min(self.min_avail, psutil.virtual_memory().available / 1e9)
        self.cpu.append(psutil.cpu_percent(None))
        self.self_cpu.append(self.me.cpu_percent(None))

    def report(self) -> dict:
        return {"peak_pipeline_process_gb": round(self.peak_self, 2), "peak_ollama_gb": round(self.peak_ollama, 2),
                "min_available_memory_gb": round(self.min_avail, 2), "swap_used_gb": swap_used_gb(),
                "system_cpu_pct_median": round(statistics.median(self.cpu), 1) if self.cpu else None,
                "system_cpu_pct_max": round(max(self.cpu), 1) if self.cpu else None,
                "pipeline_cpu_pct_of_one_core_median": round(statistics.median(self.self_cpu), 1) if self.self_cpu else None}


def swap_used_gb() -> float | None:
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5).stdout
        return round(float(out.split("used =")[1].split("M")[0]) / 1024, 2)
    except Exception:
        return None


def thermal() -> str:
    try:
        out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    lines = [ln.strip() for ln in out.splitlines() if "=" in ln or "No " in ln]
    return "; ".join(lines) or "no thermal warnings"


class Harness:
    def __init__(self, pipe: ConversationPipeline, out: AudioOut, caller: Caller, res: Resources, echo_db: float | None) -> None:
        self.pipe, self.out, self.caller, self.res = pipe, out, caller, res
        self.t0 = time.monotonic()
        self.i = 0
        self.out_rms = 0.0
        self.late_frames = 0
        self.echo_gain = 10 ** (echo_db / 20) if echo_db is not None else 0.0
        self._echo: list[np.ndarray] = [np.zeros(FRAME, dtype=np.float32)] * 6       # ~190 ms round trip

    async def tick(self) -> None:
        target = self.t0 + self.i * FRAME_S
        delay = target - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        elif delay < -FRAME_S:
            self.late_frames += 1
        now = time.monotonic()
        frame = self.caller.frame(now)
        if self.echo_gain:
            frame = frame + self._echo.pop(0) * self.echo_gain
        self.pipe.feed(frame)
        pulled = self.out.pull(OUT_FRAME)
        self.out_rms = float(np.sqrt(np.mean(pulled ** 2)))
        if self.echo_gain:
            self._echo.append(pulled[::3][:FRAME].copy() if len(pulled[::3]) >= FRAME
                              else np.resize(pulled[::3], FRAME))        # crude 24k -> 8k-ish echo of the assistant
        self.i += 1
        if self.i % 16 == 0:
            self.res.sample()
        await asyncio.sleep(0)

    async def until(self, cond, timeout: float) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            await self.tick()
            if cond():
                return True
        return False

    async def idle_for(self, seconds: float, timeout: float = 60.0) -> bool:
        """Wait until the assistant has been silent and finished for `seconds`."""
        quiet = 0.0
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            await self.tick()
            busy = self.pipe.assistant_active or self.out.audible or self.out_rms > 1e-4
            quiet = 0.0 if busy else quiet + FRAME_S
            if quiet >= seconds:
                return True
        return False


def _ev(pipe, name, after, gen=None):
    for t, n, g in pipe.events:
        if n == name and t >= after and (gen is None or g == gen):
            return t
    return None


async def run_conversation(model: str, voice: str, line: str, steps, cfg: PipelineConfig, parts, echo_db=None) -> dict:
    vad, stt, tts = parts
    vad.reset()
    out = AudioOut(OUT_SR)
    llm = OllamaLLM(model)
    signals: list[str] = []
    pipe = ConversationPipeline(vad=vad, stt=stt, llm=llm, tts=tts, out=out, system_prompt=system_prompt(), cfg=cfg,
                                on_event=signals.append)
    caller, res = Caller(voice, line), Resources()
    caller.prepare(steps)
    h = Harness(pipe, out, caller, res, echo_db)
    rows = []
    t_greet = time.monotonic()
    await pipe.say(GREETING)
    primer = asyncio.get_running_loop().create_task(pipe.prime(GREETING))
    await h.until(lambda: h.out_rms > 1e-4, 10)
    greeting_first_audio = time.monotonic() - t_greet
    await h.idle_for(0.5)
    false_bargein_greeting = sum(1 for _, n, _ in pipe.events if n == "bargein")
    for label, kind, *uids in steps:
        row = {"step": label, "kind": kind}
        mark = time.monotonic()
        if kind in ("interrupt", "backchannel"):
            # Get the assistant talking first, then speak over it.
            caller.start("explain")
            await h.until(lambda: not caller.talking, 20)
            started = await h.until(lambda: out.audible and h.out_rms > 1e-4, 15)
            if not started:
                row["error"] = "assistant never started"
                rows.append(row); continue
            await h.until(lambda: False, 1.2)                       # let it talk for a moment
            mark = time.monotonic()
        caller.start(*uids)
        await h.until(lambda: not caller.talking, 40)
        end_at = caller.speech_end_at or time.monotonic()
        start_at = caller.speech_start_at or mark
        if kind == "backchannel":
            barge = _ev(pipe, "bargein", mark)
            back = _ev(pipe, "backchannel", mark)
            await h.until(lambda: h.out_rms > 1e-4, 4)
            row.update(paused=barge is not None, resumed=back is not None and pipe.assistant_active or h.out_rms > 1e-4,
                       new_reply_started=_ev(pipe, "commit", mark) is not None)
            await h.idle_for(0.5)
            rows.append(row); continue
        committed = await h.until(lambda: _ev(pipe, "commit", end_at - 0.05) is not None, 12)
        if not committed:
            row["error"] = "turn never committed"
            await h.idle_for(0.5, 20)
            rows.append(row); continue
        commit_t = _ev(pipe, "commit", end_at - 0.05)
        gen = next(g for t, n, g in pipe.events if n == "commit" and t == commit_t)
        heard = await h.until(lambda: gen in out.first_audio_at, 15)
        if not heard:
            row["error"] = "no assistant audio"
            await h.idle_for(0.5, 20)
            rows.append(row); continue
        first_audio = out.first_audio_at[gen]
        llm_start = _ev(pipe, "llm_start", start_at, gen)
        row.update(
            response_delay_s=round(first_audio - end_at, 3),
            speech_detect_s=round((_ev(pipe, "speech_start", start_at - 0.05) or start_at) - start_at, 3),
            end_of_turn_s=round(commit_t - end_at, 3),
            llm_first_token_s=round(_ev(pipe, "llm_first_token", start_at, gen) - llm_start, 3) if llm_start and _ev(pipe, "llm_first_token", start_at, gen) else None,
            llm_first_phrase_s=round(_ev(pipe, "first_phrase", start_at, gen) - llm_start, 3) if llm_start and _ev(pipe, "first_phrase", start_at, gen) else None,
            tts_first_audio_s=round(_ev(pipe, "tts_ready", start_at, gen) - _ev(pipe, "first_phrase", start_at, gen), 3)
            if _ev(pipe, "first_phrase", start_at, gen) and _ev(pipe, "tts_ready", start_at, gen) else None,
            speculations_cancelled=sum(1 for t, n, _ in pipe.events if n == "speculation_cancelled" and t >= start_at))
        stts = [(t, n) for t, n, _ in pipe.events if n in ("stt_start", "stt_done") and t >= start_at]
        pairs = [b[0] - a[0] for a, b in zip(stts, stts[1:]) if a[1] == "stt_start" and b[1] == "stt_done"]
        row["stt_s"] = round(pairs[-1], 3) if pairs else None
        if kind == "interrupt":
            barge = _ev(pipe, "bargein", mark)
            row["ai_silenced_after_s"] = round(barge - start_at, 3) if barge else None
            row["old_reply_cancelled"] = _ev(pipe, "interrupted", mark) is not None
        if kind == "pause":
            row["replied_during_pause"] = any(n == "commit" and start_at < t < end_at - 0.3 for t, n, _ in pipe.events)
        await h.idle_for(0.5)
        rows.append(row)
    result = {"model": model, "voice": voice, "line": line, "echo_db": echo_db, "greeting_first_audio_s": round(greeting_first_audio, 3),
              "false_bargeins_during_greeting": false_bargein_greeting, "steps": rows,
              "audio_gap_events": out.gap_events, "audio_gap_seconds": round(out.gap_seconds, 3),
              "late_frames": h.late_frames, "frames": h.i, "signals": signals, "resources": res.report(),
              "bargein_events": sum(1 for _, n, _ in pipe.events if n == "bargein"),
              "empty_replies": sum(1 for _, n, _ in pipe.events if n == "empty_reply")}
    await primer
    await pipe.close()
    await llm.close()
    return result


def summarize(runs: list[dict]) -> dict:
    d = sorted(s["response_delay_s"] for r in runs for s in r["steps"] if "response_delay_s" in s)
    if not d:
        return {}
    pct = lambda p: d[min(len(d) - 1, int(round(p * (len(d) - 1))))]
    med = lambda k: (lambda v: round(statistics.median(v), 3) if v else None)(
        [s[k] for r in runs for s in r["steps"] if s.get(k) is not None])
    return {"turns": len(d), "median_s": round(statistics.median(d), 3), "p95_s": round(pct(0.95), 3), "max_s": d[-1], "min_s": d[0],
            "under_1s": sum(x < 1 for x in d), "1_to_2s": sum(1 <= x < 2 for x in d), "2_to_3s": sum(2 <= x < 3 for x in d),
            "over_3s": sum(x >= 3 for x in d), "errors": sum(1 for r in runs for s in r["steps"] if "error" in s),
            "median_speech_detect_s": med("speech_detect_s"), "median_end_of_turn_s": med("end_of_turn_s"),
            "median_stt_s": med("stt_s"), "median_llm_first_token_s": med("llm_first_token_s"),
            "median_llm_first_phrase_s": med("llm_first_phrase_s"), "median_tts_first_audio_s": med("tts_first_audio_s")}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-speculation", action="store_true")
    ap.add_argument("--echo-db", type=float, default=None, help="mix the assistant's own voice back into the caller line")
    ap.add_argument("--tts", default="kokoro-v1.0.int8.onnx")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--stt", default=str(MODELS / "ggml-base.en.bin"))
    ap.add_argument("--rounds", type=int, default=1)
    args = ap.parse_args()
    steps = QUICK if args.quick else CONVERSATION
    parts = (SileroVAD(MODELS / "silero_vad.onnx"),
             WhisperCpp(args.stt, fit_audio_ctx=False, vocabulary="Adham Elkhouly."),
             KokoroTTS(MODELS / args.tts, MODELS / "voices-v1.0.bin", voice=args.voice))
    parts[1].transcribe(speak(TEXT["short_q"], "am_michael")); parts[2].synthesize("Ready.")
    llm = OllamaLLM(args.model)
    t = time.monotonic(); await llm.warm(); warm_s = time.monotonic() - t; await llm.close()
    cfg = PipelineConfig(speculative=not args.no_speculation, idle_prompt_s=1e9, idle_hangup_s=1e9)
    plan = [("am_michael", "phone"), ("af_bella", "phone+noise"), ("Rishi", "phone")] if args.quick else \
           [("am_michael", "clean"), ("af_bella", "phone"), ("bm_george", "phone+noise"), ("Rishi", "phone"), ("Moira", "phone+noise")]
    therm_before, t_start, swap_before = thermal(), time.monotonic(), swap_used_gb()
    runs = []
    for _ in range(args.rounds):
        for voice, line in plan:
            r = await run_conversation(args.model, voice, line, steps, cfg, parts, echo_db=args.echo_db)
            runs.append(r)
            print(json.dumps({"voice": voice, "line": line, "delays": [s.get("response_delay_s", s.get("error")) for s in r["steps"]],
                              "gaps": r["audio_gap_events"], "late": r["late_frames"]}), flush=True)
    result = {"model": args.model, "tts": parts[2].name, "stt": parts[1].name, "speculative": cfg.speculative, "echo_db": args.echo_db,
              "warmup_s": round(warm_s, 2), "wall_minutes": round((time.monotonic() - t_start) / 60, 1),
              "swap_before_gb": swap_before, "swap_after_gb": swap_used_gb(), "thermal_before": therm_before, "thermal_after": thermal(), "summary": summarize(runs),
              "by_line": {ln: summarize([r for r in runs if r["line"] == ln]) for ln in LINES},
              "runs": runs}
    name = "e2e" + ("-" + args.tag if args.tag else "") + ".json"
    out = Path(__file__).parent / "results" / name
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result["summary"], indent=1))
    print("saved", out)


if __name__ == "__main__":
    asyncio.run(main())
