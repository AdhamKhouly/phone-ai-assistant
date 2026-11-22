"""What happens on the audio side of a call the assistant answered.

Modes (VOICE_MODE in .env; default "off"):
  off           the virtual microphone carries silence. Nothing is spoken, nothing is listened to.
  greeting      one fixed sentence is spoken into the virtual microphone. No listening.
  conversation  the full local pipeline: listen, think, speak. (Needs separate owner approval.)

In every mode, while the call is up, MicGuard is polled. If a real microphone shows up in the call
path, or the virtual audio stream fails, the assistant's audio is cut and the call is ended; it
never falls back to another device.

The session starts only on the manager's "active" event (which only follows the owner's `answer`)
and is torn down on every way a call can end. Caller and assistant words never leave the pipeline
object, and that object is wiped when the call ends.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

from ..audio.routing import MicGuard, RouteConfig
from ..calls.models import Direction
from ..conversation.prompt import (GREETING, outgoing_system_prompt, summarize_detailed, summarize_outcome,
                                   system_prompt)
from ..speech.audio_out import AudioOut

log = logging.getLogger(__name__)

MODES = ("off", "greeting", "conversation")
GUARD_POLL_S = 0.25
GUARD_GRACE_S = 0.0          # no grace: one bad reading ends the call
MIC_WAIT_S = 20.0            # how long DIGITS gets to open the virtual microphone before anything is spoken
MIC_SETTLE_S = 0.6           # pause after it opens, so the first syllable is not clipped
NO_ANSWER_S = 60.0           # outgoing: nobody spoke for this long after dialling -> hang up
GOODBYE_LINE = "I have to end the call now. Please text or email Adham. Goodbye."


class VoiceSession:
    def __init__(self, manager, *, mode: str = "off", route_cfg: RouteConfig | None = None,
                 guard: MicGuard | None = None, io_factory: Callable | None = None,
                 pipeline_factory: Callable | None = None, notify: Callable[[str], Awaitable[None]] | None = None,
                 max_call_s: float = 60.0, greeting: str = "", mic_wait_s: float = MIC_WAIT_S,
                 summary: str = "categories") -> None:
        if mode not in MODES:
            raise ValueError("VOICE_MODE must be one of: " + ", ".join(MODES))
        self.manager, self.mode = manager, mode
        self.cfg = route_cfg or RouteConfig()
        self.guard = guard or MicGuard(self.cfg)
        self.io_factory = io_factory or self._default_io
        self.pipeline_factory = pipeline_factory
        self.notify = notify
        self.max_call_s = max_call_s
        self.greeting = greeting.strip() or GREETING
        self.mic_wait_s = mic_wait_s
        self.no_answer_s = NO_ANSWER_S
        self.summary = summary       # categories (fixed phrases) | detailed (paraphrased note)
        self._task: asyncio.Task | None = None
        self._call_id: int | None = None
        self._models = None                      # loaded once, reused across calls
        self._preload: asyncio.Task | None = None
        self.last_report: dict = {}
        manager.subscribe(self.on_call_event)

    # ---- wiring ------------------------------------------------------------------------------
    def _default_io(self, out: AudioOut):
        from ..audio.io import CallAudioIO
        return CallAudioIO(self.cfg.capture, self.cfg.mic, out)

    async def on_call_event(self, ev) -> None:
        if ev.kind in ("incoming", "dialing") and self.mode != "off" and self._models is None and self._preload is None:
            # Load the voice (and models) while the phone is ringing. This does nothing to the call.
            self._preload = asyncio.get_running_loop().create_task(self.preload())
        if ev.kind == "active" and ev.call is not None and self._task is None:
            self._call_id = ev.call.id
            self._task = asyncio.get_running_loop().create_task(self._run(ev.call.id))
        elif ev.kind in ("ended", "failed", "missed") and ev.call is not None and ev.call.id == self._call_id:
            await self.stop()

    async def stop(self) -> None:
        task, self._task, self._call_id = self._task, None, None
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def _say(self, text: str) -> None:
        if self.notify is not None:
            try:
                await self.notify(text)
            except Exception:
                log.exception("voice session notice failed")

    async def _end_call(self, call_id: int, reason: str, notice: str) -> None:
        log.warning("ending call #%s: %s", call_id, reason)
        await self._say(notice)
        await self.manager.end_by_assistant(call_id, reason)

    # ---- one call ----------------------------------------------------------------------------
    async def _run(self, call_id: int) -> None:
        from ..audio.io import DEVICE_RATE
        out = AudioOut(DEVICE_RATE)
        io = pipe = llm = None
        started = time.monotonic()
        report = {"mode": self.mode, "guard_checks": 0, "virtual_mic_confirmed": False}
        self.last_report = report
        signals: asyncio.Queue[str] = asyncio.Queue()
        call = getattr(self.manager, "calls", {}).get(call_id)
        outgoing = call is not None and call.direction is Direction.OUTGOING
        task = (call.objective or "") if outgoing else ""          # memory only; never logged
        report["outgoing"] = outgoing
        try:
            why = self.guard.violation()
            report["guard_checks"] += 1
            if why:
                return await self._end_call(call_id, "mic_guard", f"⚠️ Call #{call_id} ended for safety: {why}.")
            try:
                io = self.io_factory(out)
                io.start()
            except Exception as e:
                return await self._end_call(call_id, "audio_route_failed",
                                            f"⚠️ Call #{call_id} ended: the virtual audio path could not be opened "
                                            f"({type(e).__name__}). No other microphone was used.")
            frames = io.frames()
            if self.mode != "off":
                pipe, llm = await self._make_pipeline(
                    out, signals.put_nowait, outgoing_system_prompt(task) if outgoing else None)
                # DIGITS opens its audio some seconds after Accept (8 s on the first live call). Speak only
                # once macOS shows DIGITS has the virtual microphone open; never into an unverified path.
                deadline = time.monotonic() + self.mic_wait_s
                while not self.guard.digits_uses_virtual_mic():
                    why = self.guard.violation()
                    report["guard_checks"] += 1
                    if why:
                        return await self._end_call(call_id, "mic_guard", f"⚠️ Call #{call_id} ended for safety: {why}.")
                    if time.monotonic() >= deadline:
                        return await self._end_call(
                            call_id, "virtual_mic_unconfirmed",
                            f"⚠️ Call #{call_id} ended: DIGITS did not open the virtual microphone within "
                            f"{self.mic_wait_s:.0f} s, so nothing was spoken.")
                    await asyncio.sleep(GUARD_POLL_S)
                report["virtual_mic_confirmed"] = True
                report["mic_wait_s"] = round(time.monotonic() - started, 1)
                await asyncio.sleep(MIC_SETTLE_S if self.mic_wait_s else 0)
                if outgoing:
                    pass        # we called them: say nothing until the other person speaks
                else:
                    await pipe.say(self.greeting)
                    if self.mode == "conversation" and hasattr(pipe, "prime"):
                        asyncio.get_running_loop().create_task(pipe.prime(self.greeting))   # while the greeting plays
            feeder = None
            if self.mode == "conversation":
                feeder = asyncio.get_running_loop().create_task(pipe.run(frames))
            else:
                feeder = asyncio.get_running_loop().create_task(self._drain(frames, pipe))
            closing_at: float | None = None
            while True:
                await asyncio.sleep(GUARD_POLL_S)
                why = self.guard.violation()
                report["guard_checks"] += 1
                report["virtual_mic_confirmed"] = report["virtual_mic_confirmed"] or self.guard.digits_uses_virtual_mic()
                if why:
                    out.clear()
                    return await self._end_call(call_id, "mic_guard", f"⚠️ Call #{call_id} ended for safety: {why}.")
                if feeder.done() or not io.healthy:
                    out.clear()
                    return await self._end_call(call_id, "audio_stream_failed",
                                                f"⚠️ Call #{call_id} ended: the virtual audio stream stopped.")
                if outgoing and pipe is not None and not pipe.turns and time.monotonic() - started > self.no_answer_s:
                    return await self._end_call(call_id, "no_answer",
                                                f"Call #{call_id}: nobody spoke within {self.no_answer_s:.0f} s, so the assistant hung up.")
                while not signals.empty():
                    sig = signals.get_nowait()
                    if sig in ("idle_timeout", "error") and closing_at is None:
                        closing_at = time.monotonic() + (0.0 if sig == "idle_timeout" else 6.0)
                if closing_at is not None and time.monotonic() >= closing_at and not out.audible:
                    return await self._end_call(call_id, "assistant_ended",
                                                f"Call #{call_id}: the assistant ended the call (no one speaking, or a model problem).")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error("voice session failed: %s", type(e).__name__)      # type only: never message text
            await self._end_call(call_id, "voice_session_error", f"⚠️ Call #{call_id} ended: the voice pipeline failed.")
        finally:
            out.clear()                                    # nothing more may be spoken
            report["seconds"] = round(time.monotonic() - started, 1)
            if io is not None:
                report.update(dropped_in_frames=io.dropped_frames, stream_warnings=io.status_events,
                              audio_gaps=out.gap_events)
                io.stop()
            if pipe is not None:
                turns = pipe.turns
                report["caller_turns"] = turns
                summary = None
                if self.mode == "conversation" and llm is not None:
                    try:
                        summary = await asyncio.wait_for(
                            summarize_detailed(llm, list(pipe.history), turns, report["seconds"], task=task)
                            if self.summary == "detailed" else
                            summarize_outcome(llm, list(pipe.history), turns, report["seconds"]), timeout=30)
                    except Exception:
                        summary = None
                await pipe.close()                         # wipes text and audio buffers
                if summary:
                    await self._say(summary)
            if llm is not None:
                try:
                    await llm.close()
                except Exception:
                    pass
            log.info("voice session over: mode=%s seconds=%s guard_checks=%s", self.mode, report["seconds"], report["guard_checks"])

    async def _drain(self, frames, pipe) -> None:
        """Modes without listening: caller audio is read and discarded immediately."""
        async for f in frames:
            f.fill(0.0)
            if pipe is not None:
                pipe._housekeeping()

    async def _make_pipeline(self, out: AudioOut, on_event, system: str | None = None):
        if self.pipeline_factory is not None:
            return self.pipeline_factory(out, on_event)
        from ..speech.pipeline import ConversationPipeline, PipelineConfig
        if self._preload is not None:
            try:
                await self._preload
            except Exception:
                pass
        if self._models is None:
            self._models = await asyncio.get_running_loop().run_in_executor(None, load_models, self.mode)
        vad, stt, tts, llm_model = self._models
        from ..speech.llm import OllamaLLM
        llm = OllamaLLM(llm_model)
        cfg = PipelineConfig() if self.mode == "conversation" else PipelineConfig(idle_prompt_s=1e9, idle_hangup_s=1e9)
        if system is not None:
            # A call the assistant placed: it never hangs up for silence (holds, people checking things).
            # The other person, the owner's `hangup`, or the microphone guard ends it.
            cfg = PipelineConfig(idle_hangup_s=1e9)
        pipe = ConversationPipeline(vad=vad, stt=stt, llm=llm, tts=tts, out=out, system_prompt=system or system_prompt(),
                                    cfg=cfg, on_event=on_event)
        return pipe, llm

    async def preload(self) -> None:
        """Load models before any call so the greeting starts promptly. No-op when mode is off."""
        if self.mode == "off" or self.pipeline_factory is not None or self._models is not None:
            return
        self._models = await asyncio.get_running_loop().run_in_executor(None, load_models, self.mode)
        if self.mode == "conversation":
            from ..speech.llm import OllamaLLM
            llm = OllamaLLM(self._models[3])
            try:
                await llm.warm()
            except Exception as e:
                log.warning("could not warm the language model: %s", type(e).__name__)
            finally:
                await llm.close()


class _NoSTT:
    name = "none"
    def transcribe(self, audio):
        return ""


class _NoVAD:
    def reset(self): pass
    def prob(self, frame): return 0.0


def load_models(mode: str):
    """(vad, stt, tts, llm model name). Greeting mode loads only the voice."""
    import os
    from pathlib import Path
    from ..speech.tts import KokoroTTS
    models = Path(os.path.expanduser(os.environ.get("VOICE_MODELS_DIR", "~/.phone-assistant/models")))
    tts = KokoroTTS(models / os.environ.get("VOICE_TTS_MODEL", "kokoro-v1.0.int8.onnx"), models / "voices-v1.0.bin",
                    voice=os.environ.get("VOICE_NAME", "af_heart"))
    tts.synthesize("Ready.")
    if mode != "conversation":
        return _NoVAD(), _NoSTT(), tts, ""
    from ..speech.stt import WhisperCpp
    from ..speech.vad import SileroVAD
    stt = WhisperCpp(models / os.environ.get("VOICE_STT_MODEL", "ggml-base.en.bin"), fit_audio_ctx=False,
                     vocabulary="Adham Elkhouly.")
    return SileroVAD(models / "silero_vad.onnx"), stt, tts, os.environ.get("VOICE_LLM_MODEL", "qwen3:8b")
