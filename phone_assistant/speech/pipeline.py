"""Real-time conversation pipeline: caller audio in, assistant audio out. Everything in memory.

    frames (16 kHz) -> VAD -> utterance buffer -> STT -> LLM (streamed) -> phrase chunks -> TTS -> AudioOut

How it stays fast without guessing:
  * As soon as the caller pauses briefly, speech recognition starts on what they have said so far.
  * When that text arrives, the language model and the first speech chunk are prepared but HELD.
  * If the caller keeps talking, that work is thrown away. If they stay quiet until the end-of-turn
    threshold, the held audio is released, so the wait the caller hears is mostly the threshold.
  * Nothing is ever spoken for a turn that was not confirmed.

How it stays correct:
  * Every response has a generation number. AudioOut only plays the released generation, and every
    worker checks its generation before handing anything on, so stale work cannot be heard.
  * If the caller talks over the assistant, the assistant's audio is faded out and paused. A pure
    acknowledgement ("okay", "mm-hmm") resumes it; anything else cancels it and gets a new reply.

Privacy: caller and assistant text exist only in this object's memory and are cleared by close().
Events record timings and counts only, never words.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import AsyncIterator, Callable

import numpy as np

from .audio_out import AudioOut
from .chunker import PhraseChunker
from .interfaces import FRAME, SAMPLE_RATE
from .telephony import resample
from .turns import is_backchannel, looks_complete

log = logging.getLogger(__name__)

FRAME_S = FRAME / SAMPLE_RATE
FALLBACK_LINE = "Sorry, I'm having a technical problem. Please text or email Adham instead. Goodbye."
STILL_THERE_LINE = "Are you still there?"
REPEAT_LINE = "Sorry, could you say that again?"


@dataclass
class PipelineConfig:
    speech_start_prob: float = 0.5
    speech_end_prob: float = 0.35
    speech_start_frames: int = 3           # ~96 ms of speech before we believe it
    preroll_s: float = 0.35                # audio kept from just before speech was detected
    early_stt_silence_s: float = 0.25      # pause length that starts recognition
    endpoint_silence_s: float = 0.6        # pause that ends a turn that sounds finished
    endpoint_incomplete_s: float = 1.4     # ...and one that trails off ("I wanted to ask about the")
    max_utterance_s: float = 30.0
    bargein_prob: float = 0.7              # stricter while the assistant is speaking
    bargein_speech_s: float = 0.3          # caller speech needed before the assistant yields
    speculative: bool = True               # prepare the reply during the end-of-turn wait
    max_pending_chunks: int = 6
    max_history_messages: int = 40         # trimmed in blocks (see _trim_history), not every turn
    chunk_pause_s: float = 0.12            # silence appended after each spoken sentence
    llm_first_token_timeout_s: float = 5.0 # a model that says nothing for this long counts as failed
    llm_stall_timeout_s: float = 8.0       # ...and so does one that stops mid-reply
    idle_prompt_s: float = 9.0             # quiet this long after the assistant finishes -> "still there?"
    idle_hangup_s: float = 20.0            # still quiet -> ask the session to end the call


class _Response:
    def __init__(self, gen: int, user_text: str | None) -> None:
        self.gen = gen
        self.user_text = user_text
        self.task: asyncio.Task | None = None
        self.cancelled = False
        self.released = False
        self.paused = False
        self.done = False
        self.failed = False
        self.chunks: list[str] = []


class _Speculation:
    def __init__(self) -> None:
        self.text: str | None = None
        self.started_at = 0.0
        self.stt_seconds = 0.0
        self.resp: _Response | None = None


class ConversationPipeline:
    def __init__(self, *, vad, stt, llm, tts, out: AudioOut, system_prompt: str,
                 cfg: PipelineConfig | None = None, clock: Callable[[], float] = time.monotonic,
                 on_event: Callable[[str], None] | None = None) -> None:
        self.vad, self.stt, self.llm, self.tts, self.out = vad, stt, llm, tts, out
        self.cfg = cfg or PipelineConfig()
        self.clock = clock
        self.on_event = on_event                    # session hooks: "idle_timeout", "error"
        self.system_prompt = system_prompt
        self.history: list[dict] = []
        self.events: deque[tuple[float, str, int]] = deque(maxlen=4000)   # (time, name, generation); no content
        self.debug_text: Callable[[str, str], None] | None = None         # offline benchmarks only
        self._gen = 0
        self._resp: _Response | None = None
        self._spec: _Speculation | None = None
        self._in_speech = False
        self._run = 0
        self._silence = 0.0
        self._voiced = 0.0
        self._utt: list[np.ndarray] = []
        self._preroll: deque[np.ndarray] = deque(maxlen=max(1, int(self.cfg.preroll_s / FRAME_S)))
        self._stt_pool = ThreadPoolExecutor(1, thread_name_prefix="stt")
        self._tts_pool = ThreadPoolExecutor(1, thread_name_prefix="tts")
        self._quiet_since: float | None = None
        self._prompted = False
        self._closed = False
        self.turns = 0

    # ---- bookkeeping ---------------------------------------------------------------------
    def _mark(self, name: str, gen: int = -1) -> None:
        self.events.append((self.clock(), name, gen))

    def _emit(self, name: str) -> None:
        if self.on_event is not None:
            try:
                self.on_event(name)
            except Exception:
                log.exception("pipeline event handler failed")

    @property
    def assistant_active(self) -> bool:
        return self._resp is not None

    # ---- assistant speech ------------------------------------------------------------------
    def _synth(self, text: str) -> np.ndarray:
        pcm = resample(self.tts.synthesize(text), self.tts.sample_rate, self.out.sample_rate)
        pad = np.zeros(int(self.cfg.chunk_pause_s * self.out.sample_rate), dtype=np.float32)
        return np.concatenate([pcm, pad])

    def _messages(self, user_text: str) -> list[dict]:
        return [{"role": "system", "content": self.system_prompt}] + self.history + [{"role": "user", "content": user_text}]

    def _trim_history(self) -> None:
        """Keep the context small, but drop old turns in one block rather than one per turn: the
        model server reuses its work for an unchanged beginning of the conversation, and a window
        that slides every turn would throw that away every turn (measured: ~1 s per reply)."""
        if len(self.history) > self.cfg.max_history_messages:
            keep = self.cfg.max_history_messages // 2
            for m in self.history[:-keep]:
                m["content"] = ""
            del self.history[:-keep]

    async def prime(self, opening: str | None = None) -> None:
        """Have the model read the rules and the greeting now (while the greeting plays), so the
        caller's first question doesn't wait for that. Generates one throwaway token."""
        if not hasattr(self.llm, "stream"):
            return
        try:
            msgs = self._messages("Hello?")
            if opening and not self.history:        # same shape the first real turn will have
                msgs.insert(1, {"role": "assistant", "content": opening})
            async for _ in self.llm.stream(msgs, num_predict=1):
                pass
        except Exception as e:
            log.warning("could not prime the language model: %s", type(e).__name__)

    def _start_response(self, user_text: str | None, *, fixed_text: str | None = None, held: bool) -> _Response:
        self._gen += 1
        r = _Response(self._gen, user_text)
        r.task = asyncio.get_running_loop().create_task(self._generate(r, fixed_text))
        if not held:
            self._release(r)
        return r

    def _release(self, r: _Response) -> None:
        r.released = True
        self._resp = r
        self.out.release(r.gen)
        self._quiet_since = None
        self._mark("released", r.gen)

    async def _generate(self, r: _Response, fixed_text: str | None) -> None:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=self.cfg.max_pending_chunks)

        async def produce() -> None:
            try:
                if fixed_text is not None:
                    fixed = PhraseChunker(first_min_words=10 ** 6)
                    for ch in fixed.feed(fixed_text + " ") + fixed.flush():
                        await q.put(ch)
                    return
                chunker = PhraseChunker()
                first = True
                produced = 0
                self._mark("llm_start", r.gen)
                stream = self.llm.stream(self._messages(r.user_text or "")).__aiter__()
                while True:
                    limit = self.cfg.llm_first_token_timeout_s if first else self.cfg.llm_stall_timeout_s
                    try:
                        piece = await asyncio.wait_for(stream.__anext__(), timeout=limit)
                    except StopAsyncIteration:
                        break
                    if r.cancelled:
                        return
                    if first:
                        self._mark("llm_first_token", r.gen)
                        first = False
                    for ch in chunker.feed(piece):
                        produced += 1
                        await q.put(ch)
                for ch in chunker.flush():
                    produced += 1
                    await q.put(ch)
                if not produced and not r.cancelled:
                    # The model answered with nothing speakable. Never leave the caller in silence.
                    self._mark("empty_reply", r.gen)
                    await q.put(REPEAT_LINE)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error("language model failed: %s", type(e).__name__)
                r.failed = True
                if not r.chunks:
                    await q.put(FALLBACK_LINE)
            finally:
                await q.put(None)

        prod = loop.create_task(produce())
        try:
            index = 0
            while True:
                ch = await q.get()
                if ch is None or r.cancelled:
                    break
                if index == 0:
                    self._mark("first_phrase", r.gen)
                pcm = await loop.run_in_executor(self._tts_pool, self._synth, ch)
                if r.cancelled:
                    break
                r.chunks.append(ch)
                self.out.submit(r.gen, index, pcm)
                self._mark("tts_ready", r.gen)
                index += 1
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error("speech synthesis failed: %s", type(e).__name__)
            r.failed = True
            self._emit("error")
        finally:
            prod.cancel()
            self.out.finish(r.gen)
            r.done = True
            if r.failed and fixed_text is None and r.user_text is not None:
                self._emit("degraded")

    def _cancel_response(self, r: _Response) -> None:
        r.cancelled = True
        if r.task is not None:
            r.task.cancel()
        self.out.cancel(r.gen)

    def _finalize_response(self, r: _Response, interrupted: bool) -> None:
        """Record what the caller actually heard, then forget the response object."""
        heard = r.chunks[: self.out.last_index_started.get(r.gen, -1) + 1]
        if heard:
            self.history.append({"role": "assistant", "content": " ".join(heard) + (" —" if interrupted else "")})
        if self.debug_text and heard:
            self.debug_text("assistant", " ".join(heard))
        if self._resp is r:
            self._resp = None
        self._mark("interrupted" if interrupted else "response_end", r.gen)
        if not interrupted:
            self._quiet_since = self.clock()

    async def say(self, text: str) -> None:
        """Speak fixed text (the greeting). Still interruptible like any other speech."""
        if self._resp is not None:
            self._cancel_response(self._resp)
            self._finalize_response(self._resp, interrupted=True)
        self._start_response(None, fixed_text=text, held=False)

    # ---- caller speech ---------------------------------------------------------------------
    def _start_speculation(self) -> None:
        spec = _Speculation()
        spec.started_at = self.clock()
        self._spec = spec
        audio = np.concatenate(self._utt)
        fut = asyncio.get_running_loop().run_in_executor(self._stt_pool, self.stt.transcribe, audio)
        self._mark("stt_start")

        def done(f: asyncio.Future) -> None:
            if self._spec is not spec or self._closed:
                return                                   # the caller kept talking: result is stale
            try:
                spec.text = f.result()
            except Exception as e:
                log.error("speech recognition failed: %s", type(e).__name__)
                spec.text = ""
                self._emit("degraded")
            spec.stt_seconds = self.clock() - spec.started_at
            self._mark("stt_done")
            if not spec.text or not self.cfg.speculative:
                return
            old = self._resp
            if old is not None and is_backchannel(spec.text):
                return                                   # the assistant will simply carry on
            spec.resp = self._start_response(spec.text, held=True)

        fut.add_done_callback(done)

    def _cancel_speculation(self) -> None:
        if self._spec is not None and self._spec.resp is not None:
            self._cancel_response(self._spec.resp)
        self._spec = None
        self._mark("speculation_cancelled")

    def _end_utterance(self) -> None:
        for f in self._utt:
            f.fill(0.0)
        self._utt = []
        self._in_speech = False
        self._run = 0
        self._silence = self._voiced = 0.0
        self._spec = None

    def _commit(self) -> None:
        spec = self._spec
        assert spec is not None and spec.text is not None
        text = spec.text
        self._end_utterance()
        old = self._resp
        if not text:                                     # a cough or line noise
            if old is not None and old.paused:
                self.out.resume()
                old.paused = False
            return
        if old is not None and is_backchannel(text):
            if spec.resp is not None:
                self._cancel_response(spec.resp)
            if old.paused:
                self.out.resume()
                old.paused = False
            self._mark("backchannel", old.gen)
            return
        if old is not None:                              # a real interruption
            self._cancel_response(old)
            self._finalize_response(old, interrupted=True)
        if self.debug_text:
            self.debug_text("user", text)
        resp = spec.resp or self._start_response(text, held=True)
        self._trim_history()
        self.history.append({"role": "user", "content": text})
        self.turns += 1
        self._prompted = False
        self._mark("commit", resp.gen)
        self._release(resp)

    def feed(self, frame: np.ndarray) -> None:
        """Process one 32 ms frame of caller audio. Call from the event loop thread."""
        if self._closed:
            return
        c = self.cfg
        p = self.vad.prob(frame)
        self._housekeeping()
        if not self._in_speech:
            self._preroll.append(frame)
            threshold = c.bargein_prob if self.out.audible else c.speech_start_prob
            self._run = self._run + 1 if p >= threshold else 0
            if self._run >= c.speech_start_frames:
                self._in_speech = True
                self._utt = [f.copy() for f in self._preroll]
                self._silence, self._voiced = 0.0, self._run * FRAME_S
                self._quiet_since = None
                self._mark("speech_start")
            return
        self._utt.append(frame.copy())
        if p >= c.speech_end_prob:
            if self._spec is not None:
                self._cancel_speculation()               # they weren't finished
            self._silence = 0.0
            self._voiced += FRAME_S
        else:
            self._silence += FRAME_S
        r = self._resp
        if r is not None and r.released and not r.paused and self.out.audible and self._voiced >= c.bargein_speech_s:
            self.out.pause()                             # yield the floor; decide what it was afterwards
            r.paused = True
            self._mark("bargein", r.gen)
        too_long = len(self._utt) * FRAME_S >= c.max_utterance_s
        if self._spec is None and (self._silence >= c.early_stt_silence_s or too_long):
            self._start_speculation()
        spec = self._spec
        if spec is not None and spec.text is not None:
            needed = c.endpoint_silence_s if looks_complete(spec.text) else c.endpoint_incomplete_s
            if self._silence >= needed or too_long:
                self._commit()

    def _housekeeping(self) -> None:
        r = self._resp
        if r is not None and r.done and not r.paused and self.out.idle(r.gen):
            self._finalize_response(r, interrupted=False)
            if r.failed:
                self._emit("idle_timeout")               # said the fallback line: end the call
        if self._resp is None and not self._in_speech and self._quiet_since is not None:
            quiet = self.clock() - self._quiet_since
            if not self._prompted and quiet >= self.cfg.idle_prompt_s:
                self._prompted = True
                self._start_response(None, fixed_text=STILL_THERE_LINE, held=False)
            elif self._prompted and quiet >= self.cfg.idle_hangup_s:
                self._quiet_since = None
                self._emit("idle_timeout")

    async def run(self, frames: AsyncIterator[np.ndarray]) -> None:
        async for frame in frames:
            self.feed(frame)
            await asyncio.sleep(0)

    # ---- teardown: nothing about the conversation survives this ---------------------------------
    async def close(self) -> None:
        self._closed = True
        for r in (self._resp, self._spec.resp if self._spec else None):
            if r is not None:
                self._cancel_response(r)
        tasks = [r.task for r in (self._resp, self._spec.resp if self._spec else None) if r is not None and r.task]
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self.out.clear()
        self._end_utterance()
        self._preroll.clear()
        for m in self.history:
            m["content"] = ""
        self.history.clear()
        self._resp = None
        self._stt_pool.shutdown(wait=False, cancel_futures=True)
        self._tts_pool.shutdown(wait=False, cancel_futures=True)
        if hasattr(self.vad, "reset"):
            self.vad.reset()
