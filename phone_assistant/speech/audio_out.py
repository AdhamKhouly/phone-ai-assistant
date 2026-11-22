"""Outgoing (AI -> caller) audio buffer.

Pull-based: whoever owns the playback clock (the virtual-device callback on a real call, or the
offline harness) calls `pull(n)` for the next n samples. That makes the same code path testable
without any audio hardware.

Every piece of audio carries the generation number of the response it belongs to. Only the
currently released generation is ever played, so audio from an earlier turn can never come out
during a later one.
"""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np


class AudioOut:
    def __init__(self, sample_rate: int, fade_ms: float = 25.0, clock=time.monotonic) -> None:
        self.sample_rate = sample_rate
        self._fade = max(1, int(sample_rate * fade_ms / 1000))
        self._clock = clock
        self._lock = threading.Lock()
        self._segs: deque[list] = deque()        # [gen, index, pcm, position]
        self._active: int | None = None          # generation allowed to play
        self._finished: set[int] = set()         # generations with no more audio coming
        self._paused = False
        self._fade_out = self._fade_in = False
        self._drop_after_fade = False
        # measurements (no content)
        self.first_audio_at: dict[int, float] = {}
        self.last_index_started: dict[int, int] = {}
        self.gap_events = 0
        self.gap_seconds = 0.0
        self._in_gap = False
        self._started: set[int] = set()

    # ---- producer side ---------------------------------------------------------------
    def submit(self, gen: int, index: int, pcm: np.ndarray) -> None:
        with self._lock:
            self._segs.append([gen, index, np.asarray(pcm, dtype=np.float32), 0])

    def finish(self, gen: int) -> None:
        with self._lock:
            self._finished.add(gen)

    def release(self, gen: int) -> None:
        """Allow this generation to be heard. Anything queued for other generations is dropped."""
        with self._lock:
            self._active = gen
            self._segs = deque(s for s in self._segs if s[0] == gen)
            self._paused = self._fade_out = self._drop_after_fade = False
            self._in_gap = False

    def cancel(self, gen: int | None = None) -> None:
        """Stop and discard. With `gen`, only that generation (e.g. a speculative one)."""
        with self._lock:
            if gen is not None and gen != self._active:
                self._segs = deque(s for s in self._segs if s[0] != gen)
                return
            if self._active is not None and not self._paused and self._has_audio():
                self._fade_out = self._drop_after_fade = True     # short fade instead of a click
            else:
                self._drop_active()

    def pause(self) -> None:
        with self._lock:
            if not self._paused:
                self._fade_out = True
                self._paused = True

    def resume(self) -> None:
        with self._lock:
            if self._paused:
                self._paused, self._fade_out, self._fade_in = False, False, True

    # ---- state -------------------------------------------------------------------------
    def _has_audio(self) -> bool:
        return any(s[0] == self._active for s in self._segs)

    def _drop_active(self) -> None:
        self._segs = deque(s for s in self._segs if s[0] != self._active)
        self._active = None
        self._paused = self._fade_out = self._drop_after_fade = False

    @property
    def audible(self) -> bool:
        """True while released audio is playing or about to (not paused)."""
        with self._lock:
            return self._active is not None and not self._paused and self._has_audio()

    def idle(self, gen: int) -> bool:
        """This generation has played everything it will ever produce."""
        with self._lock:
            return gen in self._finished and not any(s[0] == gen for s in self._segs)

    def pending_seconds(self) -> float:
        with self._lock:
            return sum(len(s[2]) - s[3] for s in self._segs if s[0] == self._active) / self.sample_rate

    # ---- consumer side: the playback clock ----------------------------------------------
    def pull(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        with self._lock:
            if self._active is None:
                return out
            if self._paused and not self._fade_out:
                return out
            want = min(n, self._fade) if self._fade_out else n
            filled = 0
            while filled < want and self._segs:
                seg = self._segs[0]
                if seg[0] != self._active:
                    self._segs.popleft()
                    continue
                gen, index, pcm, pos = seg
                if gen not in self._started:
                    self._started.add(gen)
                    self.first_audio_at[gen] = self._clock()
                self.last_index_started[gen] = index
                take = min(want - filled, len(pcm) - pos)
                out[filled:filled + take] = pcm[pos:pos + take]
                seg[3] += take
                filled += take
                if seg[3] >= len(pcm):
                    self._segs.popleft()
            if self._fade_out:
                out[:want] *= np.linspace(1.0, 0.0, want, dtype=np.float32)
                self._fade_out = False
                if self._drop_after_fade:
                    self._drop_active()
                return out
            if self._fade_in and filled:
                k = min(filled, self._fade)
                out[:k] *= np.linspace(0.0, 1.0, k, dtype=np.float32)
                self._fade_in = False
            # A gap = the response has started, more audio is still coming, but none is ready.
            starving = filled < n and self._active in self._started and self._active not in self._finished
            if starving:
                self.gap_seconds += (n - filled) / self.sample_rate
                if not self._in_gap:
                    self.gap_events += 1
                self._in_gap = True
            elif filled:
                self._in_gap = False
        return out

    def clear(self) -> None:
        with self._lock:
            for s in self._segs:
                s[2].fill(0.0)
            self._segs.clear()
            self._active = None
            self._finished.clear()
            self._started.clear()
            self._paused = self._fade_out = self._fade_in = self._drop_after_fade = False
