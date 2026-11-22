"""Local text-to-speech engines. Text in, PCM array out; nothing is written to disk."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class KokoroTTS:
    def __init__(self, model_path: str | Path, voices_path: str | Path, *, voice: str = "af_heart",
                 speed: float = 1.0) -> None:
        from kokoro_onnx import Kokoro

        self._k = Kokoro(str(model_path), str(voices_path))
        self.voice, self.speed, self.sample_rate = voice, speed, 24000
        self.name = f"Kokoro {'int8' if 'int8' in Path(model_path).name else 'fp32'} ({voice})"

    def synthesize(self, text: str) -> np.ndarray:
        pcm, sr = self._k.create(text, voice=self.voice, speed=self.speed, lang="en-us")
        assert sr == self.sample_rate
        return np.asarray(pcm, dtype=np.float32)


class PiperTTS:
    def __init__(self, model_path: str | Path) -> None:
        from piper import PiperVoice

        self._v = PiperVoice.load(str(model_path))
        self.sample_rate = int(self._v.config.sample_rate)
        self.name = "Piper " + Path(model_path).stem

    def synthesize(self, text: str) -> np.ndarray:
        parts = [c.audio_float_array for c in self._v.synthesize(text)]
        return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, dtype=np.float32)
