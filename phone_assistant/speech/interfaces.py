"""Speech interfaces. Contract for every implementation: audio and text live in memory only.
Nothing is written to disk, logged, or sent off this Mac."""
from __future__ import annotations

from typing import AsyncIterator, Protocol

import numpy as np

SAMPLE_RATE = 16000      # pipeline rate for incoming audio (mono float32)
FRAME = 512              # 32 ms; the frame size Silero VAD needs at 16 kHz


class VoiceActivityDetector(Protocol):
    def reset(self) -> None: ...
    def prob(self, frame: np.ndarray) -> float:
        """Speech probability for one FRAME-sample block."""


class SpeechToText(Protocol):
    name: str
    def transcribe(self, audio: np.ndarray) -> str:
        """16 kHz mono float32 in, text out. Blocking; called from a worker thread."""


class TextToSpeech(Protocol):
    name: str
    sample_rate: int
    def synthesize(self, text: str) -> np.ndarray:
        """Text in, mono float32 PCM at `sample_rate` out. Blocking; worker thread."""


class LanguageModel(Protocol):
    name: str
    def stream(self, messages: list[dict], num_predict: int | None = None) -> AsyncIterator[str]:
        """Yield response text pieces as they are generated."""
