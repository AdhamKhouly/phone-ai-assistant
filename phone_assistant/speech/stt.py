"""Local speech recognition engines. All take a numpy buffer and return text; none touch disk."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .interfaces import SAMPLE_RATE

# Whisper invents these on silence or noise; they are not caller speech.
_NON_SPEECH = re.compile(r"^\s*([\[(][^\])]*[\])]\s*)*$")


def clean(text: str) -> str:
    text = re.sub(r"[\[(][^\])]*[\])]", " ", text)      # [BLANK_AUDIO], (coughing) ...
    text = re.sub(r"\s+", " ", text).strip()
    return "" if _NON_SPEECH.match(text) or not re.search(r"[A-Za-z0-9]", text) else text


class WhisperCpp:
    """whisper.cpp through pywhispercpp. `fit_audio_ctx` shrinks the encoder window to the length
    of the utterance. It is OFF by default: on this Mac it saved ~20 ms but quadrupled word errors
    on telephone-quality audio and caused occasional 1-2 s decoding loops (see Audio Benchmark Report)."""

    def __init__(self, model_path: str | Path, *, threads: int = 4, fit_audio_ctx: bool = False,
                 vocabulary: str = "") -> None:
        from pywhispercpp.model import Model

        self.name = f"whisper.cpp {Path(model_path).stem.replace('ggml-', '')}" + (" (fitted ctx)" if fit_audio_ctx else "")
        self.fit_audio_ctx = fit_audio_ctx
        self._m = Model(str(model_path), n_threads=threads, print_realtime=False, print_progress=False,
                        print_timestamps=False, language="en", no_context=True, single_segment=True,
                        suppress_blank=True, redirect_whispercpp_logs_to=None,
                        **({"initial_prompt": vocabulary} if vocabulary else {}))   # names to expect, e.g. "Adham"

    def transcribe(self, audio: np.ndarray) -> str:
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if len(audio) < SAMPLE_RATE:                      # whisper.cpp wants at least 1 s
            audio = np.concatenate([audio, np.zeros(SAMPLE_RATE - len(audio) + 1600, dtype=np.float32)])
        params = {}
        if self.fit_audio_ctx:
            # 50 encoder positions per second of audio, plus margin; 1500 = the full 30 s.
            params["audio_ctx"] = int(min(1500, max(128, len(audio) / SAMPLE_RATE * 50 + 64)))
        segs = self._m.transcribe(audio, **params)
        return clean(" ".join(s.text for s in segs))


class MlxWhisper:
    """Whisper on the Apple GPU through MLX. Shares the GPU with the language model."""

    def __init__(self, repo: str) -> None:
        import mlx_whisper

        self._w, self.repo = mlx_whisper, repo
        self.name = "mlx-whisper " + repo.split("/")[-1].replace("whisper-", "").replace("-mlx", "")

    def transcribe(self, audio: np.ndarray) -> str:
        r = self._w.transcribe(np.ascontiguousarray(audio, dtype=np.float32), path_or_hf_repo=self.repo,
                               language="en", temperature=0.0, condition_on_previous_text=False, verbose=None)
        return clean(r.get("text", ""))


class Moonshine:
    """Moonshine (Useful Sensors) ONNX: small models whose cost scales with audio length."""

    def __init__(self, model: str = "moonshine/base") -> None:
        import moonshine_onnx

        self.name = model.replace("/", " ")
        self._model = moonshine_onnx.MoonshineOnnxModel(model_name=model)
        self._tok = moonshine_onnx.load_tokenizer()

    def transcribe(self, audio: np.ndarray) -> str:
        tokens = self._model.generate(np.ascontiguousarray(audio, dtype=np.float32)[None, :])
        return clean(self._tok.decode_batch(tokens)[0])
