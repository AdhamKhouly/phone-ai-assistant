"""Voice activity detection: Silero VAD v5 (ONNX, CPU, ~2 MB). Stateless about content."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .interfaces import FRAME, SAMPLE_RATE

CONTEXT = 64


class SileroVAD:
    def __init__(self, model_path: str | Path) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = opts.intra_op_num_threads = 1
        self._s = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._ctx = np.zeros(CONTEXT, dtype=np.float32)

    def prob(self, frame: np.ndarray) -> float:
        assert frame.shape == (FRAME,)
        x = np.concatenate([self._ctx, frame.astype(np.float32)])[None, :]
        out, self._state = self._s.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._ctx = x[0, -CONTEXT:]
        return float(out[0, 0])
