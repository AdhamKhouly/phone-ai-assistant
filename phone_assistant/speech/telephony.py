"""Audio helpers: resampling, and simulating telephone-line quality for offline tests."""
from __future__ import annotations

import numpy as np
from scipy import signal


def resample(pcm: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to or len(pcm) == 0:
        return pcm.astype(np.float32, copy=False)
    g = np.gcd(sr_from, sr_to)
    return signal.resample_poly(pcm, sr_to // g, sr_from // g).astype(np.float32)


def _mulaw_roundtrip(x: np.ndarray, mu: float = 255.0) -> np.ndarray:
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    y = np.round((y + 1) / 2 * 255) / 255 * 2 - 1             # 8-bit quantisation
    return np.sign(y) * (np.power(1 + mu, np.abs(y)) - 1) / mu


def telephone(pcm: np.ndarray, sr: int, *, noise_snr_db: float | None = None, seed: int = 0) -> np.ndarray:
    """What a narrowband phone line does to speech: 300-3400 Hz band, 8 kHz sampling, G.711
    mu-law companding, optional background noise. Returns audio at the original sample rate."""
    x = pcm.astype(np.float32)
    peak = float(np.max(np.abs(x))) or 1.0
    x = x / peak * 0.7
    if noise_snr_db is not None:
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(len(x)).astype(np.float32)
        noise = signal.lfilter([1.0], [1.0, -0.85], noise)        # low-frequency-heavy, like room/road noise
        p_sig = float(np.mean(x[np.abs(x) > 0.02] ** 2)) if np.any(np.abs(x) > 0.02) else float(np.mean(x ** 2))
        noise *= np.sqrt(p_sig / (10 ** (noise_snr_db / 10)) / (float(np.mean(noise ** 2)) + 1e-12))
        x = x + noise.astype(np.float32)
    sos = signal.butter(4, [300, 3400], btype="band", fs=sr, output="sos")
    x = signal.sosfilt(sos, x)
    x8 = resample(x, sr, 8000)
    x8 = _mulaw_roundtrip(np.clip(x8, -1, 1))
    return resample(x8.astype(np.float32), 8000, sr)
