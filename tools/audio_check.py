"""Read-only check of the call audio setup, plus an optional signal test.

    .venv/bin/python tools/audio_check.py              # what is installed, what is default, what DIGITS has open
    .venv/bin/python tools/audio_check.py --signal     # also send a test tone through the virtual devices
    .venv/bin/python tools/audio_check.py --restore    # put default devices back after a crash

The signal test only opens the two VIRTUAL devices. It does not change any default device, does
not open a microphone, plays nothing through the speakers, and saves nothing.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phone_assistant.audio import coreaudio as ca            # noqa: E402
from phone_assistant.audio.routing import DIGITS_BUNDLE, MicGuard, RouteConfig, check_devices, restore_stale  # noqa: E402
from phone_assistant.config import load_settings             # noqa: E402


def cfg() -> RouteConfig:
    try:
        s = load_settings()
        return RouteConfig(mic=s.audio_mic_device, capture=s.audio_capture_device)
    except Exception:
        return RouteConfig()


async def tone_through(play_name: str, listen_names: list[str], hz: float) -> dict[str, tuple[float, float]]:
    """Play a tone into `play_name`; report (level, dominant Hz) heard on each of `listen_names`."""
    import sounddevice as sd
    from phone_assistant.audio.io import DEVICE_RATE, virtual_device_index
    t = np.arange(int(DEVICE_RATE * 1.5)) / DEVICE_RATE
    tone = (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)
    got = {n: [] for n in listen_names}
    streams = []
    for n in listen_names:
        idx = virtual_device_index(n, want_input=True)
        streams.append(sd.InputStream(device=idx, samplerate=DEVICE_RATE, channels=1, dtype="float32",
                                      callback=lambda d, f, ti, st, n=n: got[n].append(d[:, 0].copy())))
    out_idx = virtual_device_index(play_name, want_input=False)
    pos = [0]

    def feed(outdata, frames, ti, st):
        chunk = tone[pos[0]:pos[0] + frames]
        outdata[:, :] = 0.0
        outdata[:len(chunk), 0] = chunk
        pos[0] += frames
    player = sd.OutputStream(device=out_idx, samplerate=DEVICE_RATE, channels=1, dtype="float32", callback=feed)
    for st_ in streams:
        st_.start()
    player.start()
    await asyncio.sleep(1.8)
    player.stop(); player.close()
    for st_ in streams:
        st_.stop(); st_.close()
    res = {}
    for n, blocks in got.items():
        x = np.concatenate(blocks) if blocks else np.zeros(1, dtype=np.float32)
        sp = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        res[n] = (float(np.sqrt(np.mean(x ** 2))), float(np.fft.rfftfreq(len(x), 1 / DEVICE_RATE)[int(sp.argmax())]))
        x.fill(0)
    return res


def main() -> int:
    if "--restore" in sys.argv:
        print("restored" if restore_stale() else "nothing to restore")
        return 0
    c = cfg()
    print("Audio devices:")
    for d in ca.devices():
        print(f"  {'virtual ' if d.is_virtual else 'PHYSICAL'} {d.name!r:32} in={d.has_input!s:5} out={d.has_output!s:5} in use={d.running}")
    di, do = ca.default_input(), ca.default_output()
    print(f"Default input : {di.name if di else None}\nDefault output: {do.name if do else None}")
    problems = check_devices(c)
    print(f"\nVirtual route ({c.mic!r} = AI voice -> DIGITS mic, {c.capture!r} = caller -> AI):")
    print("  OK, both devices present" if not problems else "\n".join("  PROBLEM: " + p for p in problems))
    procs = ca.audio_for_bundle(DIGITS_BUNDLE)
    print("\nDIGITS audio right now:", "no audio open" if not any(p.running_input or p.running_output for p in procs) else "")
    for p in procs:
        if p.running_input or p.running_output:
            print(f"  pid {p.pid}: input={p.input_devices if p.running_input else '-'} output={p.output_devices if p.running_output else '-'}")
    print("Physical microphones in use by any app:", [d.name for d in ca.devices() if d.is_physical_input and d.running] or "none")
    print("Guard verdict if a call were up now:", MicGuard(c).violation() or "ok")
    if "--signal" in sys.argv:
        if problems:
            print("\nSignal test skipped: fix the problems above first.")
            return 1
        before = (di.uid if di else None, do.uid if do else None)
        ok = True
        for play, hz in ((c.mic, 440.0), (c.capture, 880.0)):
            other = c.capture if play == c.mic else c.mic
            r = asyncio.run(tone_through(play, [play, other], hz))
            same, cross = r[play], r[other]
            good = same[0] > 0.05 and abs(same[1] - hz) < 5 and cross[0] < 0.001
            ok = ok and good
            print(f"\n  {hz:.0f} Hz tone into {play!r}: heard on {play!r} level {same[0]:.3f} at {same[1]:.0f} Hz; "
                  f"leak into {other!r} level {cross[0]:.5f}  ->  {'PASS' if good else 'FAIL'}")
        mic_used = [d.name for d in ca.devices() if d.is_physical_input and d.running]
        after = (ca.default_input().uid if ca.default_input() else None, ca.default_output().uid if ca.default_output() else None)
        print(f"\n  physical microphones in use during the test: {mic_used or 'none'}")
        print(f"  default devices unchanged: {before == after}")
        print("\nSIGNAL TEST", "PASSED" if ok and not mic_used and before == after else "FAILED")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
