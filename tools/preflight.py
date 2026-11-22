"""Read-only pre-call checklist. Changes nothing, presses nothing, opens no audio device.
    .venv/bin/python tools/preflight.py"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import dotenv_values                                             # noqa: E402

from phone_assistant.audio import coreaudio as ca                            # noqa: E402
from phone_assistant.audio.routing import RESTORE_FILE, RouteConfig, check_devices  # noqa: E402
from phone_assistant.phone.digits_native import snapshot                     # noqa: E402
from phone_assistant.phone.native_ax import MacAXSource                      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
rows: list[tuple[str, bool | None, str]] = []


def row(name: str, ok: bool | None, detail: str = "") -> None:
    rows.append((name, ok, detail))


env = dotenv_values(ROOT / ".env")
backend = env.get("PHONE_BACKEND", "simulated")
answering = backend == "digits_native_answer" and (env.get("ALLOW_REAL_ANSWER") or "").lower() == "yes"
voice = (env.get("VOICE_MODE") or "off").lower()
row("Real answering", None, "ENABLED (answers only on the owner's typed `answer`)" if answering else "disabled (observe-only)")
row("Voice mode", None, voice + (" + ALLOW_AI_VOICE" if (env.get("ALLOW_AI_VOICE") or "").lower() == "yes" else ""))
row("AI conversation mode", None, "ON (approved by the owner 2026-11-21)" if voice == "conversation" else "off")
_lim = float(env.get("MAX_CALL_SECONDS") or 60)
row("Call time limit", None, "none (ends on hang-up, or after about 30 s of silence)" if _lim <= 0 else f"{_lim:.0f} s")

bot = subprocess.run(["pgrep", "-f", "m phone_assistant"], capture_output=True, text=True).stdout.split()
row("Bot process running", len(bot) == 1, f"{len(bot)} process(es)")

src = MacAXSource()
row("Accessibility permission", src.trusted())
snap = snapshot(src.read())
row("DIGITS signed in and readable", snap["state"] == "online", snap["state"])
row("DIGITS is in the background", not src.is_frontmost(), "hidden" if src.is_hidden() else "visible, not in front" if not src.is_frontmost() else "IN FRONT")
row("No call ringing or connected now", not snap["alerts"] and not snap["inCall"])
if snap["degraded"]:
    row("DIGITS connectivity banner", None, "shown (it was also shown when real calls were received)")

cfg = RouteConfig(mic=env.get("AUDIO_MIC_DEVICE") or "BlackHole 2ch", capture=env.get("AUDIO_CAPTURE_DEVICE") or "BlackHole 16ch")
problems = check_devices(cfg)
row("Both virtual audio devices available", not problems, "; ".join(problems))
di, do = ca.default_input(), ca.default_output()
row("Mac's own audio devices are the defaults", bool(di and do and not di.is_virtual and not do.is_virtual),
    f"input={di.name if di else None}, output={do.name if do else None}")
row("No leftover audio switch from a crash", not RESTORE_FILE.exists())
mics = [d.name for d in ca.devices() if d.is_physical_input and d.running]
row("No physical microphone in use right now", not mics, ", ".join(mics))
digits_audio = [p for p in ca.audio_for_bundle("com.tmobile.phone2") if p.running_input or p.running_output]
row("DIGITS has no audio device open while idle", not digits_audio)
models = Path(os.path.expanduser("~/.phone-assistant/models"))
row("Voice model files present", all((models / f).exists() for f in ("kokoro-v1.0.int8.onnx", "voices-v1.0.bin")))

bad = 0
for name, ok, detail in rows:
    mark = "  " if ok is None else "OK" if ok else "!!"
    bad += ok is False
    print(f"[{mark}] {name}" + (f": {detail}" if detail else ""))
print("\nAll checks passed." if not bad else f"\n{bad} check(s) need attention.")
sys.exit(1 if bad else 0)
