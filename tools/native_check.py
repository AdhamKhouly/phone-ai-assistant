"""Read-only health check of the native DIGITS app, using the same code the bot uses.
Never presses anything and never prints caller details.

    .venv/bin/python tools/native_check.py            # one reading
    .venv/bin/python tools/native_check.py --watch 60 # print state changes for 60 s (e.g. during a test call)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phone_assistant.phone.digits_native import snapshot  # noqa: E402
from phone_assistant.phone.native_ax import MacAXSource  # noqa: E402

src = MacAXSource()
print("Accessibility permission:", "granted" if src.trusted() else "MISSING (System Settings > Privacy & Security > Accessibility)")
watch = float(sys.argv[sys.argv.index("--watch") + 1]) if "--watch" in sys.argv else 0
end, last = time.time() + watch, None
while True:
    t = time.time()
    wins = src.read()
    snap = snapshot(wins)
    ms = (time.time() - t) * 1000
    alerts = snap["alerts"]
    line = (f"app pid={src.pid()} state={snap['state']} ringing={len(alerts)} "
            f"caller_id_readable={[bool(a.number != 'Unknown') for a in alerts]} "
            f"caller_has_name={[a.name is not None for a in alerts]} "
            f"accept_button_found={[a.accept.get('desc') == 'Accept call' for a in alerts]} "
            f"in_call={snap['inCall']} connectivity_banner={snap['degraded']} windows={0 if wins is None else len(wins)}")
    if line != last:
        print(time.strftime("%H:%M:%S"), line, f"(read {ms:.0f} ms)")
        last = line
    if time.time() >= end:
        break
    time.sleep(0.5)
