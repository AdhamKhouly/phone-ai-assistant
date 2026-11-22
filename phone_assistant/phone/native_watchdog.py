"""Crash watchdog for calls answered through the native DIGITS app.

The DIGITS app keeps running if the bot crashes, so a call the assistant answered could stay
connected with nobody in control. The bot starts this tiny process right before pressing Accept
and stops it (SIGTERM) as soon as the call is over. If instead the bot process disappears, or the
call outlives the hard time limit, the watchdog quits DIGITS (which drops the call) and reopens it.

It never answers, dials, or presses anything.

It cannot act on a call the assistant did not answer:
  * it exists only between the Accept press and the moment the bot sees that call end;
  * it does nothing unless the bot died or the hard time limit passed;
  * it does nothing if the DIGITS process the call was answered in is no longer the one running.

    python -m phone_assistant.phone.native_watchdog --parent <bot pid> --max-seconds <n>
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time

MARGIN_S = 15.0   # the bot's own max-duration hang-up gets to act first


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def drop_call_by_quitting_digits(expected_pid: int | None = None) -> None:
    from .native_ax import MacAXSource

    src = MacAXSource()
    if expected_pid is not None and src.pid() != expected_pid:
        return      # DIGITS was restarted in the meantime: not the app instance holding our call
    front = src.frontmost_pid()
    if front == src.pid():
        front = None
    src.quit()
    time.sleep(1.0)
    src.launch()
    # DIGITS shows itself when it starts. Once it is readable, hide it and hand focus back.
    from .digits_native import snapshot
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        time.sleep(0.4)
        try:
            state = snapshot(src.read())["state"]
        except Exception:
            continue
        if state == "signed_out":
            return
        if state == "online":
            if front and src.is_frontmost():
                src.activate_pid(front)
                time.sleep(0.15)
            src.hide()
            return


def _restore_audio() -> None:
    """Put the default audio devices back if a call left them on the virtual devices."""
    try:
        from ..audio.routing import restore_stale
        restore_stale()
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", type=int, required=True)
    ap.add_argument("--max-seconds", type=float, required=True)
    ap.add_argument("--digits-pid", type=int, default=None,
                    help="act only if this DIGITS process is still running (the one the call was answered in)")
    ap.add_argument("--command", help="test hook: JSON argv to run instead of quitting DIGITS")
    ap.add_argument("--poll", type=float, default=0.5)
    args = ap.parse_args(argv)

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))   # call ended normally: do nothing
    # 0 = no time limit (owner's choice for real use): the watchdog then acts only if the bot dies.
    deadline = float("inf") if args.max_seconds == 0 else time.monotonic() + args.max_seconds + MARGIN_S
    # When the bot started us directly, a changed parent pid means it died (works even while the
    # dead bot is still a zombie, which os.kill(pid, 0) would report as alive).
    is_child = os.getppid() == args.parent
    while time.monotonic() < deadline:
        if not _alive(args.parent) or (is_child and os.getppid() != args.parent):
            break
        time.sleep(args.poll)
    if not args.command:
        _restore_audio()   # the bot can no longer do it; harmless if nothing was switched
    if args.digits_pid is not None and not _alive(args.digits_pid):
        return 0    # that run of DIGITS is gone, and the assistant's call with it: nothing to do
    if args.command:
        subprocess.run(json.loads(args.command), timeout=30)
    else:
        drop_call_by_quitting_digits(args.digits_pid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
