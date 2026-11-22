"""Native "DIGITS from T-Mobile" Mac app backend (Phase 2 native).

Two classes:
  * DigitsNativeObserver        OBSERVE-ONLY. Reads the app's accessibility tree twice a second and
                                reports incoming calls / connection state. It has no code path that
                                can act on a call.
  * DigitsNativeAnswerBackend   adds manual answering. Enabled only when BOTH
                                PHONE_BACKEND=digits_native_answer and ALLOW_REAL_ANSWER=yes.

Exactly two element presses exist in this file, and nowhere else in the project's native code:
  1. "Accept call" inside `answer()`, reachable only through CallManager.answer (owner-only, fresh
     command, issued after ringing started, inside the answer window, one unambiguous call);
  2. the in-call "End" button inside `_end_call()`; if it can't be found, the app is quit instead
     (which drops the call) and reopened.
No keystrokes or mouse events are ever sent.

Differences from the Chrome backend, on purpose:
  * The DIGITS app is YOUR app, not a bot-only browser profile. A connected call the assistant did
    not answer (e.g. you picked up by hand) is never touched.
  * The app outlives the bot, so a crash watchdog process is armed for the duration of every
    answered call: if the bot dies, the watchdog quits DIGITS (dropping the call) and reopens it.

What was verified against the real app (v2.5.19) on 2026-11-20 is listed in Progress Log.md.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .base import PhoneBackend
from .native_ax import iter_nodes

log = logging.getLogger(__name__)

POLL_SECONDS = 0.5
UNKNOWN_GRACE_S = 10.0       # unreadable / unrecognised this long -> treat as disconnected
RELAUNCH_EVERY_S = 60.0
REOPEN_EVERY_S = 900.0       # last resort for an unreadable window: ask DIGITS to show it (it may flash in front)
HIDE_GRACE_S = 8.0           # how long a quiet hide gets to make the window readable before the last resort
SETTLE_S = 10.0              # after starting DIGITS: wait this long for it to become readable, then hide it
WINDOW_MISSING_S = 3.0
CONFIRM_TIMEOUT_S = 4.0
DIAL_CONFIRM_S = 12.0        # how long DIGITS gets to show its call screen after a tel: request
ACCEPT_LABEL = "Accept call"
REJECT_LABEL = "Reject call"
LOGGED_IN_IDS = {"setting-menu", "calls", "messages", "voicemails"}
END_RE = re.compile(r"^(end( call)?|hang ?up|end the call)$", re.I)
LOGIN_RE = re.compile(r"\b(log|sign) ?in\b", re.I)
DEGRADED_TEXT = "trouble with connectivity"
_NUMBER_RE = re.compile(r"(\+?1?[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})")
BUTTON_ROLES = {"AXButton", "AXLink"}
SIGNATURE_ROLES = {"AXButton", "AXLink", "AXCheckBox", "AXRadioButton"}
SIGNATURE_SETTLE_S = 1.0     # after answering, wait this long before learning the in-call screen
ENDED_POLLS = 2              # in-call screen must be gone for this many readings in a row


# =====================================================================================
# Pure functions: accessibility tree -> snapshot. Unit-tested with captured structures.
# =====================================================================================
@dataclass(frozen=True)
class Alert:
    number: str            # normalized, or "Unknown"
    name: str | None
    key: str               # everything the alert displayed; must be unchanged at answer time
    accept: dict           # the Accept button node (pressed only by answer())


def _label(n: dict) -> str:
    return (n.get("desc") or n.get("title") or "").strip()


def _texts(node: dict) -> list[str]:
    return [n["value"].strip() for n in iter_nodes(node) if n.get("role") == "AXStaticText" and n.get("value", "").strip()]


def _by_id(node: dict, ident: str) -> dict | None:
    return next((n for n in iter_nodes(node) if n.get("id") == ident), None)


def _buttons(node: dict, label: str) -> list[dict]:
    return [n for n in iter_nodes(node) if n.get("role") == "AXButton" and _label(n) == label]


def _alert_container(node: dict) -> dict | None:
    """Smallest subtree holding exactly one Accept and one Reject button."""
    if len(_buttons(node, ACCEPT_LABEL)) != 1 or not _buttons(node, REJECT_LABEL):
        return None
    for c in node.get("children", ()):
        inner = _alert_container(c)
        if inner is not None:
            return inner
    return node


def _caller(container: dict) -> tuple[str, str | None]:
    from ..control import normalize_number  # local import avoids a cycle

    name_el, num_el = _by_id(container, "displayName"), _by_id(container, "originator")
    shown_name = " ".join(_texts(name_el)) if name_el else ""
    shown_num = " ".join(_texts(num_el)) if num_el else ""
    if not name_el and not num_el:           # unverified layout: fall back to all visible text
        shown_num = " ".join(_texts(container))
    m = _NUMBER_RE.search(shown_num) or _NUMBER_RE.search(shown_name)
    number = (normalize_number(m.group(1)) or m.group(1).strip()) if m else None
    name = shown_name.strip() or None
    if name and (normalize_number(name) is not None or _NUMBER_RE.fullmatch(name)):
        name = None                           # DIGITS shows the number twice for unknown callers
    if number is None:
        number, name = (name or "Unknown"), None
    return number, name


def find_alerts(windows: list[dict]) -> list[Alert]:
    alerts = []
    for w in windows:
        n_accept = len(_buttons(w, ACCEPT_LABEL))
        if n_accept == 0:
            continue
        box = _alert_container(w)
        if box is None:                       # several alerts in one window, or no Reject button
            for b in _buttons(w, ACCEPT_LABEL):
                alerts.append(Alert("Unknown", None, "ambiguous", b))
            continue
        number, name = _caller(box)
        alerts.append(Alert(number, name, " | ".join(_texts(box)), _buttons(box, ACCEPT_LABEL)[0]))
    return alerts


def find_end_button(windows: list[dict]) -> dict | None:
    for w in windows:
        for n in iter_nodes(w):
            if n.get("role") == "AXButton" and END_RE.match(_label(n)):
                return n
    return None


def ui_signature(windows: list[dict]) -> frozenset[str]:
    """Window titles and control labels currently on screen (no text content). Controls that
    appear when a call connects and vanish when it ends identify the in-call screen."""
    sig = set()
    for w in windows:
        if w.get("title"):
            sig.add("window:" + w["title"])
        for n in iter_nodes(w):
            if n.get("role") in SIGNATURE_ROLES and _label(n):
                sig.add("control:" + _label(n))
    return frozenset(sig)


def snapshot(windows: list[dict] | None) -> dict:
    """state: not_running | signed_out | online | unknown.  Never raises on odd trees."""
    if windows is None:
        return {"state": "not_running", "alerts": [], "inCall": False, "degraded": False, "sig": frozenset()}
    alerts = find_alerts(windows)
    in_call = find_end_button(windows) is not None
    marker = login = degraded = False
    for w in windows:
        for n in iter_nodes(w):
            if n.get("id") in LOGGED_IN_IDS:
                marker = True
            elif n.get("role") in BUTTON_ROLES and LOGIN_RE.search(_label(n) + " " + " ".join(_texts(n))):
                login = True
            elif n.get("role") == "AXStaticText" and DEGRADED_TEXT in n.get("value", ""):
                degraded = True
    if marker or alerts or in_call:
        state = "online"
    elif login:
        state = "signed_out"
    else:
        state = "unknown"      # no window, still loading, or a view we don't recognise
    return {"state": state, "alerts": alerts, "inCall": in_call, "degraded": degraded, "sig": ui_signature(windows)}


# =====================================================================================
# Observe-only backend
# =====================================================================================
class DigitsNativeObserver(PhoneBackend):
    name = "digits native app (observe-only)"
    can_answer = False
    can_dial = False

    def __init__(self, source=None, *, keep_awake: bool = True, autolaunch: bool = True) -> None:
        super().__init__()
        if source is None:
            from .native_ax import MacAXSource
            source = MacAXSource()
        self.source = source
        self.keep_awake = keep_awake
        self.autolaunch = autolaunch
        self.unknown_grace_s = UNKNOWN_GRACE_S
        self.app_state = "unknown"
        self.degraded = False
        self._current_call_id: int | None = None
        self._current: Alert | None = None
        self._unknown_since: float | None = None
        self._status_reported = False
        self._last_launch = 0.0
        self._last_hide = 0.0
        self._last_reopen = 0.0
        self._settle: tuple[float, int | None] | None = None   # (deadline, app that was in front)
        self._front_other: int | None = None                   # last app in front that wasn't DIGITS
        self._digits_front_when_idle = False
        self._was_busy = False
        self._stop = asyncio.Event()
        self._caffeinate: subprocess.Popen | None = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="digits-ax")

    # ---- commands: all refused in observe-only mode -------------------------------
    async def answer(self, call_id: int) -> bool:
        return False

    async def hangup(self, call_id: int) -> bool:
        return False

    async def dial(self, call_id: int, number: str) -> bool:
        return False

    # ---- reading ---------------------------------------------------------------------
    async def _ax(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self._pool, fn, *args)

    async def _read(self) -> dict:
        try:
            return snapshot(await self._ax(self.source.read))
        except Exception as e:  # accessibility hiccup: unknown, never "ringing"
            log.warning("DIGITS app unreadable: %s", type(e).__name__)
            return {"state": "unknown", "alerts": [], "inCall": False, "degraded": False, "sig": frozenset()}

    def status_note(self) -> str:
        notes = {"online": "signed in", "signed_out": "SIGNED OUT, open the DIGITS app and log in",
                 "not_running": "NOT RUNNING", "unknown": "window not readable"}
        note = "DIGITS app: " + notes.get(self.app_state, self.app_state)
        if self.degraded and self.app_state == "online":
            note += " (the app shows its 'trouble with connectivity' banner)"
        return note

    # ---- state handling (pure; unit-tested with fake trees) ------------------------------
    async def _set_connected(self, connected: bool) -> None:
        if connected == self.connected:
            if not connected and not self._status_reported:
                self._status_reported = True      # started offline: say so once
                await self.sink.on_backend_status(False)
            return
        self.connected = connected
        self._status_reported = True
        if not connected:
            self._current_call_id, self._current = None, None   # manager releases ringing calls
        await self.sink.on_backend_status(connected)

    def _hold_state_while_unknown(self) -> bool:
        return False

    async def apply_snapshot(self, snap: dict) -> None:
        assert self.sink is not None
        self.app_state, self.degraded = snap["state"], snap["degraded"]
        now = time.monotonic()
        if snap["state"] in ("not_running", "signed_out"):
            self._unknown_since = None
            await self._set_connected(False)
            return
        if snap["state"] == "unknown":
            if self._unknown_since is None:
                self._unknown_since = now
            if now - self._unknown_since >= self.unknown_grace_s and not self._hold_state_while_unknown():
                await self._set_connected(False)
            return
        self._unknown_since = None
        await self._set_connected(True)
        alerts: list[Alert] = snap["alerts"]
        first = alerts[0] if alerts else None
        if self._current_call_id is not None and (first is None or first.number != self._current.number):
            # Stopped ringing, or a different caller replaced it: close the old one first.
            cid, self._current_call_id, self._current = self._current_call_id, None, None
            await self.sink.on_remote_ended(cid, "stopped_ringing")
        if first is not None and self._current_call_id is None:
            call = await self.sink.on_incoming(first.number, first.name)
            self._current_call_id, self._current = call.id, first

    # ---- main loop -------------------------------------------------------------------
    async def run(self) -> None:
        self._start_keep_awake()
        if hasattr(self.source, "trusted") and not await self._ax(self.source.trusted):
            log.error("macOS Accessibility permission is missing: System Settings > Privacy & Security > "
                      "Accessibility > enable the app that runs this bot. Until then DIGITS can't be read.")
        last_diag = None
        try:
            while not self._stop.is_set():
                snap = await self._read()
                diag = (snap["state"], len(snap["alerts"]), snap["inCall"], snap["degraded"])
                if diag != last_diag:   # state-change diagnostics only; never caller details
                    log.info("digits native: state=%s alerts=%s inCall=%s connectivityBanner=%s", *diag)
                    last_diag = diag
                await self.apply_snapshot(snap)
                await self._stay_in_background(snap)
                if self.autolaunch and snap["state"] == "not_running":
                    await self._maybe_launch()
                elif self.autolaunch and self._settle is None and self._window_missing():
                    await self._repair_window()
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=POLL_SECONDS)
                except asyncio.TimeoutError:
                    pass
        finally:
            self._stop_keep_awake()
            if self.connected and self.sink is not None:
                self.connected = False
                await self.sink.on_backend_status(False)

    def _window_missing(self) -> bool:
        """Running but unreadable for a few seconds, e.g. its window was closed (verified: the app
        keeps running without a window, and opening it again brings the window back)."""
        return (self.app_state == "unknown" and self._unknown_since is not None
                and time.monotonic() - self._unknown_since >= WINDOW_MISSING_S)

    async def _maybe_launch(self) -> None:
        """Start DIGITS if it isn't running. Starting the app never answers or places a call.
        DIGITS shows itself when it starts; `_stay_in_background` hides it once it is readable."""
        now = time.monotonic()
        if self._last_launch and now - self._last_launch < RELAUNCH_EVERY_S:
            return
        self._last_launch = now
        log.info("DIGITS app is %s; starting it in the background", self.app_state.replace("_", " "))
        self._settle = (now + SETTLE_S, self._front_other)
        self._last_reopen = self._last_hide = 0.0     # a fresh start may need its window shown once more
        await self._ax(self.source.launch)

    async def _repair_window(self) -> None:
        """DIGITS is running but its window can't be read (typically: you closed the window).
        Hiding the app makes the window readable again without showing anything or taking focus.
        Only if that doesn't work is DIGITS asked to show its window, and at most every 15 minutes."""
        src, now = self.source, time.monotonic()
        if not hasattr(src, "hide"):
            return await self._maybe_launch()
        if await self._ax(src.is_frontmost):
            return      # you are using DIGITS: hands off
        hidden = await self._ax(src.is_hidden)
        if not hidden and getattr(src, "last_window_count", 0) > 0:
            return      # a visible window showing something we don't recognise: hands off
        if not hidden and (not self._last_hide or now - self._last_hide >= RELAUNCH_EVERY_S):
            self._last_hide = now
            log.info("DIGITS window is closed; hiding the app so it can be read in the background")
            await self._ax(src.hide)
        elif (hidden or now - self._last_hide >= HIDE_GRACE_S) and (not self._last_reopen or now - self._last_reopen >= REOPEN_EVERY_S):
            self._last_reopen = now
            log.info("DIGITS still unreadable; asking it to show its window once (it will be hidden again)")
            self._settle = (now + SETTLE_S, self._front_other)
            await self._ax(src.launch)

    async def _send_to_background(self, give_focus_to: int | None) -> None:
        """Hide DIGITS. If it had pushed itself in front, give focus back to the app that had it.
        One hide and at most one activation: never a loop."""
        src = self.source
        if give_focus_to and await self._ax(src.is_frontmost):
            await self._ax(src.activate_pid, give_focus_to)      # first, so no third app flashes in between
            await asyncio.sleep(0.15)
        await self._ax(src.hide)

    async def _stay_in_background(self, snap: dict) -> None:
        src = self.source
        if not hasattr(src, "frontmost_pid"):
            return
        try:
            front, digits_front = await self._ax(src.frontmost_pid), await self._ax(src.is_frontmost)
            if front is not None and not digits_front:
                self._front_other = front
            busy = bool(snap["alerts"]) or bool(snap["inCall"]) or self._owns_call()
            if self._settle is not None:
                deadline, prev = self._settle
                if snap["state"] == "online":
                    self._settle = None
                    if not busy:
                        log.info("DIGITS is readable again; hiding it")
                        await self._send_to_background(prev)
                elif snap["state"] == "signed_out" or time.monotonic() > deadline:
                    self._settle = None          # needs you (sign-in), or never became readable: leave it visible
            elif self._was_busy and not busy and snap["state"] == "online":
                # A ring or an assistant-handled call just ended. If DIGITS put itself in front for it
                # (it was not in front before), send it back and return focus.
                if digits_front and not self._digits_front_when_idle:
                    log.info("call over; sending DIGITS back to the background")
                    await self._send_to_background(self._front_other)
            if not busy:
                self._digits_front_when_idle = digits_front and not self._was_busy
            elif not self._was_busy:
                log.info("ring started: DIGITS in front=%s (was in front before=%s)", digits_front, self._digits_front_when_idle)
            self._was_busy = busy
        except Exception as e:
            log.warning("background handling skipped: %s", type(e).__name__)

    def _owns_call(self) -> bool:
        return False

    def stop(self) -> None:
        self._stop.set()

    # ---- keep the Mac awake while observing (lid-close still sleeps) --------------------
    def _start_keep_awake(self) -> None:
        if not self.keep_awake:
            return
        try:
            self._caffeinate = subprocess.Popen(["/usr/bin/caffeinate", "-is", "-w", str(os.getpid())])
        except OSError:
            log.warning("caffeinate unavailable; the Mac may sleep and miss calls")

    def _stop_keep_awake(self) -> None:
        if self._caffeinate and self._caffeinate.poll() is None:
            self._caffeinate.terminate()


# =====================================================================================
# Manual-answer backend (both safety switches required; see config.py)
# =====================================================================================
def default_watchdog_cmd(max_seconds: float) -> list[str]:
    return [sys.executable, "-m", "phone_assistant.phone.native_watchdog",
            "--parent", str(os.getpid()), "--max-seconds", str(int(max_seconds))]


class DigitsNativeAnswerBackend(DigitsNativeObserver):
    name = "digits native app (manual answer)"
    can_answer = True
    can_dial = False   # becomes True only with allow_dial (ALLOW_OUTGOING=yes)
    active_on_dial = True   # dial() returns once the call screen is up; the AI then waits for the person

    def __init__(self, source=None, *, watchdog_cmd: list[str] | None = None, audio_route=None,
                 allow_dial: bool = False, **kwargs) -> None:
        super().__init__(source, **kwargs)
        self.can_dial = bool(allow_dial)
        self.watchdog_cmd = watchdog_cmd
        # Optional virtual-audio route (audio/routing.CallAudioRoute). When present it MUST engage
        # before Accept is pressed; if it can't, the call is not answered. Released when the call ends.
        self.audio_route = audio_route
        self.last_refusal = ""
        self._watchdog: subprocess.Popen | None = None
        self._answering = False
        self._in_call_id: int | None = None
        self._seen_in_call = False
        self._warned_foreign_call = False
        self._baseline_sig: frozenset[str] = frozenset()   # what was on screen just before Accept
        self._call_sig: frozenset[str] | None = None       # controls that appeared with the call
        self._answered_at = 0.0
        self._gone_polls = 0
        self._caller_words: tuple[str, ...] = ()

    # ---- crash watchdog: armed only while the assistant owns a call --------------------------
    def _arm_watchdog(self) -> None:
        if self.watchdog_cmd is None or self._watchdog is not None:
            return
        cmd = list(self.watchdog_cmd)
        pid = self.source.pid() if hasattr(self.source, "pid") else None
        if pid:
            # The watchdog acts only on THIS run of the DIGITS app. If the app has been restarted
            # since, the assistant's call is gone and anything now in the app is not ours.
            cmd += ["--digits-pid", str(pid)]
        try:
            self._watchdog = subprocess.Popen(cmd, start_new_session=True)
        except OSError as e:
            log.error("could not start the crash watchdog: %s", type(e).__name__)

    def _disarm_watchdog(self) -> None:
        if self._watchdog is not None:
            if self._watchdog.poll() is None:
                self._watchdog.terminate()
            self._watchdog = None

    def _clear_call(self) -> None:
        self._in_call_id = None
        self._seen_in_call = False
        self._call_sig, self._baseline_sig, self._gone_polls, self._caller_words = None, frozenset(), 0, ()
        self._disarm_watchdog()
        self._release_audio()

    def _release_audio(self) -> None:
        if self.audio_route is not None:
            try:
                self.audio_route.release()
            except Exception as e:
                log.error("could not restore audio devices: %s", type(e).__name__)

    def _call_visible(self, snap: dict) -> bool:
        """Is the call the assistant answered still on screen?"""
        return bool(snap["inCall"] or (self._call_sig and self._call_sig & snap["sig"]))

    def _learn_call_screen(self, snap: dict) -> None:
        self._call_sig = snap["sig"] - self._baseline_sig
        if not snap["inCall"]:
            # Log the new control names once so the End button can be confirmed. Digits and the
            # caller's name are stripped: logs never hold caller details.
            safe = []
            for label in sorted(self._call_sig):
                for word in self._caller_words:
                    label = re.sub(re.escape(word), "<caller>", label, flags=re.I)
                safe.append(re.sub(r"\d", "#", label)[:50])
            log.warning("in-call screen has no known End button; new controls: %s", safe[:25])

    def _hold_state_while_unknown(self) -> bool:
        return self._in_call_id is not None   # an in-call view we can't read is not "disconnected"

    def _owns_call(self) -> bool:
        return self._in_call_id is not None or self._answering

    # ---- state ---------------------------------------------------------------------------
    async def apply_snapshot(self, snap: dict) -> None:
        if self._answering:
            return  # answer() owns the state while it confirms
        if self._in_call_id is not None and snap["state"] in ("online", "unknown"):
            self.app_state, self.degraded = snap["state"], snap["degraded"]
            if snap["state"] != "online":
                return
            if self._call_sig is None and time.monotonic() - self._answered_at >= SIGNATURE_SETTLE_S:
                self._learn_call_screen(snap)
            if self._call_visible(snap):
                self._seen_in_call, self._gone_polls = True, 0
            elif self._seen_in_call and not snap["alerts"]:
                self._gone_polls += 1
                if self._gone_polls >= ENDED_POLLS:
                    # The assistant's call is over: release it and stand the watchdog down at once,
                    # so nothing can later be done to a different call.
                    cid = self._in_call_id
                    self._clear_call()
                    await self.sink.on_remote_ended(cid, "remote_hangup")
            return  # a screen that never changed at all: rely on `hangup` / max call duration
        if snap["inCall"] and self._in_call_id is None and not self._warned_foreign_call:
            # You (not the assistant) are on a call in the DIGITS app. Never touch it.
            self._warned_foreign_call = True
            log.info("a call is connected in DIGITS that the assistant did not answer; leaving it alone")
        if not snap["inCall"]:
            self._warned_foreign_call = False
        await super().apply_snapshot(snap)
        if self._in_call_id is not None and not self.connected:
            self._clear_call()  # app quit / signed out: the manager ends the active call

    # ---- answer: verify, then one press ----------------------------------------------------
    async def answer(self, call_id: int) -> bool:
        if call_id != self._current_call_id or self._current is None or self._in_call_id is not None:
            return False
        expected = self._current
        snap = await self._read()
        if snap["state"] != "online" or len(snap["alerts"]) != 1:
            log.warning("answer refused: call #%s is no longer the single ringing call", call_id)
            return False
        now_alert: Alert = snap["alerts"][0]
        if now_alert.number != expected.number or now_alert.key != expected.key:
            log.warning("answer refused: ringing caller changed for call #%s", call_id)
            return False
        if snap["inCall"]:
            log.warning("answer refused: DIGITS already shows a connected call")
            return False

        if self.audio_route is not None:
            try:
                why = self.audio_route.engage()
            except Exception as e:
                why = f"audio routing error ({type(e).__name__})"
            if why:
                # No safe audio path: do not answer. The call keeps ringing on the phone as usual.
                self.last_refusal = why
                log.warning("answer refused: virtual audio is not available (%s)", why)
                self._release_audio()
                return False
        self.last_refusal = ""

        self._answering = True
        try:
            self._arm_watchdog()
            await self._ax(self.source.press, now_alert.accept)   # the one Accept press; never repeated
            log.info("pressed Accept for call #%s", call_id)
            deadline = asyncio.get_running_loop().time() + CONFIRM_TIMEOUT_S
            while asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.25)
                s = await self._read()
                if s["state"] == "not_running":
                    break
                if s["state"] == "online" and (not s["alerts"] or s["inCall"]):
                    self._in_call_id = call_id
                    self._seen_in_call = bool(s["inCall"])
                    self._baseline_sig, self._call_sig, self._gone_polls = snap["sig"], None, 0
                    self._answered_at = time.monotonic()
                    self._caller_words = tuple(w for w in (expected.name or "").split() if len(w) > 1)
                    self._current_call_id, self._current = None, None
                    return True
            log.warning("call #%s: not confirmed %ss after Accept; treating as not answered", call_id, CONFIRM_TIMEOUT_S)
            self._disarm_watchdog()
            self._release_audio()
            return False
        except Exception as e:
            log.warning("answer for call #%s failed safely: %s", call_id, type(e).__name__)
            self._disarm_watchdog()
            self._release_audio()
            return False
        finally:
            self._answering = False

    # ---- dial: one tel: link, only from CallManager.dial (owner's confirmed `call` command) -----
    async def dial(self, call_id: int, number: str) -> bool:
        self.last_refusal = ""
        if not self.can_dial:
            return False
        if self._owns_call():
            self.last_refusal = "the assistant is already on a call"
            return False
        snap = await self._read()
        if snap["state"] != "online" or snap["alerts"] or snap["inCall"]:
            self.last_refusal = "DIGITS is not idle (signed out, ringing, or already on a call)"
            return False
        if self.audio_route is None:
            self.last_refusal = "no virtual audio route is configured"
            return False
        try:
            why = self.audio_route.engage()
        except Exception as e:
            why = f"audio routing error ({type(e).__name__})"
        if why:
            self.last_refusal = why
            log.warning("dial refused: virtual audio is not available (%s)", why)
            self._release_audio()
            return False
        self._answering = True
        try:
            self._arm_watchdog()
            if not await self._ax(self.source.open_tel, number):      # the one request; never repeated
                self.last_refusal = "DIGITS did not accept the number"
                self._disarm_watchdog()
                self._release_audio()
                return False
            log.info("asked DIGITS to place call #%s", call_id)
            deadline = asyncio.get_running_loop().time() + DIAL_CONFIRM_S
            while asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.25)
                s = await self._read()
                if s["state"] == "not_running":
                    break
                if s["state"] == "online" and s["inCall"]:
                    self._in_call_id = call_id
                    self._seen_in_call = True
                    self._baseline_sig, self._call_sig, self._gone_polls = snap["sig"], None, 0
                    self._answered_at = time.monotonic()
                    self._caller_words = ()
                    return True
            # No call screen appeared. We cannot tell what DIGITS is showing, so make sure nothing is
            # left dialling: restart the app (drops anything in progress), then put the audio back.
            log.warning("call #%s: DIGITS showed no call screen %ss after the request; restarting it to be safe",
                        call_id, DIAL_CONFIRM_S)
            self.last_refusal = "DIGITS did not start the call; it was restarted to be safe"
            await self._quit_and_reopen()
            self._disarm_watchdog()
            self._release_audio()
            return False
        except Exception as e:
            log.warning("dial for call #%s failed safely: %s", call_id, type(e).__name__)
            self.last_refusal = "an error occurred; DIGITS was restarted to be safe"
            try:
                await self._quit_and_reopen()
            except Exception:
                pass
            self._disarm_watchdog()
            self._release_audio()
            return False
        finally:
            self._answering = False

    # ---- hang up: End button, else quit the app ----------------------------------------------
    async def hangup(self, call_id: int) -> bool:
        if self._in_call_id is None or call_id != self._in_call_id:
            return False    # never end a call the assistant doesn't own
        return await self._end_call()

    async def _end_call(self) -> bool:
        ended = False
        try:
            end = find_end_button(await self._ax(self.source.read) or [])
            if end is not None:
                await self._ax(self.source.press, end)
                for _ in range(12):
                    await asyncio.sleep(0.25)
                    s = await self._read()
                    if s["state"] != "unknown" and not s["inCall"]:
                        ended = True
                        break
        except Exception as e:
            log.warning("End button failed: %s", type(e).__name__)
        if not ended and self._seen_in_call:
            s = await self._read()
            if s["state"] == "online" and not self._call_visible(s):
                # The assistant's call was on screen earlier and is gone now: it already ended.
                # Quitting the app here could only hurt a different call, so don't.
                log.info("call already over; DIGITS left running")
                ended = True
        if not ended:
            log.warning("ending the call by quitting the DIGITS app (fallback), then reopening it")
            ended = await self._quit_and_reopen()
        self._clear_call()
        return ended

    async def _quit_and_reopen(self) -> bool:
        """Quitting the app is what drops the call; reopening it is best-effort afterwards."""
        try:
            quit_ok = bool(await self._ax(self.source.quit))
        except Exception as e:
            log.error("could not quit the DIGITS app: %s", type(e).__name__)
            return False
        if not quit_ok:
            log.error("the DIGITS app is still running; the call may still be connected")
            return False
        try:
            self._last_launch = time.monotonic()
            self._settle = (self._last_launch + SETTLE_S, self._front_other)   # hide it again once readable
            await self._ax(self.source.launch)
        except Exception as e:
            log.error("DIGITS quit (call dropped) but reopening failed: %s", type(e).__name__)
        return True
