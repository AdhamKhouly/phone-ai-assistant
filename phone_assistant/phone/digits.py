"""T-Mobile DIGITS web-client backend: OBSERVE-ONLY (Phase 2a).

What it does:
  * Opens https://digits.t-mobile.com in a dedicated Chrome profile (you log in yourself;
    this code never sees or stores credentials).
  * Every 0.5 s runs one READ-ONLY DOM query that reports:
      - whether the web client is logged in,
      - whether the incoming-call alert is showing, and the caller name/number it displays.
  * Reports those as events to CallManager -> Discord #calls.

What it never does (by construction, not by a flag):
  * No clicks, no keystrokes (pressing Enter on the ringing alert may ACCEPT the call),
    no network interception, no page-script modification, no audio capture.
  * `answer`, `hangup`, `dial` are refused. Answering arrives in Phase 2b, as a separate change.

Selectors were verified against a real ringing call on 2026-11-20 (see Progress Log).
"""
from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .base import PhoneBackend

log = logging.getLogger(__name__)

DIGITS_URL = "https://digits.t-mobile.com/"
DEFAULT_PROFILE_DIR = Path.home() / ".phone-assistant" / "digits-chrome-profile"
POLL_SECONDS = 0.5
UNAVAILABLE_GRACE_S = 10.0   # page unreadable this long -> treat as disconnected
ALERT_SEL = '[role="alert"][data-testid="name-contact"]'
ACCEPT_SEL = 'button[aria-label="Accept call"]'

# One read-only DOM query. Returns plain data; touches nothing.
SNAPSHOT_JS = r"""
() => {
  const alerts = [...document.querySelectorAll('[role="alert"][data-testid="name-contact"]')];
  const alert = alerts[0];
  const accept = alert && alert.querySelector('button[aria-label="Accept call"]');
  const text = s => (s || '').replace(/\s+/g, ' ').trim();
  const loggedOut = [...document.querySelectorAll('button,a')].some(b => /^log ?in\b/i.test(text(b.innerText)));
  // In-call view: an "end call / hang up" control (exact label confirmed during the first live answer test).
  const inCall = [...document.querySelectorAll('button,[role=button]')].some(b =>
    /^(end( call)?|hang ?up|end the call)$/i.test(text(b.getAttribute('aria-label') || b.title || b.innerText)));
  const loggedIn = alerts.length > 0 || inCall
    || [...document.querySelectorAll('button,a,div,span')].some(e => e.children.length === 0 && /^(all phone numbers|new call|new message)$/i.test(text(e.innerText)));
  return {
    loggedIn: loggedIn && !loggedOut,
    loggedOut: loggedOut,
    ringing: !!(alert && accept),
    alertCount: alerts.length,
    inCall: inCall,
    label: alert ? text(alert.getAttribute('aria-label')) : null,
    body: alert ? text(alert.innerText).slice(0, 200) : null,
  };
}
"""

_NUMBER_RE = re.compile(r"(\+?1?[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})")
_FROM_RE = re.compile(r"incoming (?:audio |video )?call from (.+?)(?:\.\s|\.$|$)", re.I)


@dataclass(frozen=True)
class Caller:
    number: str        # as displayed, normalized to digits/+ where possible
    name: str | None   # contact name if DIGITS shows one


def parse_caller(label: str | None, body: str | None) -> Caller:
    """Extract caller name/number from the alert's aria-label and visible text."""
    from ..control import normalize_number  # local import avoids a cycle

    number = None
    m = _NUMBER_RE.search(body or "") or _NUMBER_RE.search(label or "")
    if m:
        number = normalize_number(m.group(1)) or m.group(1).strip()
    name = None
    f = _FROM_RE.search(label or "")
    if f:
        candidate = f.group(1).strip()
        if not _NUMBER_RE.fullmatch(candidate) and normalize_number(candidate) is None:
            name = candidate
    if number is None:
        # Unknown / withheld caller: keep whatever DIGITS shows, or "Unknown".
        number = name or "Unknown"
        name = None if number == name else name
    return Caller(number=number, name=name)


class DigitsObserver(PhoneBackend):
    name = "digits (observe-only)"
    can_answer = False
    can_dial = False

    def __init__(self, profile_dir: Path = DEFAULT_PROFILE_DIR, keep_awake: bool = True) -> None:
        super().__init__()
        self.profile_dir = profile_dir
        self.keep_awake = keep_awake
        self._current_call_id: int | None = None
        self._current_caller: str | None = None
        self._unavailable_since: float | None = None
        self._status_reported = False
        self.unavailable_grace_s = UNAVAILABLE_GRACE_S
        self.page = None   # the live DIGITS page (read by subclasses)
        self.ctx = None
        self._stop = asyncio.Event()
        self._caffeinate: subprocess.Popen | None = None

    # ---- commands: all refused in observe-only mode -------------------------------
    async def answer(self, call_id: int) -> bool:
        return False

    async def hangup(self, call_id: int) -> bool:
        return False

    async def dial(self, call_id: int, number: str) -> bool:
        return False

    # ---- state handling (pure; unit-tested with fake snapshots) ----------------------
    async def _set_connected(self, connected: bool) -> None:
        if connected == self.connected:
            return
        self.connected = connected
        self._status_reported = True
        if not connected:
            self._current_call_id = None   # manager releases ringing calls on disconnect
            self._current_caller = None
        await self.sink.on_backend_status(connected)

    async def apply_snapshot(self, snap: dict | None) -> None:
        """Turn one page snapshot into CallManager events.

        `None` = page unreadable (navigating, crashed). Only a definite logged-out page, or an
        unreadable page for UNAVAILABLE_GRACE_S, counts as disconnected; an unrecognised view
        (e.g. an in-call screen) keeps the previous state.
        """
        assert self.sink is not None
        now = time.monotonic()
        if snap is None:
            if self._unavailable_since is None:
                self._unavailable_since = now
            if now - self._unavailable_since >= self.unavailable_grace_s:
                await self._set_connected(False)
            return
        self._unavailable_since = None
        if snap.get("loggedOut"):
            if not self.connected and not self._status_reported:
                # Started (or stayed) signed out: warn once even though nothing "changed".
                self._status_reported = True
                await self.sink.on_backend_status(False)
            await self._set_connected(False)
            return
        if snap.get("loggedIn"):
            await self._set_connected(True)
        if not self.connected:
            return
        ringing = bool(snap.get("ringing"))
        caller = parse_caller(snap.get("label"), snap.get("body")) if ringing else None
        if self._current_call_id is not None and (not ringing or caller.number != self._current_caller):
            # Stopped ringing, or a different caller replaced it: close the old one first.
            cid, self._current_call_id, self._current_caller = self._current_call_id, None, None
            await self.sink.on_remote_ended(cid, "stopped_ringing")
        if ringing and self._current_call_id is None:
            call = await self.sink.on_incoming(caller.number, caller.name)
            self._current_call_id, self._current_caller = call.id, caller.number

    # ---- browser loop ------------------------------------------------------------------
    async def run(self) -> None:
        from playwright.async_api import async_playwright

        self._start_keep_awake()
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        try:
            async with async_playwright() as pw:
                ctx = await pw.chromium.launch_persistent_context(
                    str(self.profile_dir), channel="chrome", headless=False, no_viewport=True,
                )
                page = ctx.pages[0] if ctx.pages else await ctx.new_page()
                if not page.url.startswith(DIGITS_URL):
                    await page.goto(DIGITS_URL)
                last_diag = None
                while not self._stop.is_set():
                    snap, err = None, None
                    try:
                        # Follow whichever tab currently shows DIGITS (login may open a new one).
                        digits_pages = [p for p in ctx.pages if not p.is_closed() and p.url.startswith(DIGITS_URL)]
                        if digits_pages:
                            page = digits_pages[-1]
                        elif not ctx.pages:
                            page = await ctx.new_page()
                            await page.goto(DIGITS_URL)
                        snap = await page.evaluate(SNAPSHOT_JS)
                    except Exception as e:  # navigation in progress, tab crashed, etc.
                        err = type(e).__name__
                    if page.is_closed() or not page.url.startswith(DIGITS_URL):
                        snap = None  # sign-in redirect, closed tab: unknown, not "logged out"
                    self.page, self.ctx = page, ctx
                    diag = (len(ctx.pages), page.url.split("#")[0][:60] if not page.is_closed() else "closed",
                            err, snap and (snap.get("loggedIn"), snap.get("loggedOut"), snap.get("ringing"), snap.get("inCall")))
                    if diag != last_diag:  # state-change diagnostics only; never caller details
                        log.info("digits watch: tabs=%s url=%s err=%s (loggedIn, loggedOut, ringing)=%s", *diag)
                        if snap and snap.get("alertCount", 0) > 1:
                            log.warning("DIGITS shows %s call alerts at once", snap["alertCount"])
                        last_diag = diag
                    await self.apply_snapshot(snap)
                    try:
                        await asyncio.wait_for(self._stop.wait(), timeout=POLL_SECONDS)
                    except asyncio.TimeoutError:
                        pass
                await ctx.close()
        finally:
            self._stop_keep_awake()
            if self.connected and self.sink is not None:
                self.connected = False
                await self.sink.on_backend_status(False)

    def stop(self) -> None:
        self._stop.set()

    # ---- keep the Mac awake while observing (lid-close still sleeps) --------------------
    def _start_keep_awake(self) -> None:
        if not self.keep_awake:
            return
        try:
            import os
            self._caffeinate = subprocess.Popen(["/usr/bin/caffeinate", "-is", "-w", str(os.getpid())])
        except OSError:
            log.warning("caffeinate unavailable; the Mac may sleep and DIGITS may sign out")

    def _stop_keep_awake(self) -> None:
        if self._caffeinate and self._caffeinate.poll() is None:
            self._caffeinate.terminate()
