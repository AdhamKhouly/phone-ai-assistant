"""DIGITS backend WITH manual answering (Phase 2b).

Enabled only when BOTH config switches are set:
    PHONE_BACKEND=digits_answer   and   ALLOW_REAL_ANSWER=yes

Exactly two kinds of page actions exist in this file, and nowhere else in the project:
  1. a click on the ringing alert's "Accept call" button, only inside `answer()`, which is only
     reachable through CallManager.answer (owner-only, fresh command, issued after ringing
     started, inside the answer window, one unambiguous call);
  2. a click on the in-call "End" button, or closing the DIGITS tab as a fallback, to end a call.
No keystrokes are ever sent (Enter on the ringing alert may accept a call).

Right before clicking Accept, the page is re-read and the click is refused unless exactly one
alert is showing, for the SAME caller (number and alert label) that call #N was created for.
The click itself targets a locator filtered by that caller's number, in Playwright strict mode,
so it cannot land on a different call's alert.
"""
from __future__ import annotations

import asyncio
import logging
import re

from .digits import ACCEPT_SEL, ALERT_SEL, DIGITS_URL, SNAPSHOT_JS, DigitsObserver, parse_caller

log = logging.getLogger(__name__)

END_RE = re.compile(r"^(end( call)?|hang ?up|end the call)$", re.I)
CONFIRM_TIMEOUT_S = 4.0
CLICK_TIMEOUT_MS = 1500


def _number_regex(number: str) -> re.Pattern | None:
    """Match a +1XXXXXXXXXX number in DIGITS' display format, e.g. (617) 555-0123."""
    digits = re.sub(r"\D", "", number)[-10:]
    if len(digits) != 10:
        return None
    a, b, c = digits[:3], digits[3:6], digits[6:]
    return re.compile(rf"\(?{a}\)?[\s.-]?{b}[\s.-]?{c}")


class DigitsAnswerBackend(DigitsObserver):
    name = "digits (manual answer)"
    can_answer = True
    can_dial = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.home_url = DIGITS_URL
        self._current_label: str | None = None
        self._answering = False
        self._in_call_id: int | None = None
        self._seen_in_call = False

    # ---- helpers ------------------------------------------------------------------------
    async def _snap(self) -> dict | None:
        try:
            if self.page is None or self.page.is_closed():
                return None
            return await self.page.evaluate(SNAPSHOT_JS)
        except Exception:
            return None

    async def _button_labels(self) -> list[str]:
        """UI control labels only (never call content); used to learn the in-call view."""
        try:
            return await self.page.evaluate(
                "() => [...document.querySelectorAll('button,[role=button]')]"
                ".map(b => (b.getAttribute('aria-label') || b.title || '').trim()).filter(Boolean).slice(0, 30)")
        except Exception:
            return []

    def _clear_call(self) -> None:
        self._in_call_id = None
        self._seen_in_call = False

    # ---- state: remember the exact alert each call was created from ------------------------
    async def apply_snapshot(self, snap: dict | None) -> None:
        if self._answering:
            return  # answer() owns the page state while it confirms
        if self._in_call_id is not None and snap is not None and not snap.get("loggedOut"):
            if snap.get("inCall"):
                self._seen_in_call = True
            elif self._seen_in_call and not snap.get("ringing"):
                cid = self._in_call_id
                self._clear_call()
                await self.sink.on_remote_ended(cid, "remote_hangup")
            return  # unknown in-call view: rely on `hangup` / max call duration
        if snap and snap.get("inCall") and self._in_call_id is None and self.page is not None:
            # A connected call the assistant doesn't own (e.g. left over from a crash): end it.
            log.warning("in-call view with no assistant-owned call; ending it")
            await self._end_call()
            return
        before = self._current_call_id
        await super().apply_snapshot(snap)
        if self._current_call_id is not None and self._current_call_id != before:
            self._current_label = (snap or {}).get("label")
        if self._current_call_id is None:
            self._current_label = None
        if self._in_call_id is not None and not self.connected:
            self._clear_call()  # manager ends the active call on disconnect

    # ---- answer: verify, then one click -------------------------------------------------------
    async def answer(self, call_id: int) -> bool:
        if call_id != self._current_call_id or self.page is None or self._in_call_id is not None:
            return False
        expected_number, expected_label = self._current_caller, self._current_label
        snap = await self._snap()
        if not snap or not snap.get("ringing") or snap.get("alertCount") != 1:
            log.warning("answer refused: call #%s no longer the single ringing alert", call_id)
            return False
        now_caller = parse_caller(snap.get("label"), snap.get("body"))
        if now_caller.number != expected_number or snap.get("label") != expected_label:
            log.warning("answer refused: ringing caller changed for call #%s", call_id)
            return False

        alert = self.page.locator(ALERT_SEL)
        rx = _number_regex(expected_number)
        if rx is not None:
            alert = alert.filter(has_text=rx)
        button = alert.locator(ACCEPT_SEL)
        self._answering = True
        try:
            if await button.count() != 1:
                log.warning("answer refused: Accept button for call #%s not uniquely present", call_id)
                return False
            await button.click(timeout=CLICK_TIMEOUT_MS)   # strict: exactly one match or it throws
            log.info("clicked Accept for call #%s", call_id)
            deadline = asyncio.get_running_loop().time() + CONFIRM_TIMEOUT_S
            while asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.25)
                s = await self._snap()
                if s and not s.get("ringing"):
                    self._in_call_id = call_id
                    self._seen_in_call = bool(s.get("inCall"))
                    self._current_call_id, self._current_caller, self._current_label = None, None, None
                    if not self._seen_in_call:
                        log.warning("call #%s: alert gone but in-call view not recognised; controls=%s",
                                    call_id, await self._button_labels())
                    return True
            log.warning("call #%s: still ringing %ss after Accept; treating as not answered", call_id, CONFIRM_TIMEOUT_S)
            return False
        except Exception as e:
            log.warning("answer for call #%s failed safely: %s", call_id, type(e).__name__)
            return False
        finally:
            self._answering = False

    # ---- hang up: End button, else close the tab ------------------------------------------------
    async def hangup(self, call_id: int) -> bool:
        if self._in_call_id is not None and call_id != self._in_call_id:
            return False
        return await self._end_call()

    async def _end_call(self) -> bool:
        ended = False
        try:
            end = self.page.get_by_role("button", name=END_RE)
            if await end.count() >= 1:
                await end.first.click(timeout=CLICK_TIMEOUT_MS)
                for _ in range(12):
                    await asyncio.sleep(0.25)
                    s = await self._snap()
                    if s and not s.get("inCall"):
                        ended = True
                        break
        except Exception as e:
            log.warning("End button failed: %s", type(e).__name__)
        if not ended:
            log.warning("ending call by closing the DIGITS tab (fallback)")
            ended = await self._close_and_reopen()
        self._clear_call()
        return ended

    async def _close_and_reopen(self) -> bool:
        """Closing the tab is what drops the call; reopening DIGITS is best-effort afterwards."""
        ctx = self.ctx or self.page.context
        try:
            await self.page.close()
        except Exception as e:
            log.error("could not close the DIGITS tab: %s", type(e).__name__)
            return False
        try:
            page = await ctx.new_page()
            await page.goto(self.home_url)
            self.page = page
        except Exception as e:
            log.error("DIGITS tab closed (call dropped) but reopening failed: %s", type(e).__name__)
        return True
