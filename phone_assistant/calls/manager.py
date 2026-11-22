"""CallManager — the single source of truth for call state.

Safety rules enforced here (and covered by tests/test_call_manager.py):
  1. Nothing is answered unless the authorized owner issues `answer`.
  2. `answer` only applies to a call that is ringing *right now*:
     - the command must have been issued AFTER that call started ringing,
     - the command must be fresh (not delayed/replayed),
     - the call must still be inside the answer window,
     - if more than one call is ringing, the owner must pick one explicitly.
  3. With no decision, the call is simply left alone (normal iPhone ring + voicemail).
  4. Backend offline -> no call can be answered; ringing calls are released.
  5. Only metadata is stored or logged. No audio, transcripts, or conversation content.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

from ..phone.base import PhoneBackend
from .models import Call, CallState, Direction

log = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def mask_number(number: str) -> str:
    """For local logs only: keep the last 4 digits."""
    digits = [c for c in number if c.isdigit()]
    return "***" + "".join(digits[-4:]) if len(digits) >= 4 else "***"


@dataclass
class Result:
    ok: bool
    message: str
    call: Call | None = None


@dataclass
class CallEvent:
    kind: str          # incoming | active | ignored | missed | ended | failed | dialing | backend
    call: Call | None
    detail: str = ""


Listener = Callable[[CallEvent], Awaitable[None]]


class CallManager:
    def __init__(
        self,
        backend: PhoneBackend,
        owner_id: int,
        *,
        answer_window_s: float = 25.0,
        max_command_age_s: float = 15.0,
        max_call_s: float | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.backend = backend
        self.owner_id = owner_id
        self.answer_window = timedelta(seconds=answer_window_s)
        self.max_command_age = timedelta(seconds=max_command_age_s)
        self.max_call = timedelta(seconds=max_call_s) if max_call_s else None
        self.clock = clock
        self.calls: dict[int, Call] = {}
        self._next_id = 1
        self._lock = asyncio.Lock()
        self._listeners: list[Listener] = []
        backend.attach(self)

    # ------------------------------------------------------------------ helpers
    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)

    async def _emit(self, kind: str, call: Call | None, detail: str = "") -> None:
        for listener in self._listeners:
            try:
                await listener(CallEvent(kind, call, detail))
            except Exception:  # a broken notifier must never affect call handling
                log.exception("listener failed for event %s", kind)

    def _new_call(self, direction: Direction, number: str, name: str | None, state: CallState) -> Call:
        call = Call(self._next_id, direction, number, self.clock(), remote_name=name, state=state)
        self.calls[call.id] = call
        self._next_id += 1
        return call

    def _finish(self, call: Call, state: CallState, reason: str) -> None:
        call.state = state
        call.ended_at = self.clock()
        call.end_reason = reason
        call.objective = None  # drop owner instructions as soon as the call is over
        log.info("call #%s -> %s (%s)", call.id, state.value, reason)

    def ringing(self) -> list[Call]:
        return [c for c in self.calls.values() if c.state is CallState.RINGING]

    def active(self) -> list[Call]:
        return [c for c in self.calls.values()
                if c.state in (CallState.ACTIVE, CallState.ANSWERING, CallState.DIALING)]

    def _check_command(self, user_id: int, issued_at: datetime) -> str | None:
        if user_id != self.owner_id:
            return "Not authorized."
        age = self.clock() - issued_at
        if age > self.max_command_age:
            return f"Ignored a stale command (sent {int(age.total_seconds())}s ago)."
        return None

    # ------------------------------------------------------------ backend events
    async def on_incoming(self, remote_number: str, remote_name: str | None = None) -> Call:
        async with self._lock:
            call = self._new_call(Direction.INCOMING, remote_number, remote_name, CallState.RINGING)
            log.info("call #%s incoming from %s", call.id, mask_number(remote_number))
        await self._emit("incoming", call)
        return call

    async def on_remote_ended(self, call_id: int, reason: str) -> None:
        async with self._lock:
            call = self.calls.get(call_id)
            if call is None:
                return
            if call.state.is_final:
                # Answer window already closed (or ignored): still report when ringing really stops.
                if (call.direction is Direction.INCOMING and call.ring_stopped_at is None
                        and call.answered_at is None):
                    call.ring_stopped_at = self.clock()
                    kind = "stopped"
                else:
                    return
            elif call.state is CallState.RINGING:
                call.ring_stopped_at = self.clock()
                # Caller hung up, voicemail took it, or it was answered on the iPhone.
                self._finish(call, CallState.MISSED, reason)
                kind = "missed"
            else:
                self._finish(call, CallState.ENDED, reason)
                kind = "ended"
        await self._emit(kind, call)

    async def on_remote_answered(self, call_id: int) -> None:
        async with self._lock:
            call = self.calls.get(call_id)
            if call is None or call.state is not CallState.DIALING:
                return
            call.state = CallState.ACTIVE
            call.answered_at = self.clock()
        await self._emit("active", call)

    async def on_backend_status(self, connected: bool) -> None:
        released: list[Call] = []
        async with self._lock:
            if not connected:
                for call in self.calls.values():
                    if call.state is CallState.RINGING:
                        self._finish(call, CallState.MISSED, "backend_disconnected")
                        released.append(call)
                    elif call.state in (CallState.ACTIVE, CallState.ANSWERING, CallState.DIALING):
                        self._finish(call, CallState.ENDED, "backend_disconnected")
                        released.append(call)
        await self._emit("backend", None, "connected" if connected else "disconnected")
        for call in released:
            await self._emit(call.state.value, call)

    async def expire_stale(self) -> list[Call]:
        """Release ringing calls that are past the answer window (left to voicemail)."""
        expired: list[Call] = []
        async with self._lock:
            now = self.clock()
            for call in self.ringing():
                if now - call.started_at > self.answer_window:
                    self._finish(call, CallState.MISSED, "no_decision_in_time")
                    expired.append(call)
        for call in expired:
            await self._emit("missed", call)
        return expired

    async def enforce_max_duration(self) -> list[Call]:
        """Safety net: end AI-handled calls that exceed max_call_s (e.g. Discord unreachable)."""
        if self.max_call is None:
            return []
        now = self.clock()
        async with self._lock:
            overdue = [c for c in self.calls.values() if c.state is CallState.ACTIVE
                       and c.answered_at is not None and now - c.answered_at > self.max_call]
        ended = []
        for call in overdue:
            ok = await self.backend.hangup(call.id)
            async with self._lock:
                if call.state.is_final:
                    continue
                self._finish(call, CallState.ENDED, "max_duration" if ok else "max_duration_backend_error")
            ended.append(call)
            await self._emit("ended", call)
        return ended

    # ------------------------------------------------------------ owner commands
    async def end_by_assistant(self, call_id: int, reason: str) -> bool:
        """The assistant itself ends a call it is handling (audio safety guard, pipeline failure,
        nobody speaking). It can only ever END an assistant-handled call; it cannot answer or dial."""
        async with self._lock:
            call = self.calls.get(call_id)
            if call is None or call.state is not CallState.ACTIVE:
                return False
        ok = await self.backend.hangup(call.id)
        async with self._lock:
            if call.state.is_final:
                return True
            self._finish(call, CallState.ENDED, reason if ok else reason + "_backend_error")
        await self._emit("ended", call)
        return ok

    def _resolve(self, candidates: list[Call], call_id: int | None, verb: str) -> Call | str:
        if call_id is not None:
            match = [c for c in candidates if c.id == call_id]
            if not match:
                existing = self.calls.get(call_id)
                if existing is None:
                    return f"There is no call #{call_id}."
                return f"Call #{call_id} is {existing.state.value}; nothing to {verb}."
            return match[0]
        if not candidates:
            return f"There is no call to {verb}."
        if len(candidates) > 1:
            ids = ", ".join(f"#{c.id} {c.caller_label}" for c in candidates)
            return f"More than one call: {ids}. Reply `{verb} <number>`, e.g. `{verb} {candidates[0].id}`."
        return candidates[0]

    async def answer(self, user_id: int, issued_at: datetime, call_id: int | None = None) -> Result:
        await self.expire_stale()  # an out-of-window call can never be picked
        async with self._lock:
            if err := self._check_command(user_id, issued_at):
                return Result(False, err)
            if not self.backend.connected:
                return Result(False, "Phone connection is offline; the call is left alone.")
            target = self._resolve(self.ringing(), call_id, "answer")
            if isinstance(target, str):
                return Result(False, target)
            if self.clock() - target.started_at > self.answer_window:
                return Result(False, f"Call #{target.id} is past the answer window; left to voicemail.")
            if issued_at < target.started_at:
                return Result(False, f"That command was sent before call #{target.id} started ringing; ignored.")
            if self.active():
                return Result(False, "The assistant is already on a call. `hangup` first.")
            if not self.backend.can_answer:
                return Result(False, f"Real answering isn't enabled yet (observe-only mode). "
                                     f"Call #{target.id} is left to your phone/voicemail.", target)
            target.state = CallState.ANSWERING

        ok = await self.backend.answer(target.id)  # outside the lock: may take a moment

        async with self._lock:
            if target.state is not CallState.ANSWERING:  # remote ended meanwhile
                return Result(False, f"Call #{target.id} ended before it could be answered.", target)
            if not ok:
                self._finish(target, CallState.FAILED, "answer_failed")
                failed = True
            else:
                target.state = CallState.ACTIVE
                target.answered_at = self.clock()
                failed = False
        if failed:
            await self._emit("failed", target)
            why = getattr(self.backend, "last_refusal", "")
            return Result(False, f"Couldn't answer call #{target.id}; it was left to ring/voicemail."
                                 + (f" Reason: {why}." if why else ""), target)
        await self._emit("active", target)
        return Result(True, f"Answering call #{target.id} from {target.caller_label}.", target)

    async def ignore(self, user_id: int, issued_at: datetime, call_id: int | None = None) -> Result:
        async with self._lock:
            if err := self._check_command(user_id, issued_at):
                return Result(False, err)
            target = self._resolve(self.ringing(), call_id, "ignore")
            if isinstance(target, str):
                return Result(False, target)
            # Do nothing to the phone: it keeps ringing and goes to voicemail normally.
            self._finish(target, CallState.IGNORED, "owner_ignored")
        await self._emit("ignored", target)
        return Result(True, f"Ignoring call #{target.id}. Your phone and voicemail behave normally.", target)

    async def hangup(self, user_id: int, issued_at: datetime, call_id: int | None = None) -> Result:
        async with self._lock:
            if err := self._check_command(user_id, issued_at):
                return Result(False, err)
            target = self._resolve(self.active(), call_id, "hangup")
            if isinstance(target, str):
                return Result(False, target)
        ok = await self.backend.hangup(target.id)
        async with self._lock:
            if target.state.is_final:
                return Result(True, f"Call #{target.id} had already ended.", target)
            self._finish(target, CallState.ENDED, "owner_hangup" if ok else "owner_hangup_backend_error")
        await self._emit("ended", target)
        if not ok:
            return Result(False, f"Marked call #{target.id} ended, but the phone backend didn't confirm. Check your phone.", target)
        return Result(True, f"Hung up call #{target.id}.", target)

    async def dial(self, user_id: int, issued_at: datetime, number: str,
                   name: str | None = None, objective: str | None = None) -> Result:
        async with self._lock:
            if err := self._check_command(user_id, issued_at):
                return Result(False, err)
            if not self.backend.can_dial:
                return Result(False, "Outgoing calls aren't enabled (observe-only mode, or ALLOW_OUTGOING is off).")
            if not self.backend.connected:
                return Result(False, "Phone connection is offline; can't place a call.")
            if self.active():
                return Result(False, "The assistant is already on a call. `hangup` first.")
            call = self._new_call(Direction.OUTGOING, number, name, CallState.DIALING)
            call.objective = objective
        ok = await self.backend.dial(call.id, number)
        if not ok:
            async with self._lock:
                self._finish(call, CallState.FAILED, "dial_failed")
            await self._emit("failed", call)
            why = getattr(self.backend, "last_refusal", "")
            return Result(False, f"Couldn't place call #{call.id}" + (f": {why}." if why else "."), call)
        await self._emit("dialing", call)
        # The backend returns once DIGITS shows its call screen. From here the assistant listens and
        # speaks only after the other person does.
        if getattr(self.backend, "active_on_dial", False):
            await self.on_remote_answered(call.id)
        return Result(True, f"Calling {call.caller_label} (call #{call.id}). Type `hangup` to end it.", call)

    async def shutdown(self) -> list[Call]:
        """Assistant is stopping: release ringing calls untouched, hang up AI-handled calls."""
        released: list[Call] = []
        async with self._lock:
            live = [c for c in self.calls.values() if not c.state.is_final]
        for call in live:
            if call.state in (CallState.ACTIVE, CallState.ANSWERING, CallState.DIALING):
                await self.backend.hangup(call.id)
            async with self._lock:
                if not call.state.is_final:
                    state = CallState.MISSED if call.state is CallState.RINGING else CallState.ENDED
                    self._finish(call, state, "assistant_shutdown")
                    released.append(call)
        return released

    # ------------------------------------------------------------------ status
    def status_text(self, discord_online: bool = True) -> str:
        now = self.clock()
        lines = [
            f"Assistant: {'online' if discord_online else 'offline'}",
            f"Phone backend: {self.backend.name} ({'connected' if self.backend.connected else 'DISCONNECTED'})",
        ]
        note = getattr(self.backend, "status_note", None)
        if callable(note):
            lines.append(note())
        ringing, active = self.ringing(), self.active()
        if not ringing and not active:
            lines.append("Calls: none")
        for c in ringing:
            age = int((now - c.started_at).total_seconds())
            lines.append(f"Ringing: #{c.id} {c.caller_label} ({age}s)")
        for c in active:
            dur = c.duration_seconds(now)
            lines.append(f"{c.state.value.capitalize()}: #{c.id} {c.direction.value} {c.caller_label}"
                         + (f" ({dur}s)" if dur is not None else ""))
        return "\n".join(lines)
