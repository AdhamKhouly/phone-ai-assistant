"""Phone backend interface.

A backend is the ONLY component that touches the real phone line. Safety contract:
  * It must never answer, decline, or dial on its own. It only reports events
    (incoming / ended / connection status) and executes explicit commands from CallManager.
  * If the backend process dies, the phone must behave exactly as if this project
    did not exist (iPhone rings, voicemail works). Backends must therefore never
    change carrier settings (forwarding, ring time, DND) to function.
  * Backends must never write call audio to disk.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol


class PhoneEventSink(Protocol):
    async def on_incoming(self, remote_number: str, remote_name: str | None = None) -> object: ...
    async def on_remote_ended(self, call_id: int, reason: str) -> object: ...
    async def on_remote_answered(self, call_id: int) -> object: ...
    async def on_backend_status(self, connected: bool) -> object: ...


class PhoneBackend(ABC):
    name: str = "abstract"
    can_answer: bool = True   # False = observe-only; CallManager refuses `answer`
    can_dial: bool = True

    def __init__(self) -> None:
        self.sink: PhoneEventSink | None = None
        self.connected: bool = False

    def attach(self, sink: PhoneEventSink) -> None:
        self.sink = sink

    @abstractmethod
    async def answer(self, call_id: int) -> bool:
        """Pick up the specific ringing call. Return False if it could not be answered."""

    @abstractmethod
    async def hangup(self, call_id: int) -> bool:
        """End an AI-handled call."""

    @abstractmethod
    async def dial(self, call_id: int, number: str) -> bool:
        """Start an outgoing call. Return False if it could not be started."""
