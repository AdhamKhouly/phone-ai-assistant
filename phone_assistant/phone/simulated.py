"""Simulated phone backend for Phase 1. Touches no real phone line."""
from __future__ import annotations

from .base import PhoneBackend


class SimulatedPhoneBackend(PhoneBackend):
    name = "simulated"

    def __init__(self) -> None:
        super().__init__()
        self.connected = True
        self.fail_next_answer = False   # test hook: simulate a backend failure
        self.actions: list[tuple[str, int]] = []  # (action, call_id) — metadata only

    # --- commands from CallManager -------------------------------------------------
    async def answer(self, call_id: int) -> bool:
        self.actions.append(("answer", call_id))
        if not self.connected or self.fail_next_answer:
            self.fail_next_answer = False
            return False
        return True

    async def hangup(self, call_id: int) -> bool:
        self.actions.append(("hangup", call_id))
        return self.connected

    async def dial(self, call_id: int, number: str) -> bool:
        self.actions.append(("dial", call_id))
        return self.connected

    # --- simulated events from "the phone network" --------------------------------------
    async def simulate_incoming(self, number: str, name: str | None = None):
        assert self.sink is not None
        return await self.sink.on_incoming(number, name)

    async def simulate_remote_end(self, call_id: int, reason: str = "remote_hangup"):
        assert self.sink is not None
        return await self.sink.on_remote_ended(call_id, reason)

    async def simulate_remote_answer(self, call_id: int):
        assert self.sink is not None
        return await self.sink.on_remote_answered(call_id)

    async def set_connected(self, connected: bool):
        self.connected = connected
        if self.sink is not None:
            await self.sink.on_backend_status(connected)
