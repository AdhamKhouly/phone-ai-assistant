from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from phone_assistant.calls.manager import CallManager
from phone_assistant.control import ControlService
from phone_assistant.phone.simulated import SimulatedPhoneBackend

OWNER = 111
STRANGER = 999
ASSISTANT_CH = 10
CALLS_CH = 20
OTHER_CH = 30


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 11, 19, 12, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> datetime:
        self.now += timedelta(seconds=seconds)
        return self.now


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def backend():
    return SimulatedPhoneBackend()


@pytest.fixture
def events():
    return []


@pytest.fixture
def manager(backend, clock, events):
    m = CallManager(backend, OWNER, answer_window_s=25, max_command_age_s=15, clock=clock)

    async def record(ev):
        events.append(ev)

    m.subscribe(record)
    return m


@pytest.fixture
def control(manager):
    return ControlService(manager, OWNER, {ASSISTANT_CH, CALLS_CH},
                          {"Mom": "(617) 555-0199", "Broken": "nope"})
