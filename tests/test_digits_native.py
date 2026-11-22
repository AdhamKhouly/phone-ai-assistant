"""Native DIGITS app backend: safety suite against a fake accessibility tree.

The fake app reproduces the structure captured from the real app (v2.5.19, 2026-11-20) and counts
every press. Each rejection rule is verified by asserting that Accept was NEVER pressed.
Nothing here touches the real DIGITS app or macOS Accessibility.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import phone_assistant.phone.digits_native as dn
from phone_assistant.calls.manager import CallManager
from phone_assistant.calls.models import CallState
from phone_assistant.control import ControlService
from phone_assistant.phone.digits_native import (DigitsNativeAnswerBackend, DigitsNativeObserver,
                                                 find_alerts, snapshot)

from .conftest import ASSISTANT_CH, CALLS_CH, OWNER, STRANGER

ROOT = Path(__file__).resolve().parent.parent


def N(role, *, id="", desc="", title="", value="", subrole="", children=(), el=None):
    return {"role": role, "subrole": subrole, "title": title, "desc": desc, "id": id, "value": value,
            "children": list(children), "_el": el}


def T(value):
    return N("AXStaticText", value=value)


def chrome(title, *content):
    """Electron window wrapper exactly as the real app exposes it."""
    return N("AXWindow", subrole="AXStandardWindow", title=title, children=[
        N("AXGroup", children=[N("AXGroup", children=[N("AXGroup", children=[N("AXGroup", children=[
            N("AXWebArea", children=list(content))])])])]),
        N("AXButton", subrole="AXCloseButton"), N("AXButton", subrole="AXMinimizeButton")])


def incoming_window(name, number, el=None):
    return chrome("Incoming Call", N("AXGroup", id="body", children=[
        N("AXGroup", id="hiddenARIA", children=[T("Incoming call. Press Enter to accept it")]),
        N("AXImage", desc="caller photo", id="callerPhoto"),
        N("AXGroup", id="title", children=[T("Incoming audio call...")]),
        N("AXGroup", id="displayName", children=[T(name)]),
        N("AXGroup", id="originator", children=[T(number)]),
        N("AXButton", desc="Reject call", id="reject", el=("reject", number)),
        N("AXButton", desc="Accept call", id="accept", el=el or ("accept", number))]))


OTHER_APP, DIGITS_APP = 501, 777


class FakeDigitsApp:
    """Stands in for native_ax.MacAXSource."""

    def __init__(self) -> None:
        self.running = True
        self.signed_in = True
        self.window_open = True
        self.banner = True               # the real app currently shows its connectivity banner
        self.rings: list[tuple[str, str]] = []
        self.in_call_label: str | None = None
        self.end_label = "End call"
        self.broken_accept = False
        self.broken_end = False
        self.quit_fails = False
        self.accepts = self.ends = self.rejects = self.quits = self.launches = self.other_presses = 0
        # focus model: which "app" is in front (OTHER_APP stands for Word/Safari), and DIGITS' hidden flag
        self.hidden = False
        self.front = OTHER_APP
        self.hides = 0
        self.activations: list[int] = []
        self.grabs_focus_on_launch = True     # measured: DIGITS puts itself in front when started/reopened
        self.last_window_count = 1

    def trusted(self):
        return True

    def read(self):
        if not self.running:
            return None
        wins = []
        # measured: a closed window is unreadable while the app is unhidden, readable once hidden
        self.last_window_count = 1 if (self.window_open or self.hidden) else 0
        if self.window_open or self.hidden:
            if self.signed_in:
                content = [N("AXButton", desc="Settings menu", id="setting-menu"),
                           N("AXRadioButton", desc="Calls", id="calls"),
                           N("AXButton", desc="New call", children=[T("New call")]),
                           N("AXButton", title="Sign out", children=[T("Sign out")])]
                if self.banner:
                    content.insert(0, N("AXGroup", subrole="AXApplicationAlert", children=[
                        T("We are having trouble with connectivity, some system notifications may not be available"),
                        N("AXButton", desc="dismiss banner")]))
                if self.in_call_label:
                    content.append(N("AXButton", desc=self.in_call_label, el=("end",)))
            else:
                content = [N("AXHeading", title="Free your phone and your number with DIGITS"),
                           N("AXButton", desc="Press Enter to log in to Digits from T-Mobile!", children=[T("Log In")])]
            wins.append(chrome("DIGITS from T-Mobile", *content))
        wins += [incoming_window(name, num) for name, num in self.rings]
        return wins

    def press(self, node):
        el = node["_el"]
        if el and el[0] == "accept":
            self.accepts += 1
            if not self.broken_accept:
                self.rings = [r for r in self.rings if r[1] != el[1]]
                self.in_call_label = self.end_label
        elif el and el[0] == "end":
            self.ends += 1
            if not self.broken_end:
                self.in_call_label = None
        elif el and el[0] == "reject":
            self.rejects += 1
        else:
            self.other_presses += 1
        return True

    def open_tel(self, number):
        self.tel_requests = getattr(self, "tel_requests", []) + [number]
        if getattr(self, "tel_starts_call", True):
            self.in_call_label = self.end_label
        return True

    def quit(self):
        self.quits += 1
        if self.quit_fails:
            return False
        self.running, self.rings, self.in_call_label = False, [], None
        return True

    def launch(self):
        self.launches += 1
        self.running = True
        self.window_open, self.hidden = True, False
        if self.grabs_focus_on_launch:
            self.front = DIGITS_APP
        return True

    def frontmost_pid(self):
        return self.front

    def is_frontmost(self):
        return self.running and self.front == DIGITS_APP

    def is_hidden(self):
        return self.hidden

    def hide(self):
        self.hides += 1
        self.hidden = True
        if self.front == DIGITS_APP:
            self.front = 999              # macOS picks some other app, not necessarily the one you were in
        return True

    def activate_pid(self, pid):
        self.activations.append(pid)
        self.front = pid
        return True

    @property
    def call_actions(self):
        return self.accepts + self.ends + self.rejects + self.other_presses + self.quits


def make(cls, app, clock, events, **kw):
    b = cls(app, keep_awake=False, **kw)
    b.unknown_grace_s = 0
    m = CallManager(b, OWNER, answer_window_s=22, max_command_age_s=15, max_call_s=60, clock=clock)

    async def rec(ev):
        events.append(ev)
    m.subscribe(rec)

    async def tick():
        await b.apply_snapshot(await b._read())
    return b, m, tick


@pytest.fixture
def app():
    return FakeDigitsApp()


@pytest.fixture
async def rig(app, clock, events, monkeypatch):
    monkeypatch.setattr(dn, "CONFIRM_TIMEOUT_S", 1.0)
    monkeypatch.setattr(dn, "SIGNATURE_SETTLE_S", 0.0)
    b, m, tick = make(DigitsNativeAnswerBackend, app, clock, events)
    await tick()   # connected
    assert b.connected
    return app, b, m, tick


async def ring(app, tick, name="Jane Doe", num="(617) 555-0123"):
    app.rings = [(name, num)]
    await tick()


def kinds(events):
    return [e.kind for e in events if e.kind != "backend"]


# ------------------------------------------------------------ reading the captured structure
def test_snapshot_of_real_structure_signed_in_idle():
    app = FakeDigitsApp()
    s = snapshot(app.read())
    assert s["state"] == "online" and s["alerts"] == [] and not s["inCall"] and s["degraded"]


def test_snapshot_ringing_reads_caller_and_exactly_one_accept_button():
    app = FakeDigitsApp()
    app.rings = [("Jane Doe", "(617) 555-0123")]
    s = snapshot(app.read())
    (a,) = s["alerts"]
    assert (a.number, a.name) == ("+16175550123", "Jane Doe")
    assert a.accept["desc"] == "Accept call" and a.accept["id"] == "accept"


def test_unknown_caller_shows_number_twice_and_has_no_name():
    (a,) = find_alerts([incoming_window("(617) 555-0123", "(617) 555-0123")])
    assert (a.number, a.name) == ("+16175550123", None)


def test_caller_names_with_periods_and_withheld_numbers():
    (a,) = find_alerts([incoming_window("Dr. A. Smith Jr.", "(617) 555-0123")])
    assert a.name == "Dr. A. Smith Jr."
    (a,) = find_alerts([incoming_window("Private number", "")])
    assert (a.number, a.name) == ("Private number", None)


def test_states_not_running_signed_out_unknown():
    app = FakeDigitsApp()
    app.running = False
    assert snapshot(app.read())["state"] == "not_running"
    app.running, app.signed_in = True, False
    assert snapshot(app.read())["state"] == "signed_out"
    app.signed_in, app.window_open = True, False
    assert snapshot(app.read())["state"] == "unknown"      # no window: can't tell, never "ringing"
    assert snapshot([])["state"] == "unknown"
    assert snapshot([{"role": "AXWindow"}])["state"] == "unknown"   # malformed tree doesn't raise


def test_sign_out_button_is_not_mistaken_for_logged_out():
    app = FakeDigitsApp()
    assert snapshot(app.read())["state"] == "online"


def test_ring_with_main_window_closed_still_counts_as_online():
    app = FakeDigitsApp()
    app.window_open = False
    app.rings = [("Jane Doe", "(617) 555-0123")]
    s = snapshot(app.read())
    assert s["state"] == "online" and len(s["alerts"]) == 1


# ------------------------------------------------------------ observe-only backend
async def test_observer_reports_incoming_and_stop_with_zero_actions(app, clock, events):
    b, m, tick = make(DigitsNativeObserver, app, clock, events)
    await tick()
    await ring(app, tick)
    (call,) = m.calls.values()
    assert call.state is CallState.RINGING and call.remote_number == "+16175550123" and call.remote_name == "Jane Doe"
    r = await m.answer(OWNER, clock.advance(2))
    assert not r.ok and "observe-only" in r.message
    app.rings = []
    await tick()
    assert call.state is CallState.MISSED
    assert app.call_actions == 0


def test_observer_class_cannot_press_anything():
    src = inspect.getsource(DigitsNativeObserver)
    for forbidden in (".press(", ".quit(", "PerformAction"):
        assert forbidden not in src, forbidden
    assert DigitsNativeObserver.can_answer is False and DigitsNativeObserver.can_dial is False


async def test_observer_direct_backend_commands_do_nothing(app, clock, events):
    b, m, tick = make(DigitsNativeObserver, app, clock, events)
    await tick()
    await ring(app, tick)
    assert not await b.answer(1) and not await b.hangup(1) and not await b.dial(1, "+16175550123")
    assert app.call_actions == 0


async def test_signed_out_and_not_running_are_reported_once_and_recover(app, clock, events):
    app.signed_in = False
    b, m, tick = make(DigitsNativeObserver, app, clock, events)
    await tick(); await tick()
    assert [e.detail for e in events] == ["disconnected"]          # started signed out: warned once
    app.signed_in = True
    await tick()
    assert b.connected and events[-1].detail == "connected"
    app.running = False                                            # app quit
    await tick()
    assert not b.connected and events[-1].detail == "disconnected"
    assert "NOT RUNNING" in m.status_text()
    app.running = True                                             # app restarted
    await tick()
    assert b.connected and events[-1].detail == "connected"
    assert "signed in" in m.status_text() and "connectivity" in m.status_text()
    assert app.call_actions == 0


async def test_app_restart_while_ringing_releases_call_without_touching_it(app, clock, events):
    b, m, tick = make(DigitsNativeObserver, app, clock, events)
    await tick()
    await ring(app, tick)
    app.running, app.rings = False, []
    await tick()
    (call,) = m.calls.values()
    assert call.state is CallState.MISSED and call.end_reason == "backend_disconnected"
    assert app.call_actions == 0


async def test_unreadable_window_uses_a_grace_period(app, clock, events):
    b, m, tick = make(DigitsNativeObserver, app, clock, events)
    b.unknown_grace_s = 60
    await tick()
    app.window_open = False
    await tick()
    assert b.connected            # brief unknown: keep state
    b.unknown_grace_s = 0
    await tick()
    assert not b.connected


async def test_run_loop_relaunches_app_but_never_acts_on_calls(app, clock, events, monkeypatch):
    monkeypatch.setattr(dn, "POLL_SECONDS", 0.02)
    app.running = False
    b, m, tick = make(DigitsNativeObserver, app, clock, events, autolaunch=True)
    task = asyncio.create_task(b.run())
    await asyncio.sleep(0.3)
    assert app.launches == 1 and b.connected      # reopened once, reconnected
    app.rings = [("Jane Doe", "(617) 555-0123")]
    await asyncio.sleep(0.2)
    assert kinds(events) == ["incoming"]
    b.stop()
    await task
    assert app.call_actions == 0


async def test_closed_window_is_hidden_not_reopened_and_focus_is_never_taken(app, clock, events, monkeypatch):
    monkeypatch.setattr(dn, "POLL_SECONDS", 0.02)
    monkeypatch.setattr(dn, "WINDOW_MISSING_S", 0.1)
    b, m, tick = make(DigitsNativeObserver, app, clock, events, autolaunch=True)
    b.unknown_grace_s = 60
    task = asyncio.create_task(b.run())
    await asyncio.sleep(0.1)
    app.window_open = False                 # you close the DIGITS window; the app keeps running
    fronts = []
    for _ in range(25):
        await asyncio.sleep(0.02)
        fronts.append(app.front)
    b.stop()
    await task
    assert app.launches == 0 and app.hides == 1 and app.call_actions == 0
    assert set(fronts) == {OTHER_APP} and app.activations == []          # your app stayed in front throughout
    assert b.app_state == "online"                                        # readable again while hidden
    assert [e.detail for e in events] == ["connected", "disconnected"]   # only the final stop


async def test_a_window_you_are_using_is_never_hidden(app, clock, events, monkeypatch):
    monkeypatch.setattr(dn, "POLL_SECONDS", 0.02)
    monkeypatch.setattr(dn, "WINDOW_MISSING_S", 0.05)
    b, m, tick = make(DigitsNativeObserver, app, clock, events, autolaunch=True)
    b.unknown_grace_s = 60
    task = asyncio.create_task(b.run())
    await asyncio.sleep(0.05)
    app.front = DIGITS_APP                  # you clicked into DIGITS yourself...
    app.signed_in = False                   # ...and it shows something the bot doesn't recognise
    app.read = lambda: [chrome("DIGITS", N("AXButton", desc="Something new"))]
    await asyncio.sleep(0.4)
    b.stop()
    await task
    assert app.hides == 0 and app.launches == 0 and app.activations == [] and app.front == DIGITS_APP


async def test_last_resort_reopen_happens_once_then_hides_and_returns_focus(app, clock, events, monkeypatch):
    monkeypatch.setattr(dn, "POLL_SECONDS", 0.02)
    monkeypatch.setattr(dn, "WINDOW_MISSING_S", 0.05)
    monkeypatch.setattr(dn, "HIDE_GRACE_S", 0.1)
    b, m, tick = make(DigitsNativeObserver, app, clock, events, autolaunch=True)
    b.unknown_grace_s = 60
    stuck = {"on": True}

    real_read = app.read

    def read():                                 # a window that was never shown: unreadable even when hidden
        if stuck["on"] and not app.window_open:
            app.last_window_count = 1 if app.hidden else 0
            return [chrome("DIGITS")] if app.hidden else []
        return real_read()
    app.read = read
    real_launch = app.launch
    app.launch = lambda: (stuck.update(on=False), real_launch())[1]
    task = asyncio.create_task(b.run())
    await asyncio.sleep(0.05)
    app.window_open = False
    await asyncio.sleep(0.8)
    b.stop()
    await task
    assert app.launches == 1                      # asked to show its window exactly once, not every minute
    assert app.hides == 2 and app.activations == [OTHER_APP] and app.front == OTHER_APP and app.hidden
    assert app.call_actions == 0


async def test_restart_of_digits_hides_it_once_readable_and_returns_focus(app, clock, events, monkeypatch):
    monkeypatch.setattr(dn, "POLL_SECONDS", 0.02)
    app.running = False
    b, m, tick = make(DigitsNativeObserver, app, clock, events, autolaunch=True)
    task = asyncio.create_task(b.run())
    await asyncio.sleep(0.4)
    b.stop()
    await task
    assert app.launches == 1 and app.hides == 1 and app.hidden and b.app_state == "online"
    assert app.activations == [OTHER_APP] and app.front == OTHER_APP


async def test_signed_out_digits_is_left_visible_so_you_can_log_in(app, clock, events, monkeypatch):
    monkeypatch.setattr(dn, "POLL_SECONDS", 0.02)
    app.running, app.signed_in = False, False
    b, m, tick = make(DigitsNativeObserver, app, clock, events, autolaunch=True)
    task = asyncio.create_task(b.run())
    await asyncio.sleep(0.3)
    b.stop()
    await task
    assert app.launches == 1 and app.hides == 0 and not app.hidden


async def test_bot_start_with_digits_already_running_touches_nothing(app, clock, events, monkeypatch):
    monkeypatch.setattr(dn, "POLL_SECONDS", 0.02)
    b, m, tick = make(DigitsNativeObserver, app, clock, events, autolaunch=True)
    task = asyncio.create_task(b.run())
    await asyncio.sleep(0.3)
    b.stop()
    await task
    assert (app.launches, app.hides, app.activations, app.front) == (0, 0, [], OTHER_APP)


async def test_digits_that_pops_up_for_a_ring_is_sent_back_when_the_ring_ends(app, clock, events, monkeypatch):
    monkeypatch.setattr(dn, "POLL_SECONDS", 0.02)
    b, m, tick = make(DigitsNativeObserver, app, clock, events, autolaunch=True)
    task = asyncio.create_task(b.run())
    await asyncio.sleep(0.1)
    app.rings, app.front = [("Jane Doe", "(617) 555-0123")], DIGITS_APP     # DIGITS raises its own alert
    await asyncio.sleep(0.2)
    assert app.hides == 0                       # nothing is done to DIGITS while it rings
    app.rings = []
    await asyncio.sleep(0.2)
    b.stop()
    await task
    assert app.hides == 1 and app.activations == [OTHER_APP] and app.front == OTHER_APP
    assert app.call_actions == 0


async def test_digits_you_were_already_using_is_left_in_front_after_a_ring(app, clock, events, monkeypatch):
    monkeypatch.setattr(dn, "POLL_SECONDS", 0.02)
    b, m, tick = make(DigitsNativeObserver, app, clock, events, autolaunch=True)
    app.front = DIGITS_APP
    task = asyncio.create_task(b.run())
    await asyncio.sleep(0.1)
    app.rings = [("Jane Doe", "(617) 555-0123")]
    await asyncio.sleep(0.1)
    app.rings = []
    await asyncio.sleep(0.2)
    b.stop()
    await task
    assert app.hides == 0 and app.activations == [] and app.front == DIGITS_APP


async def test_autolaunch_can_be_disabled(app, clock, events, monkeypatch):
    monkeypatch.setattr(dn, "POLL_SECONDS", 0.02)
    app.running = False
    b, m, tick = make(DigitsNativeObserver, app, clock, events, autolaunch=False)
    task = asyncio.create_task(b.run())
    await asyncio.sleep(0.2)
    b.stop()
    await task
    assert app.launches == 0 and not b.connected


# ------------------------------------------------------------ manual answer: happy path
async def test_owner_answer_presses_accept_once_then_hangup(rig, clock):
    app, b, m, tick = rig
    await ring(app, tick)
    (call,) = m.calls.values()
    r = await m.answer(OWNER, clock.advance(3))
    assert r.ok and call.state is CallState.ACTIVE and app.accepts == 1
    await tick()
    r = await m.hangup(OWNER, clock.advance(5))
    assert r.ok and call.state is CallState.ENDED and call.end_reason == "owner_hangup"
    assert (app.ends, app.quits, app.accepts) == (1, 0, 1)


async def test_plain_text_answer_through_discord_layer(rig, clock):
    app, b, m, tick = rig
    control = ControlService(m, OWNER, {ASSISTANT_CH, CALLS_CH})
    await ring(app, tick)
    reply = await control.handle(OWNER, CALLS_CH, "answer", clock.advance(2))
    assert reply.startswith("Answering call #1") and app.accepts == 1


# ------------------------------------------------------------ never without a valid command
async def test_no_command_means_zero_call_actions(rig, clock, events):
    app, b, m, tick = rig
    await ring(app, tick)
    for _ in range(5):
        await tick()
    clock.advance(23)
    await m.expire_stale()
    app.rings = []
    await tick()
    assert app.call_actions == 0
    assert kinds(events) == ["incoming", "missed", "stopped"]


async def test_unauthorized_user_never_presses(rig, clock):
    app, b, m, tick = rig
    control = ControlService(m, OWNER, {ASSISTANT_CH, CALLS_CH})
    await ring(app, tick)
    assert "only the owner" in await control.handle(STRANGER, CALLS_CH, "answer", clock.advance(1))
    assert "only the owner" in await control.handle(STRANGER, CALLS_CH, "/answer", clock.now)
    assert "only the owner" in await control.handle(STRANGER, CALLS_CH, "hangup", clock.now)
    assert not (await m.answer(STRANGER, clock.now)).ok
    assert app.call_actions == 0


async def test_command_sent_before_the_ring_is_ignored(rig, clock):
    app, b, m, tick = rig
    early = clock.now
    clock.advance(5)
    await ring(app, tick)
    r = await m.answer(OWNER, early)
    assert not r.ok and "before call #1 started ringing" in r.message and app.accepts == 0


async def test_late_delivered_command_is_stale(rig, clock):
    app, b, m, tick = rig
    await ring(app, tick)
    sent = clock.advance(1)
    clock.advance(20)
    r = await m.answer(OWNER, sent)
    assert not r.ok and "stale" in r.message and app.accepts == 0


async def test_answer_after_window_expired(rig, clock):
    app, b, m, tick = rig
    await ring(app, tick)
    r = await m.answer(OWNER, clock.advance(23))
    assert not r.ok and app.accepts == 0


async def test_answer_after_ignore(rig, clock):
    app, b, m, tick = rig
    await ring(app, tick)
    assert (await m.ignore(OWNER, clock.advance(1))).ok
    assert not (await m.answer(OWNER, clock.advance(1))).ok
    assert app.call_actions == 0


async def test_stale_answer_cannot_hit_a_later_call(rig, clock):
    app, b, m, tick = rig
    await ring(app, tick, "First", "(617) 555-0001")
    answer_sent = clock.advance(2)
    app.rings = []                        # first call stops ringing
    await tick()
    clock.advance(3)
    await ring(app, tick, "Second", "(617) 555-0002")
    r = await m.answer(OWNER, answer_sent)   # the old command arrives now
    assert not r.ok and app.accepts == 0
    assert [c.state for c in m.calls.values()] == [CallState.MISSED, CallState.RINGING]


async def test_caller_changed_just_before_press(rig, clock):
    app, b, m, tick = rig
    await ring(app, tick, "First", "(617) 555-0001")
    app.rings = [("Second", "(617) 555-0002")]   # replaced between poll and command; no tick
    r = await m.answer(OWNER, clock.advance(2))
    assert not r.ok and app.accepts == 0


async def test_same_number_but_changed_alert_is_refused(rig, clock):
    app, b, m, tick = rig
    await ring(app, tick, "Jane Doe", "(617) 555-0123")
    app.rings = [("Someone Else", "(617) 555-0123")]
    assert not (await m.answer(OWNER, clock.advance(2))).ok and app.accepts == 0


async def test_alert_gone_just_before_press(rig, clock):
    app, b, m, tick = rig
    await ring(app, tick)
    app.rings = []
    r = await m.answer(OWNER, clock.advance(2))
    assert not r.ok and app.accepts == 0
    assert list(m.calls.values())[0].state is CallState.FAILED


async def test_two_alerts_at_once_are_refused(rig, clock):
    app, b, m, tick = rig
    app.rings = [("A", "(617) 555-0001"), ("B", "(617) 555-0002")]
    await tick()
    r = await m.answer(OWNER, clock.advance(2))
    assert not r.ok and app.accepts == 0


async def test_app_signed_out_or_quit_before_press(rig, clock):
    app, b, m, tick = rig
    await ring(app, tick)
    app.running = False
    assert not (await m.answer(OWNER, clock.advance(2))).ok
    assert app.accepts == 0


async def test_three_simultaneous_answers_press_once(rig, clock):
    app, b, m, tick = rig
    await ring(app, tick)
    t = clock.advance(2)
    results = await asyncio.gather(*(m.answer(OWNER, t) for _ in range(3)))
    assert sum(r.ok for r in results) == 1 and app.accepts == 1


async def test_repeated_answer_after_success_does_not_press_again(rig, clock):
    app, b, m, tick = rig
    await ring(app, tick)
    assert (await m.answer(OWNER, clock.advance(2))).ok
    assert not (await m.answer(OWNER, clock.advance(1))).ok
    assert app.accepts == 1


async def test_accept_that_does_nothing_fails_without_retry(rig, clock):
    app, b, m, tick = rig
    app.broken_accept = True
    await ring(app, tick)
    r = await m.answer(OWNER, clock.advance(2))
    (call,) = m.calls.values()
    assert not r.ok and call.state is CallState.FAILED and app.accepts == 1
    for _ in range(3):
        await tick()
    assert app.accepts == 1 and app.quits == 0     # left to the phone; nothing else touched


# ------------------------------------------------------------ restarts
async def test_bot_restart_while_ringing_changes_nothing(app, clock, events):
    """A new bot process finds a call already ringing: it reports it and presses nothing; an
    `answer` typed before the restart cannot apply to it."""
    pre_restart_answer = clock.now
    clock.advance(4)
    app.rings = [("Jane Doe", "(617) 555-0123")]
    b, m, tick = make(DigitsNativeAnswerBackend, app, clock, events)   # the "restarted" bot
    await tick(); await tick()
    assert kinds(events) == ["incoming"] and app.call_actions == 0
    r = await m.answer(OWNER, pre_restart_answer)
    assert not r.ok and app.accepts == 0
    assert app.rings and app.call_actions == 0        # still ringing, untouched


async def test_bot_shutdown_while_ringing_touches_nothing(rig, clock):
    app, b, m, tick = rig
    await ring(app, tick)
    await m.shutdown()
    assert app.call_actions == 0 and app.rings


async def test_no_auto_answer_after_app_restart_or_session_restore(rig, clock, events):
    app, b, m, tick = rig
    await ring(app, tick)
    app.running = False; await tick()                 # app dies mid-ring
    app.running = True; await tick()                  # comes back, still ringing
    app.signed_in = False; await tick()               # then needs sign-in
    app.signed_in = True; await tick()                # session restored
    assert app.call_actions == 0


# ------------------------------------------------------------ ending calls
async def test_remote_hangup_detected(rig, clock):
    app, b, m, tick = rig
    await ring(app, tick)
    await m.answer(OWNER, clock.advance(2))
    await tick()
    app.in_call_label = None       # other side hangs up
    await tick()
    (call,) = m.calls.values()
    assert call.state is CallState.ACTIVE          # one missing reading is not enough
    await tick()
    assert call.state is CallState.ENDED and call.end_reason == "remote_hangup"
    assert app.ends == 0 and app.quits == 0


# ------------------------------------------------------------ unrelated calls are never ended
UNKNOWN_END = "Leave conversation"     # an in-call screen whose End button we don't recognise


async def _answer_with_unknown_screen(app, m, tick, clock):
    app.end_label = UNKNOWN_END
    await ring(app, tick)
    assert (await m.answer(OWNER, clock.advance(2))).ok
    await tick()                                   # learns the in-call screen from what appeared
    return list(m.calls.values())[0]


async def test_unknown_in_call_screen_is_learned_and_its_end_detected(rig, clock):
    app, b, m, tick = rig
    call = await _answer_with_unknown_screen(app, m, tick, clock)
    assert b._call_sig == {"control:" + UNKNOWN_END}
    app.in_call_label = None                       # caller hangs up
    await tick(); await tick()
    assert call.state is CallState.ENDED and call.end_reason == "remote_hangup"
    assert (app.ends, app.quits) == (0, 0) and b._in_call_id is None


async def test_your_own_call_after_the_assistants_call_survives_everything(rig, clock):
    """The assistant's call ends; then you take a call yourself in DIGITS. Hang-up commands, the
    time limit, and bot shutdown must all leave your call and the app alone."""
    app, b, m, tick = rig
    call = await _answer_with_unknown_screen(app, m, tick, clock)
    app.in_call_label = None
    await tick(); await tick()
    assert call.state is CallState.ENDED
    app.in_call_label = UNKNOWN_END                # you are now on your own call
    for _ in range(3):
        await tick()
    assert not (await m.hangup(OWNER, clock.advance(1))).ok
    clock.advance(120)
    await m.enforce_max_duration()
    await m.shutdown()
    assert app.call_actions == 1 and app.accepts == 1      # only the original Accept, ever
    assert app.in_call_label == UNKNOWN_END and app.running


async def test_hangup_after_call_already_ended_does_not_quit_the_app(rig, clock):
    app, b, m, tick = rig
    await _answer_with_unknown_screen(app, m, tick, clock)
    app.in_call_label = None                       # ended, but the bot hasn't polled yet
    r = await m.hangup(OWNER, clock.advance(2))
    assert r.ok and app.quits == 0 and app.running


async def test_time_limit_after_call_already_ended_does_not_quit_the_app(rig, clock):
    app, b, m, tick = rig
    await _answer_with_unknown_screen(app, m, tick, clock)
    app.in_call_label = None
    clock.advance(61)
    await m.enforce_max_duration()
    assert app.quits == 0 and app.running


async def test_unknown_screen_still_connected_is_ended_by_quitting(rig, clock):
    app, b, m, tick = rig
    await _answer_with_unknown_screen(app, m, tick, clock)
    r = await m.hangup(OWNER, clock.advance(2))
    assert r.ok and (app.ends, app.quits, app.launches) == (0, 1, 1)


async def test_watchdog_stands_down_as_soon_as_the_assistants_call_ends(app, clock, events, monkeypatch):
    monkeypatch.setattr(dn, "CONFIRM_TIMEOUT_S", 1.0)
    monkeypatch.setattr(dn, "SIGNATURE_SETTLE_S", 0.0)
    b, m, tick = make(DigitsNativeAnswerBackend, app, clock, events, watchdog_cmd=["/bin/sleep", "30"])
    await tick()
    await _answer_with_unknown_screen(app, m, tick, clock)
    proc = b._watchdog
    assert proc.poll() is None
    app.in_call_label = None                       # caller hangs up; no command from you
    await tick(); await tick()
    proc.wait(timeout=3)
    assert b._watchdog is None                     # nothing left that could act on a later call


async def test_watchdog_is_tied_to_the_running_digits_process(app, clock, events, monkeypatch):
    monkeypatch.setattr(dn, "CONFIRM_TIMEOUT_S", 1.0)
    app.pid = lambda: 4242
    started = []
    monkeypatch.setattr(dn.subprocess, "Popen", lambda cmd, **kw: started.append(cmd) or _FakeProc())
    b, m, tick = make(DigitsNativeAnswerBackend, app, clock, events, watchdog_cmd=["wd"])
    await tick()
    await ring(app, tick)
    await m.answer(OWNER, clock.advance(2))
    assert started == [["wd", "--digits-pid", "4242"]]


class _FakeProc:
    def poll(self):
        return None

    def terminate(self):
        pass


async def test_unknown_in_call_view_falls_back_to_quitting_and_reopening(rig, clock):
    app, b, m, tick = rig
    app.end_label = "Something unexpected"
    await ring(app, tick)
    assert (await m.answer(OWNER, clock.advance(2))).ok
    r = await m.hangup(OWNER, clock.advance(3))
    assert r.ok and (app.ends, app.quits, app.launches) == (0, 1, 1)


async def test_end_button_that_does_nothing_falls_back_to_quit(rig, clock):
    app, b, m, tick = rig
    app.broken_end = True
    await ring(app, tick)
    await m.answer(OWNER, clock.advance(2))
    r = await m.hangup(OWNER, clock.advance(3))
    assert r.ok and (app.ends, app.quits) == (1, 1)


async def test_failed_quit_is_reported_not_hidden(rig, clock):
    app, b, m, tick = rig
    app.broken_end = app.quit_fails = True
    await ring(app, tick)
    await m.answer(OWNER, clock.advance(2))
    r = await m.hangup(OWNER, clock.advance(3))
    assert not r.ok and "Check your phone" in r.message


async def test_max_duration_auto_hangup(rig, clock):
    app, b, m, tick = rig
    await ring(app, tick)
    await m.answer(OWNER, clock.advance(2))
    clock.advance(61)
    await m.enforce_max_duration()
    (call,) = m.calls.values()
    assert call.state is CallState.ENDED and call.end_reason == "max_duration" and app.ends == 1


async def test_graceful_shutdown_during_answered_call_ends_it(rig, clock):
    app, b, m, tick = rig
    await ring(app, tick)
    await m.answer(OWNER, clock.advance(2))
    await m.shutdown()
    assert app.ends == 1 and app.in_call_label is None


async def test_call_you_answered_by_hand_is_never_touched(rig, clock):
    """The DIGITS app is the owner's own app: a connected call the assistant doesn't own stays."""
    app, b, m, tick = rig
    app.in_call_label = "End call"            # owner picked up manually in the app
    for _ in range(4):
        await tick()
    assert not await b.hangup(1)
    r = await m.hangup(OWNER, clock.advance(1))
    assert not r.ok
    assert app.call_actions == 0 and app.in_call_label == "End call"


async def test_answer_refused_while_another_call_is_connected(rig, clock):
    app, b, m, tick = rig
    app.in_call_label = "End call"
    await ring(app, tick)
    assert not (await m.answer(OWNER, clock.advance(2))).ok and app.accepts == 0


async def test_reject_is_never_pressed_by_any_command(rig, clock):
    app, b, m, tick = rig
    control = ControlService(m, OWNER, {ASSISTANT_CH, CALLS_CH})
    await ring(app, tick)
    await control.handle(OWNER, CALLS_CH, "ignore", clock.advance(1))
    await control.handle(OWNER, CALLS_CH, "hangup", clock.advance(1))
    assert app.rejects == 0 and app.call_actions == 0 and app.rings


async def test_outgoing_calls_are_not_enabled(rig, clock):
    app, b, m, tick = rig
    r = await m.dial(OWNER, clock.advance(1), "+16175550199")
    assert not r.ok and app.call_actions == 0


# ------------------------------------------------------------ crash watchdog
async def test_watchdog_is_armed_only_while_the_assistant_owns_a_call(app, clock, events, monkeypatch):
    monkeypatch.setattr(dn, "CONFIRM_TIMEOUT_S", 1.0)
    b, m, tick = make(DigitsNativeAnswerBackend, app, clock, events, watchdog_cmd=["/bin/sleep", "30"])
    await tick()
    await ring(app, tick)
    assert b._watchdog is None                       # ringing alone never arms it
    await m.answer(OWNER, clock.advance(2))
    proc = b._watchdog
    assert proc is not None and proc.poll() is None
    await m.hangup(OWNER, clock.advance(2))
    proc.wait(timeout=3)
    assert b._watchdog is None


async def test_watchdog_disarmed_when_answer_fails(app, clock, events, monkeypatch):
    monkeypatch.setattr(dn, "CONFIRM_TIMEOUT_S", 0.5)
    app.broken_accept = True
    b, m, tick = make(DigitsNativeAnswerBackend, app, clock, events, watchdog_cmd=["/bin/sleep", "30"])
    await tick()
    await ring(app, tick)
    await m.answer(OWNER, clock.advance(2))
    assert b._watchdog is None


def _wait_for(path: Path, seconds: float) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if path.exists():
            return True
        time.sleep(0.1)
    return False


def _watchdog_argv(parent: str, marker: Path, max_seconds="60"):
    return [sys.executable, "-m", "phone_assistant.phone.native_watchdog", "--parent", parent,
            "--max-seconds", max_seconds, "--poll", "0.1", "--command", json.dumps(["/usr/bin/touch", str(marker)])]


def test_watchdog_fires_when_the_bot_is_killed(tmp_path):
    """kill -9 of the bot (left unreaped, i.e. a zombie) must trigger the drop-the-call action."""
    marker = tmp_path / "fired"
    bot = tmp_path / "bot.py"
    bot.write_text("import os, subprocess, sys, time, json\n"
                   f"argv = {_watchdog_argv('PID', marker)!r}\n"
                   "argv[argv.index('PID')] = str(os.getpid())\n"
                   "subprocess.Popen(argv, start_new_session=True)\n"
                   "time.sleep(60)\n")
    proc = subprocess.Popen([sys.executable, str(bot)], cwd=ROOT)
    try:
        time.sleep(1.0)
        assert not marker.exists()
        os.kill(proc.pid, signal.SIGKILL)       # not reaped yet: still a zombie
        assert _wait_for(marker, 5)
    finally:
        proc.kill(); proc.wait()


def _dead_pid() -> int:
    p = subprocess.Popen(["/usr/bin/true"])
    p.wait()
    return p.pid


def test_watchdog_does_nothing_if_digits_was_restarted_since_the_call(tmp_path):
    """Bot gone AND time limit passed, but the DIGITS process that held the call no longer exists:
    whatever is in the app now is not the assistant's call."""
    marker = tmp_path / "fired"
    argv = _watchdog_argv(str(_dead_pid()), marker, max_seconds="-14") + ["--digits-pid", str(_dead_pid())]
    assert subprocess.run(argv, cwd=ROOT, timeout=10).returncode == 0
    assert not marker.exists()


def test_watchdog_acts_when_that_digits_process_is_still_running(tmp_path):
    marker = tmp_path / "fired"
    argv = _watchdog_argv(str(_dead_pid()), marker) + ["--digits-pid", str(os.getpid())]
    assert subprocess.run(argv, cwd=ROOT, timeout=10).returncode == 0
    assert marker.exists()


def test_watchdog_does_nothing_when_the_call_ends_normally(tmp_path):
    marker = tmp_path / "fired"
    wd = subprocess.Popen(_watchdog_argv(str(os.getpid()), marker), cwd=ROOT)
    time.sleep(0.8)
    wd.terminate()
    assert wd.wait(timeout=3) == 0
    time.sleep(0.3)
    assert not marker.exists()


def test_watchdog_fires_at_the_hard_time_limit(tmp_path):
    marker = tmp_path / "fired"
    wd = subprocess.Popen(_watchdog_argv(str(os.getpid()), marker, max_seconds="-14"), cwd=ROOT)  # limit = 1 s
    try:
        assert _wait_for(marker, 5)
    finally:
        wd.wait(timeout=5)


def test_watchdog_with_no_time_limit_waits_but_still_fires_if_the_bot_dies(tmp_path):
    marker = tmp_path / "fired"
    wd = subprocess.Popen(_watchdog_argv(str(os.getpid()), marker, max_seconds="0"), cwd=ROOT)
    try:
        time.sleep(2.0)
        assert wd.poll() is None and not marker.exists()        # 0 = no limit: still just watching
    finally:
        wd.terminate(); wd.wait(timeout=3)
    marker2 = tmp_path / "fired2"
    wd = subprocess.Popen(_watchdog_argv(str(_dead_pid()), marker2, max_seconds="0"), cwd=ROOT)
    try:
        assert _wait_for(marker2, 5)                            # bot gone: acts at once
    finally:
        wd.wait(timeout=5)


# ------------------------------------------------------------ config + code guardrails
BASE_ENV = "DISCORD_TOKEN=x\nOWNER_USER_ID=1\nGUILD_ID=2\nASSISTANT_CHANNEL_ID=3\nCALLS_CHANNEL_ID=4\n"


def _load(tmp_path, monkeypatch, extra):
    from phone_assistant.config import load_settings
    for k in ["PHONE_BACKEND", "ALLOW_REAL_ANSWER", "DIGITS_AUTOLAUNCH"]:
        monkeypatch.delenv(k, raising=False)
    env = tmp_path / ".env"
    env.write_text(BASE_ENV + extra)
    return load_settings(env)


def test_native_observe_needs_no_extra_switch(tmp_path, monkeypatch):
    s = _load(tmp_path, monkeypatch, "PHONE_BACKEND=digits_native_observe\n")
    assert s.phone_backend == "digits_native_observe" and s.digits_autolaunch is True
    assert _load(tmp_path, monkeypatch, "PHONE_BACKEND=digits_native_observe\nDIGITS_AUTOLAUNCH=no\n").digits_autolaunch is False


def test_native_answer_requires_both_switches(tmp_path, monkeypatch):
    from phone_assistant.config import ConfigError
    with pytest.raises(ConfigError, match="ALLOW_REAL_ANSWER=yes"):
        _load(tmp_path, monkeypatch, "PHONE_BACKEND=digits_native_answer\n")
    with pytest.raises(ConfigError):
        _load(tmp_path, monkeypatch, "PHONE_BACKEND=digits_native_answer\nALLOW_REAL_ANSWER=no\n")
    assert _load(tmp_path, monkeypatch,
                 "PHONE_BACKEND=digits_native_answer\nALLOW_REAL_ANSWER=yes\n").phone_backend == "digits_native_answer"


def test_unknown_backend_and_default(tmp_path, monkeypatch):
    from phone_assistant.config import ConfigError
    with pytest.raises(ConfigError):
        _load(tmp_path, monkeypatch, "PHONE_BACKEND=digits_native\n")
    assert _load(tmp_path, monkeypatch, "").phone_backend == "simulated"   # safe default


def test_current_env_matches_the_owners_decisions():
    """Owner decision 2026-11-21: answering stays on (manual `answer` only). Outgoing calls stay off
    until separately approved. Only the named keys are read, so a failure can never print a secret."""
    env = ROOT / ".env"
    if not env.exists():
        pytest.skip("no .env")
    want = ("PHONE_BACKEND", "ALLOW_REAL_ANSWER", "ALLOW_OUTGOING")
    got = {}
    for line in env.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() in want:
            got[key.strip()] = value.strip().lower()
    backend = got.get("PHONE_BACKEND", "simulated")
    assert backend in ("digits_native_answer", "digits_native_observe", "simulated")
    assert (got.get("ALLOW_REAL_ANSWER") == "yes") == (backend == "digits_native_answer")


def test_press_sites_are_exactly_accept_and_end():
    src = inspect.getsource(dn)
    assert src.count(".press,") == 2          # self._ax(self.source.press, <node>)
    assert src.count(".press(") == 0
    assert inspect.getsource(DigitsNativeAnswerBackend.answer).count(".press,") == 1
    assert inspect.getsource(DigitsNativeAnswerBackend._end_call).count(".press,") == 1
    assert "now_alert.accept" in inspect.getsource(DigitsNativeAnswerBackend.answer)
    assert "REJECT_LABEL" not in inspect.getsource(DigitsNativeAnswerBackend)


def test_native_code_has_one_action_primitive_and_no_synthetic_input():
    import phone_assistant.phone.native_ax as ax
    import phone_assistant.phone.native_watchdog as wd
    ax_src = inspect.getsource(ax)
    assert ax_src.count("AXUIElementPerformAction(") == 1
    for src in (ax_src, inspect.getsource(dn), inspect.getsource(wd)):
        for forbidden in ("CGEvent", "keystroke ", "key code", "System Events", "osascript", "AXConfirm",
                          "Cookies", "Local Storage", "password", "pyautogui", "Quartz"):
            assert forbidden not in src, forbidden
    assert "PerformAction" not in inspect.getsource(wd) and ".press" not in inspect.getsource(wd)


def test_nothing_in_the_native_code_can_bring_digits_to_the_front():
    import phone_assistant.phone.native_ax as ax
    import phone_assistant.phone.native_watchdog as wd
    for src in (inspect.getsource(ax), inspect.getsource(dn), inspect.getsource(wd)):
        for forbidden in ("activateIgnoringOtherApps", "AXRaise", '"AXMain"', "setActivates_(True)", '"-a"', "unhide("):
            assert forbidden not in src, forbidden
    ax_src = inspect.getsource(ax)
    assert "setActivates_(False)" in ax_src and '"/usr/bin/open", "-g"' in ax_src       # background requests only
    assert "bundleIdentifier() == self.bundle_id" in ax_src                    # focus hand-back refuses DIGITS itself


def test_logs_never_contain_caller_details(rig_logs):
    text = rig_logs
    assert "555" not in text and "Jane" not in text and "Doe" not in text
    assert "new controls" in text and "<caller>" in text


@pytest.fixture
async def rig_logs(rig, clock, caplog):
    import logging
    app, b, m, tick = rig
    with caplog.at_level(logging.DEBUG):
        await ring(app, tick)
        await m.answer(OWNER, clock.advance(2))
        await m.hangup(OWNER, clock.advance(2))
        app.end_label = "Leave call with Jane Doe (617) 555-0188"
        await ring(app, tick, "Jane Doe", "(617) 555-0188")
        await m.answer(OWNER, clock.advance(2))
        await tick()
        await m.hangup(OWNER, clock.advance(2))
    return "\n".join(r.getMessage() for r in caplog.records if r.name.startswith("phone_assistant.phone"))
