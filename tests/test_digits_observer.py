"""DIGITS observe-only backend: state handling, caller parsing, and the real DOM query.

The fixture HTML reproduces the structure captured from a real ringing call on 2026-11-20
(role=alert + data-testid=name-contact + aria-labelled Accept/Reject buttons).
"""
from __future__ import annotations

import asyncio
import shutil

import pytest

from phone_assistant.calls.manager import CallManager
from phone_assistant.calls.models import CallState
from phone_assistant.phone.digits import SNAPSHOT_JS, DigitsObserver, parse_caller

from .conftest import OWNER

IDLE = {"loggedIn": True, "loggedOut": False, "ringing": False, "label": None, "body": None}
RING = {"loggedIn": True, "loggedOut": False, "ringing": True,
        "label": "Incoming audio call from Jane Doe. Press Enter to accept",
        "body": "J Incoming audio call... Jane Doe (617) 555-0123 Reject Accept"}
LOGGED_OUT = {"loggedIn": False, "loggedOut": True, "ringing": False, "label": None, "body": None}


@pytest.fixture
def obs(clock, events):
    b = DigitsObserver(keep_awake=False)
    b.unavailable_grace_s = 0  # page loss counts immediately in these tests
    m = CallManager(b, OWNER, answer_window_s=22, max_command_age_s=15, clock=clock)

    async def rec(ev):
        events.append(ev)
    m.subscribe(rec)
    return b, m


def test_parse_contact_caller():
    c = parse_caller(RING["label"], RING["body"])
    assert c.number == "+16175550123" and c.name == "Jane Doe"


def test_parse_unknown_caller_number_only():
    c = parse_caller("Incoming audio call from (617) 555-0199. Press Enter to accept",
                     "( Incoming audio call... (617) 555-0199 Reject Accept")
    assert c.number == "+16175550199" and c.name is None


def test_parse_withheld_caller():
    c = parse_caller("Incoming audio call from Unknown. Press Enter", "U Incoming audio call... Unknown Reject Accept")
    assert c.number == "Unknown" and c.name is None


async def test_ring_then_stop_reports_incoming_then_missed(obs, events):
    b, m = obs
    await b.apply_snapshot(IDLE)
    assert b.connected and events[-1].kind == "backend"
    await b.apply_snapshot(RING)
    await b.apply_snapshot(RING)  # still ringing: no duplicate event
    calls = list(m.calls.values())
    assert len(calls) == 1 and calls[0].remote_name == "Jane Doe" and calls[0].state is CallState.RINGING
    await b.apply_snapshot(IDLE)
    assert calls[0].state is CallState.MISSED and calls[0].end_reason == "stopped_ringing"
    assert [e.kind for e in events] == ["backend", "incoming", "missed"]


async def test_answer_is_refused_and_call_left_ringing(obs, clock):
    b, m = obs
    await b.apply_snapshot(IDLE)
    await b.apply_snapshot(RING)
    r = await m.answer(OWNER, clock.advance(2))
    assert not r.ok and "observe-only" in r.message
    assert list(m.calls.values())[0].state is CallState.RINGING
    r = await m.dial(OWNER, clock.now, "+16175550000")
    assert not r.ok and "observe-only" in r.message


async def test_logout_and_page_loss_release_calls(obs, events):
    b, m = obs
    await b.apply_snapshot(IDLE)
    await b.apply_snapshot(RING)
    await b.apply_snapshot(LOGGED_OUT)
    call = list(m.calls.values())[0]
    assert not b.connected and call.state is CallState.MISSED and call.end_reason == "backend_disconnected"
    await b.apply_snapshot(IDLE)
    await b.apply_snapshot(None)  # tab crashed / navigating
    assert not b.connected
    kinds = [e.detail for e in events if e.kind == "backend"]
    assert kinds == ["connected", "disconnected", "connected", "disconnected"]


async def test_new_ring_after_previous_gets_new_call(obs):
    b, m = obs
    for snap in (IDLE, RING, IDLE, RING):
        await b.apply_snapshot(snap)
    assert len(m.calls) == 2


def test_observer_source_never_clicks_or_types():
    """Guardrail: the observe-only backend must contain no input-producing Playwright calls."""
    import inspect
    import phone_assistant.phone.digits as d
    src = inspect.getsource(d)
    for forbidden in (".click(", ".press(", ".type(", ".fill(", "keyboard", "mouse", "dispatch_event", "route("):
        assert forbidden not in src, forbidden


# ---- real Chrome, headless, against a local copy of the DIGITS alert structure -----------
FIXTURE_LOGGED_IN = """<div id="root"><div class="fullscreen column">
<button>All phone numbers</button><button><span>New call</span></button></div></div>"""
FIXTURE_RINGING = """<div id="root"><div class="fullscreen column">
<button>All phone numbers</button><button><span>New call</span></button>
<div class="_1crj76o styles__container--jkHib" role="alert" data-testid="name-contact"
     aria-label="Incoming audio call from Jane Doe. Press Enter to accept">
  <div>J</div><div>Incoming audio call...</div><div>Jane Doe</div><div>(617) 555-0123</div>
  <span class="incomingCallOptions"><div><div class="styles__iconRow--Egg7k">
    <button class="styles__iconButton--R_7DU" aria-label="Reject call" title="Reject call">Reject</button>
    <button class="styles__iconButton--R_7DU" aria-label="Accept call" title="Accept call">Accept</button>
  </div></div></span></div></div></div>"""
FIXTURE_LANDING = """<div id="root"><h1>Free your phone and your number with DIGITS</h1>
<button aria-label="Press Enter to log in to Digits from T-Mobile!">Log In</button></div>"""


@pytest.mark.skipif(shutil.which("true") is None, reason="posix only")
async def test_snapshot_js_against_real_chrome():
    pw_mod = pytest.importorskip("playwright.async_api")
    try:
        async with pw_mod.async_playwright() as pw:
            browser = await pw.chromium.launch(channel="chrome", headless=True)
            page = await browser.new_page()
            await page.set_content(FIXTURE_LOGGED_IN)
            s = await page.evaluate(SNAPSHOT_JS)
            assert s["loggedIn"] and not s["ringing"]
            await page.set_content(FIXTURE_RINGING)
            s = await page.evaluate(SNAPSHOT_JS)
            assert s["loggedIn"] and s["ringing"]
            c = parse_caller(s["label"], s["body"])
            assert c.number == "+16175550123" and c.name == "Jane Doe"
            await page.set_content(FIXTURE_LANDING)
            s = await page.evaluate(SNAPSHOT_JS)
            assert not s["loggedIn"] and s["loggedOut"] and not s["ringing"]
            await browser.close()
    except Exception as e:  # pragma: no cover
        if "Executable doesn't exist" in str(e) or "channel" in str(e).lower():
            pytest.skip(f"Chrome not available to Playwright: {e}")
        raise


async def test_window_close_and_real_stop_are_separate_notices(obs, clock, events):
    from phone_assistant.discord_bot.notifications import format_event
    b, m = obs
    await b.apply_snapshot(IDLE)
    await b.apply_snapshot(RING)
    clock.advance(23)
    await m.expire_stale()                       # answer window (22 s) closes
    clock.advance(3)
    await b.apply_snapshot(IDLE)                 # DIGITS actually stops ringing at 26 s
    kinds = [e.kind for e in events if e.kind != "backend"]
    assert kinds == ["incoming", "missed", "stopped"]
    texts = [format_event(e, clock.now, 22, answer_enabled=False) for e in events if e.kind != "backend"]
    assert "Answer window closed" in texts[1]
    assert "stopped ringing after 26s" in texts[2]
    await b.apply_snapshot(RING)                 # a later call is a new call, not a duplicate
    assert len(m.calls) == 2


async def test_ignored_call_still_reports_real_stop(obs, clock, events):
    b, m = obs
    await b.apply_snapshot(IDLE)
    await b.apply_snapshot(RING)
    await m.ignore(OWNER, clock.advance(1))
    await b.apply_snapshot(IDLE)
    assert [e.kind for e in events if e.kind != "backend"] == ["incoming", "ignored", "stopped"]


async def test_starting_signed_out_warns_once(obs, events):
    b, m = obs
    await b.apply_snapshot(LOGGED_OUT)
    await b.apply_snapshot(LOGGED_OUT)
    assert [(e.kind, e.detail) for e in events] == [("backend", "disconnected")]
    await b.apply_snapshot(IDLE)
    assert events[-1].detail == "connected"
