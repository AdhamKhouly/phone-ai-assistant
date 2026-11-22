"""Phase 2b safety suite: real headless Chrome against a fake DIGITS page.

The fake page reproduces the captured DIGITS structure and counts every Accept/End click
(window.__accepts / window.__ends). Each rejection rule is verified by asserting that the
Accept button was NEVER clicked.
"""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from phone_assistant.calls.manager import CallManager
from phone_assistant.calls.models import CallState
from phone_assistant.control import ControlService
from phone_assistant.phone.digits_answer import DigitsAnswerBackend

from .conftest import ASSISTANT_CH, CALLS_CH, OWNER, STRANGER

pw_api = pytest.importorskip("playwright.async_api")

FAKE_DIGITS = """
<div id="root"><div class="fullscreen column">
  <button>All phone numbers</button><button><span>New call</span></button>
  <div id="slot"></div></div></div>
<script>
  window.__accepts = 0; window.__ends = 0;
  const slot = document.getElementById('slot');
  const alertHtml = (name, num) => `<div role="alert" data-testid="name-contact"
      aria-label="Incoming audio call from ${name}. Press Enter to accept">
      <div>${name[0]}</div><div>Incoming audio call...</div><div>${name}</div><div>${num}</div>
      <span class="incomingCallOptions"><div><div>
        <button aria-label="Reject call" title="Reject call">Reject</button>
        <button aria-label="Accept call" title="Accept call" onclick="accept()">Accept</button>
      </div></div></span></div>`;
  window.ring = (name, num) => { slot.innerHTML = alertHtml(name, num); };
  window.ringTwo = () => { slot.innerHTML = alertHtml('A', '(617) 555-0001') + alertHtml('B', '(617) 555-0002'); };
  window.stopRing = () => { slot.innerHTML = ''; };
  window.inCallLabel = 'End call';
  window.accept = () => {
    window.__accepts++;
    if (window.brokenAccept) return;
    slot.innerHTML = `<div id="incall"><button aria-label="${window.inCallLabel}" onclick="endCall()">x</button></div>`;
  };
  window.endCall = () => { window.__ends++; if (!window.brokenEnd) slot.innerHTML = ''; };
  window.remoteHangup = () => { slot.innerHTML = ''; };
</script>
"""


@pytest.fixture
async def browser():
    try:
        async with pw_api.async_playwright() as pw:
            b = await pw.chromium.launch(channel="chrome", headless=True)
            yield b
            await b.close()
    except Exception as e:  # pragma: no cover
        if "Executable doesn't exist" in str(e):
            pytest.skip("Chrome not available")
        raise


@pytest.fixture
async def rig(browser, clock, events):
    page = await (await browser.new_context()).new_page()
    await page.set_content(FAKE_DIGITS)
    b = DigitsAnswerBackend(keep_awake=False)
    b.page, b.ctx, b.home_url = page, page.context, "about:blank"
    b.unavailable_grace_s = 0
    m = CallManager(b, OWNER, answer_window_s=22, max_command_age_s=15, max_call_s=60, clock=clock)

    async def rec(ev):
        events.append(ev)
    m.subscribe(rec)

    async def tick():
        await b.apply_snapshot(await b._snap())

    await tick()  # connected
    return page, b, m, tick


async def accepts(page) -> int:
    return await page.evaluate("window.__accepts")


async def ring(page, tick, name="Jane Doe", num="(617) 555-0123"):
    await page.evaluate(f"ring({name!r}, {num!r})")
    await tick()


# ---------------------------------------------------------------- the happy path
async def test_owner_answer_clicks_accept_once_then_hangup(rig, clock):
    page, b, m, tick = rig
    await ring(page, tick)
    call = list(m.calls.values())[0]
    r = await m.answer(OWNER, clock.advance(3))
    assert r.ok and call.state is CallState.ACTIVE
    assert await accepts(page) == 1
    await tick()
    r = await m.hangup(OWNER, clock.advance(5))
    assert r.ok and call.state is CallState.ENDED and call.end_reason == "owner_hangup"
    assert await page.evaluate("window.__ends") == 1


async def test_plain_text_answer_through_discord_layer(rig, clock):
    page, b, m, tick = rig
    control = ControlService(m, OWNER, {ASSISTANT_CH, CALLS_CH})
    await ring(page, tick)
    reply = await control.handle(OWNER, CALLS_CH, "answer", clock.advance(2))
    assert reply.startswith("Answering call #1") and await accepts(page) == 1


# ---------------------------------------------------------------- never without a valid command
async def test_no_command_never_answers(rig, clock, events):
    page, b, m, tick = rig
    await ring(page, tick)
    for _ in range(5):
        await tick()
    clock.advance(23)
    await m.expire_stale()
    await page.evaluate("stopRing()")
    await tick()
    assert await accepts(page) == 0
    assert [e.kind for e in events if e.kind != "backend"] == ["incoming", "missed", "stopped"]


async def test_unauthorized_user_never_clicks(rig, clock):
    page, b, m, tick = rig
    control = ControlService(m, OWNER, {ASSISTANT_CH, CALLS_CH})
    await ring(page, tick)
    assert "only the owner" in await control.handle(STRANGER, CALLS_CH, "answer", clock.advance(1))
    assert "only the owner" in await control.handle(STRANGER, CALLS_CH, "/answer", clock.now)
    assert await accepts(page) == 0


async def test_command_sent_before_ring_never_clicks(rig, clock):
    page, b, m, tick = rig
    queued = clock.now                      # e.g. typed before a restart / before this call
    clock.advance(2)
    await ring(page, tick)
    assert not (await m.answer(OWNER, queued)).ok
    assert await accepts(page) == 0


async def test_delayed_command_never_clicks(rig, clock):
    page, b, m, tick = rig
    m.answer_window = m.answer_window * 10
    await ring(page, tick)
    issued = clock.advance(1)
    clock.advance(20)                       # delivered 20 s late
    assert not (await m.answer(OWNER, issued)).ok
    assert await accepts(page) == 0


async def test_expired_call_never_clicks(rig, clock):
    page, b, m, tick = rig
    await ring(page, tick)
    clock.advance(23)
    assert not (await m.answer(OWNER, clock.now)).ok
    assert await accepts(page) == 0


async def test_duplicate_simultaneous_answers_click_once(rig, clock):
    page, b, m, tick = rig
    await ring(page, tick)
    t = clock.advance(2)
    results = await asyncio.gather(m.answer(OWNER, t), m.answer(OWNER, t), m.answer(OWNER, t))
    assert sum(r.ok for r in results) == 1
    assert await accepts(page) == 1


async def test_answer_after_ignore_never_clicks(rig, clock):
    page, b, m, tick = rig
    await ring(page, tick)
    await m.ignore(OWNER, clock.advance(1))
    assert not (await m.answer(OWNER, clock.advance(1))).ok
    assert await accepts(page) == 0


# ---------------------------------------------------------------- verify-immediately-before-click
async def test_caller_changed_before_click_is_refused(rig, clock):
    page, b, m, tick = rig
    await ring(page, tick, "Jane Doe", "(617) 555-0123")
    # Page now shows a DIFFERENT caller, and the observer hasn't ticked yet:
    await page.evaluate("ring('Someone Else', '(617) 555-0999')")
    r = await m.answer(OWNER, clock.advance(2))
    assert not r.ok and await accepts(page) == 0


async def test_alert_gone_before_click_is_refused(rig, clock):
    page, b, m, tick = rig
    await ring(page, tick)
    await page.evaluate("stopRing()")      # caller hung up a moment before `answer` arrived
    r = await m.answer(OWNER, clock.advance(2))
    assert not r.ok and await accepts(page) == 0


async def test_two_alerts_at_once_is_refused(rig, clock):
    page, b, m, tick = rig
    await ring(page, tick)
    await page.evaluate("ringTwo()")
    r = await m.answer(OWNER, clock.advance(2))
    assert not r.ok and await accepts(page) == 0


async def test_accept_that_does_nothing_fails_safely(rig, clock):
    page, b, m, tick = rig
    await page.evaluate("window.brokenAccept = true")
    await ring(page, tick)
    r = await m.answer(OWNER, clock.advance(2))
    call = list(m.calls.values())[0]
    assert not r.ok and call.state is CallState.FAILED
    assert await accepts(page) == 1        # one attempt, no retries


# ---------------------------------------------------------------- ending calls
async def test_remote_hangup_is_detected(rig, clock):
    page, b, m, tick = rig
    await ring(page, tick)
    await m.answer(OWNER, clock.advance(2))
    await tick()
    await page.evaluate("remoteHangup()")
    await tick()
    call = list(m.calls.values())[0]
    assert call.state is CallState.ENDED and call.end_reason == "remote_hangup"


async def test_unrecognised_in_call_view_hangup_falls_back_to_closing_tab(rig, clock):
    page, b, m, tick = rig
    await page.evaluate("window.inCallLabel = 'Disconnect'")   # unknown label
    await ring(page, tick)
    assert (await m.answer(OWNER, clock.advance(2))).ok
    await tick()
    r = await m.hangup(OWNER, clock.advance(3))
    assert r.ok and page.is_closed()        # fallback: the DIGITS tab was closed
    assert b.page is not page               # and a fresh tab was opened


async def test_max_call_duration_forces_hangup(rig, clock, events):
    page, b, m, tick = rig
    await ring(page, tick)
    await m.answer(OWNER, clock.advance(2))
    await tick()
    clock.advance(61)
    ended = await m.enforce_max_duration()
    assert len(ended) == 1 and ended[0].end_reason == "max_duration"
    assert await page.evaluate("window.__ends") == 1


async def test_orphan_in_call_view_after_restart_is_ended_not_answered(browser, clock):
    """Bot restarts while DIGITS still shows a connected call: end it, never 'adopt' it."""
    page = await (await browser.new_context()).new_page()
    await page.set_content(FAKE_DIGITS)
    await page.evaluate("ring('X', '(617) 555-0100'); accept()")   # connected call, no owner
    b = DigitsAnswerBackend(keep_awake=False)
    b.page, b.ctx, b.home_url = page, page.context, "about:blank"
    m = CallManager(b, OWNER, clock=clock)
    await b.apply_snapshot(await b._snap())
    assert await page.evaluate("window.__ends") == 1
    assert m.calls == {}


async def test_restart_with_ringing_call_needs_a_new_answer(browser, clock):
    """A ringing call seen after restart is a NEW call; an `answer` from before the restart is stale."""
    before_restart = clock.now
    clock.advance(5)
    page = await (await browser.new_context()).new_page()
    await page.set_content(FAKE_DIGITS)
    await page.evaluate("ring('Jane Doe', '(617) 555-0123')")
    b = DigitsAnswerBackend(keep_awake=False)
    b.page, b.ctx = page, page.context
    m = CallManager(b, OWNER, answer_window_s=22, clock=clock)
    await b.apply_snapshot(await b._snap())
    assert not (await m.answer(OWNER, before_restart)).ok
    assert await page.evaluate("window.__accepts") == 0


# ---------------------------------------------------------------- config + code guardrails
def test_answer_mode_requires_both_switches(tmp_path, monkeypatch):
    from phone_assistant.config import ConfigError, load_settings
    for k in ["PHONE_BACKEND", "ALLOW_REAL_ANSWER"]:
        monkeypatch.delenv(k, raising=False)
    base = "DISCORD_TOKEN=x\nOWNER_USER_ID=1\nGUILD_ID=2\nASSISTANT_CHANNEL_ID=3\nCALLS_CHANNEL_ID=4\n"
    env = tmp_path / ".env"
    env.write_text(base + "PHONE_BACKEND=digits_answer\n")
    with pytest.raises(ConfigError, match="ALLOW_REAL_ANSWER=yes"):
        load_settings(env)
    monkeypatch.delenv("PHONE_BACKEND", raising=False)
    env.write_text(base + "PHONE_BACKEND=digits_answer\nALLOW_REAL_ANSWER=YES please\n")
    with pytest.raises(ConfigError):
        load_settings(env)
    monkeypatch.delenv("PHONE_BACKEND", raising=False)
    monkeypatch.delenv("ALLOW_REAL_ANSWER", raising=False)
    env.write_text(base + "PHONE_BACKEND=digits_answer\nALLOW_REAL_ANSWER=yes\n")
    assert load_settings(env).phone_backend == "digits_answer"


def test_click_sites_are_exactly_accept_and_end():
    import inspect
    import phone_assistant.phone.digits_answer as da
    src = inspect.getsource(da)
    assert src.count(".click(") == 2
    for forbidden in (".press(", ".type(", ".fill(", "keyboard", "mouse", "dispatch_event", "route("):
        assert forbidden not in src, forbidden
    answer_src = inspect.getsource(da.DigitsAnswerBackend.answer)
    end_src = inspect.getsource(da.DigitsAnswerBackend._end_call)
    assert answer_src.count(".click(") == 1 and end_src.count(".click(") == 1


def test_current_env_keeps_real_answering_off():
    """The project's actual .env must not have real answering enabled until you approve it."""
    from pathlib import Path
    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        pytest.skip("no .env")
    lines = dict(l.split("=", 1) for l in env.read_text().splitlines() if "=" in l and not l.startswith("#"))
    assert not (lines.get("PHONE_BACKEND") == "digits_answer" and lines.get("ALLOW_REAL_ANSWER", "").strip() == "yes")


# ---------------------------------------------------------------- crash behaviour
def test_hard_crash_takes_chrome_down(tmp_path):
    """kill -9 of the Python process must not leave a Chrome (and thus a call) running."""
    profile = tmp_path / "profile"
    script = tmp_path / "launch.py"
    script.write_text(textwrap.dedent(f"""
        import asyncio
        from playwright.async_api import async_playwright
        async def main():
            async with async_playwright() as pw:
                ctx = await pw.chromium.launch_persistent_context({str(profile)!r}, channel="chrome", headless=True)
                print("READY", flush=True)
                await asyncio.sleep(3600)
        asyncio.run(main())
    """))
    proc = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "READY"
        count = lambda: len(subprocess.run(["pgrep", "-f", str(profile)], capture_output=True, text=True).stdout.split())
        assert count() > 0
        os.kill(proc.pid, signal.SIGKILL)
        deadline = time.time() + 10
        while time.time() < deadline and count() > 0:
            time.sleep(0.25)
        assert count() == 0, "Chrome survived the crash"
    finally:
        subprocess.run(["pkill", "-9", "-f", str(profile)])


async def test_graceful_shutdown_during_answered_call_ends_it(rig, clock):
    page, b, m, tick = rig
    await ring(page, tick)
    await m.answer(OWNER, clock.advance(2))
    await tick()
    released = await m.shutdown()
    assert len(released) == 1 and released[0].state is CallState.ENDED
    assert await page.evaluate("window.__ends") == 1
