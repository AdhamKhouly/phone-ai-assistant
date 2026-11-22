"""Virtual audio routing, the microphone guard, the audio hook on answering, and the call voice
session. Everything here uses a fake Core Audio and fake streams: no real device is touched."""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import phone_assistant.phone.digits_native as dn
from phone_assistant.audio.routing import CallAudioRoute, MicGuard, RouteConfig, check_devices, restore_stale
from phone_assistant.calls.models import CallState
from phone_assistant.calls.voice_session import VoiceSession
from phone_assistant.config import ConfigError, load_settings
from phone_assistant.conversation import prompt as P
from phone_assistant.phone.digits_native import DigitsNativeAnswerBackend

from .conftest import OWNER
from .test_digits_native import FakeDigitsApp, make, ring

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Dev:
    id: int
    name: str
    uid: str
    transport: str
    has_input: bool = True
    has_output: bool = True
    running: bool = False

    @property
    def is_virtual(self): return self.transport == "virtual"

    @property
    def is_physical_input(self): return self.has_input and not self.is_virtual


@dataclass(frozen=True)
class Proc:
    pid: int
    bundle_id: str
    running_input: bool
    running_output: bool
    input_devices: tuple
    output_devices: tuple


class FakeAudio:
    def __init__(self, blackhole=True):
        self.devs = [Dev(1, "MacBook Pro Microphone", "mic", "built-in", True, False),
                     Dev(2, "MacBook Pro Speakers", "spk", "built-in", False, True)]
        if blackhole:
            self.devs += [Dev(3, "BlackHole 2ch", "bh2", "virtual"), Dev(4, "BlackHole 16ch", "bh16", "virtual")]
        self.din, self.dout = "mic", "spk"
        self.procs: list[Proc] = []
        self.sets: list[tuple[str, str]] = []
        self.fail_set = False
        self.ignore_set = False

    def devices(self): return list(self.devs)
    def find_device(self, name): return next((d for d in self.devs if d.name == name), None)
    def _uid(self, uid): return next((d for d in self.devs if d.uid == uid), None)
    def default_input(self): return self._uid(self.din)
    def default_output(self): return self._uid(self.dout)
    def audio_for_bundle(self, prefix): return [p for p in self.procs if p.bundle_id.startswith(prefix)]

    def set_default_input(self, d):
        if self.fail_set: raise RuntimeError("nope")
        self.sets.append(("in", d.name))
        if not self.ignore_set: self.din = d.uid

    def set_default_output(self, d):
        if self.fail_set: raise RuntimeError("nope")
        self.sets.append(("out", d.name))
        if not self.ignore_set: self.dout = d.uid

    def set_running(self, name, running=True):
        self.devs = [Dev(d.id, d.name, d.uid, d.transport, d.has_input, d.has_output, running if d.name == name else d.running)
                     for d in self.devs]


@pytest.fixture
def audio(): return FakeAudio()


@pytest.fixture
def route(audio, tmp_path): return CallAudioRoute(RouteConfig(), api=audio, restore_file=tmp_path / "restore.json")


# ------------------------------------------------------------------ routing
def test_route_switches_defaults_and_puts_them_back(route, audio):
    assert route.engage() is None and route.engaged
    assert (audio.din, audio.dout) == ("bh2", "bh16")
    assert json.loads(route.restore_file.read_text()) == {"input": "mic", "output": "spk"}
    route.release()
    assert (audio.din, audio.dout) == ("mic", "spk") and not route.restore_file.exists() and not route.engaged


def test_missing_virtual_devices_change_nothing(tmp_path):
    a = FakeAudio(blackhole=False)
    r = CallAudioRoute(RouteConfig(), api=a, restore_file=tmp_path / "r.json")
    why = r.engage()
    assert why and "not installed" in why
    assert a.sets == [] and not r.engaged and not (tmp_path / "r.json").exists()


def test_a_physical_device_can_never_be_the_route(audio, tmp_path):
    cfg = RouteConfig(mic="MacBook Pro Microphone", capture="BlackHole 16ch")
    assert any("not a virtual device" in p for p in check_devices(cfg, audio))
    r = CallAudioRoute(cfg, api=audio, restore_file=tmp_path / "r.json")
    assert r.engage() and audio.sets == []


def test_same_device_for_both_directions_is_refused(audio):
    assert check_devices(RouteConfig(mic="BlackHole 2ch", capture="BlackHole 2ch"), audio)


def test_switch_failure_restores_and_reports(route, audio):
    audio.fail_set = True
    assert "could not switch" in route.engage()
    assert (audio.din, audio.dout) == ("mic", "spk") and not route.engaged and not route.restore_file.exists()


def test_switch_that_does_not_take_effect_is_a_failure(route, audio):
    audio.ignore_set = True
    assert "did not accept" in route.engage()
    assert not route.engaged


def test_crash_recovery_restores_from_file(route, audio):
    route.engage()
    assert restore_stale(api=audio, path=route.restore_file) is True      # as a fresh process would
    assert (audio.din, audio.dout) == ("mic", "spk") and not route.restore_file.exists()
    assert restore_stale(api=audio, path=route.restore_file) is False     # nothing left to do


def test_release_without_engage_changes_nothing(route, audio):
    route.release()
    assert audio.sets == []


# ------------------------------------------------------------------ microphone guard
def test_guard_ok_when_digits_uses_virtual_mic(route, audio):
    route.engage()
    audio.procs = [Proc(5, "com.tmobile.phone2.helper", True, True, ("BlackHole 2ch",), ("BlackHole 16ch",))]
    g = MicGuard(api=audio)
    assert g.violation() is None and g.digits_uses_virtual_mic()


def test_guard_trips_when_default_input_is_changed(route, audio):
    route.engage()
    audio.din = "mic"                                   # someone/something switched the input back
    assert "input device changed" in MicGuard(api=audio).violation()


def test_guard_trips_when_digits_opens_the_builtin_mic(route, audio):
    route.engage()
    audio.procs = [Proc(5, "com.tmobile.phone2.helper", True, True, ("MacBook Pro Microphone",), ())]
    assert "not the virtual microphone" in MicGuard(api=audio).violation()


def test_guard_assumes_the_worst_without_per_app_detail(route, audio):
    route.engage()
    audio.set_running("MacBook Pro Microphone")
    assert "physical microphone is active" in MicGuard(api=audio).violation()
    # ...but another app's microphone use is not blamed on DIGITS when DIGITS is shown on the virtual mic
    audio.procs = [Proc(5, "com.tmobile.phone2", True, True, ("BlackHole 2ch",), ())]
    assert MicGuard(api=audio).violation() is None


def test_guard_fails_closed_when_audio_state_unreadable(route, audio):
    route.engage()
    audio.devices = lambda: (_ for _ in ()).throw(OSError("x"))
    assert "could not be read" in MicGuard(api=audio).violation()


def test_guard_trips_before_routing_is_engaged(audio):
    assert MicGuard(api=audio).violation() is not None   # default input is the real microphone


# ------------------------------------------------------------------ answering requires the route
@pytest.fixture
async def arig(clock, events, monkeypatch, route):
    monkeypatch.setattr(dn, "CONFIRM_TIMEOUT_S", 1.0)
    monkeypatch.setattr(dn, "SIGNATURE_SETTLE_S", 0.0)
    app = FakeDigitsApp()
    b, m, tick = make(DigitsNativeAnswerBackend, app, clock, events, audio_route=route)
    await tick()
    return app, b, m, tick


async def test_route_is_engaged_before_accept_and_released_on_hangup(arig, clock, route, audio):
    app, b, m, tick = arig
    await ring(app, tick)
    assert not route.engaged and audio.sets == []          # ringing alone changes nothing
    r = await m.answer(OWNER, clock.advance(3))
    assert r.ok and app.accepts == 1 and route.engaged and audio.din == "bh2"
    assert audio.sets[:2] == [("in", "BlackHole 2ch"), ("out", "BlackHole 16ch")]
    await tick()
    await m.hangup(OWNER, clock.advance(5))
    assert not route.engaged and (audio.din, audio.dout) == ("mic", "spk")


async def test_no_virtual_audio_means_no_answer(clock, events, monkeypatch, tmp_path):
    monkeypatch.setattr(dn, "CONFIRM_TIMEOUT_S", 1.0)
    a = FakeAudio(blackhole=False)
    app = FakeDigitsApp()
    b, m, tick = make(DigitsNativeAnswerBackend, app, clock, events,
                      audio_route=CallAudioRoute(RouteConfig(), api=a, restore_file=tmp_path / "r.json"))
    await tick()
    await ring(app, tick)
    r = await m.answer(OWNER, clock.advance(3))
    assert not r.ok and "not installed" in r.message
    assert app.call_actions == 0 and a.sets == []          # nothing pressed, nothing switched


async def test_failed_accept_restores_audio(arig, clock, route, audio):
    app, b, m, tick = arig
    await ring(app, tick)
    app.broken_accept = True
    r = await m.answer(OWNER, clock.advance(3))
    assert not r.ok and not route.engaged and (audio.din, audio.dout) == ("mic", "spk")


async def test_remote_hangup_restores_audio(arig, clock, route, audio):
    app, b, m, tick = arig
    await ring(app, tick)
    await m.answer(OWNER, clock.advance(3))
    await tick()
    app.in_call_label = None
    app.rings = []
    for _ in range(4):
        await tick()
    assert not m.active() and not route.engaged and audio.din == "mic"


async def test_no_answer_command_never_touches_audio(arig, audio):
    app, b, m, tick = arig
    await ring(app, tick)
    app.rings = []
    await tick()
    assert audio.sets == [] and app.call_actions == 0


# ------------------------------------------------------------------ voice session
class FakeIO:
    def __init__(self, out, fail_start=False):
        self.out, self.fail_start = out, fail_start
        self.started = self.stopped = False
        self.dropped_frames = self.status_events = 0
        self.healthy = True

    def start(self):
        if self.fail_start: raise RuntimeError("no device")
        self.started = True

    async def frames(self):
        while not self.stopped:
            await asyncio.sleep(0.01)
            yield np.zeros(512, dtype=np.float32)

    def stop(self): self.stopped = True


class FakePipe:
    def __init__(self):
        self.said, self.closed, self.turns, self.history = [], False, 0, []

    async def say(self, text): self.said.append(text)
    async def run(self, frames):
        async for _ in frames:
            pass
    def _housekeeping(self): pass
    async def close(self): self.closed = True


class FakeGuard:
    def __init__(self): self.why, self.checks, self.mic_open = None, 0, True
    def violation(self):
        self.checks += 1
        return self.why
    def digits_uses_virtual_mic(self): return self.mic_open


async def session_rig(arig, mode, monkeypatch, **kw):
    import phone_assistant.calls.voice_session as vs
    monkeypatch.setattr(vs, "GUARD_POLL_S", 0.02)
    monkeypatch.setattr(vs, "MIC_SETTLE_S", 0.0)
    app, b, m, tick = arig
    guard, pipe, ios, notes = FakeGuard(), FakePipe(), [], []

    def io_factory(out):
        ios.append(FakeIO(out, **kw)); return ios[-1]

    async def notify(text): notes.append(text)
    s = VoiceSession(m, mode=mode, guard=guard, io_factory=io_factory, notify=notify,
                     pipeline_factory=lambda out, on_event: (pipe, None))
    return s, guard, pipe, ios, notes


async def ended(call, timeout=3.0):
    """Wait for the hang-up (the fake app takes a few polls to confirm it)."""
    for _ in range(int(timeout / 0.02)):
        if call.state is CallState.ENDED:
            await asyncio.sleep(0.05)
            return True
        await asyncio.sleep(0.02)
    return False


async def answered(arig, clock):
    app, b, m, tick = arig
    await ring(app, tick)
    assert (await m.answer(OWNER, clock.advance(3))).ok
    await tick()
    await asyncio.sleep(0.08)
    return next(iter(m.calls.values()))


async def test_session_does_nothing_until_a_call_is_answered(arig, monkeypatch):
    s, guard, pipe, ios, notes = await session_rig(arig, "conversation", monkeypatch)
    app, b, m, tick = arig
    await ring(app, tick)
    await asyncio.sleep(0.05)
    assert ios == [] and pipe.said == [] and guard.checks == 0 and app.call_actions == 0


async def test_off_mode_guards_but_never_speaks_or_listens(arig, clock, monkeypatch):
    s, guard, pipe, ios, notes = await session_rig(arig, "off", monkeypatch)
    call = await answered(arig, clock)
    assert ios[0].started and pipe.said == [] and guard.checks >= 2
    await arig[2].hangup(OWNER, clock.advance(2))
    await asyncio.sleep(0.05)
    assert ios[0].stopped and call.state is CallState.ENDED


async def test_greeting_mode_speaks_the_fixed_greeting_only(arig, clock, monkeypatch):
    s, guard, pipe, ios, notes = await session_rig(arig, "greeting", monkeypatch)
    await answered(arig, clock)
    await asyncio.sleep(0.05)
    assert pipe.said == [P.GREETING]
    await arig[2].hangup(OWNER, clock.advance(2))
    await asyncio.sleep(0.05)
    assert pipe.closed and ios[0].stopped


async def test_greeting_waits_until_digits_has_opened_the_virtual_microphone(arig, clock, monkeypatch):
    import phone_assistant.calls.voice_session as vs
    monkeypatch.setattr(vs, "MIC_SETTLE_S", 0.0)
    s, guard, pipe, ios, notes = await session_rig(arig, "greeting", monkeypatch)
    guard.mic_open = False
    s.greeting = "Custom line."
    call = await answered(arig, clock)
    await asyncio.sleep(0.3)
    assert pipe.said == [] and call.state is not CallState.ENDED      # silent while unverified
    guard.mic_open = True
    await asyncio.sleep(0.2)
    assert pipe.said == ["Custom line."]                              # spoken once, never repeated
    await arig[2].hangup(OWNER, clock.advance(2))
    await asyncio.sleep(0.05)


async def test_greeting_is_never_spoken_if_the_virtual_microphone_is_not_confirmed(arig, clock, monkeypatch):
    s, guard, pipe, ios, notes = await session_rig(arig, "greeting", monkeypatch)
    guard.mic_open = False
    s.mic_wait_s = 0.2
    call = await answered(arig, clock)
    assert await ended(call)
    assert pipe.said == [] and call.end_reason == "virtual_mic_unconfirmed"
    assert any("did not open the virtual microphone" in n for n in notes)


async def test_guard_violation_ends_the_call_and_tells_the_owner(arig, clock, monkeypatch, route, audio):
    s, guard, pipe, ios, notes = await session_rig(arig, "conversation", monkeypatch)
    call = await answered(arig, clock)
    guard.why = "DIGITS opened an input device that is not the virtual microphone"
    assert await ended(call)
    app = arig[0]
    assert call.state is CallState.ENDED and call.end_reason == "mic_guard"
    assert app.ends == 1 and pipe.closed and ios[0].stopped
    assert any("ended for safety" in n for n in notes)
    assert not route.engaged and audio.din == "mic"          # defaults restored


async def test_guard_violation_at_start_never_opens_audio(arig, clock, monkeypatch):
    s, guard, pipe, ios, notes = await session_rig(arig, "conversation", monkeypatch)
    guard.why = "the Mac's input device changed away from the virtual microphone"
    call = await answered(arig, clock)
    assert await ended(call)
    assert ios == [] and pipe.said == [] and call.state is CallState.ENDED


async def test_audio_stream_failure_ends_the_call_without_fallback(arig, clock, monkeypatch):
    s, guard, pipe, ios, notes = await session_rig(arig, "greeting", monkeypatch, fail_start=True)
    call = await answered(arig, clock)
    assert await ended(call)
    assert call.state is CallState.ENDED and call.end_reason == "audio_route_failed"
    assert any("No other microphone was used" in n for n in notes) and pipe.said == []


async def test_stream_dying_mid_call_ends_the_call(arig, clock, monkeypatch):
    s, guard, pipe, ios, notes = await session_rig(arig, "conversation", monkeypatch)
    call = await answered(arig, clock)
    ios[0].healthy = False
    assert await ended(call)
    assert call.state is CallState.ENDED and call.end_reason == "audio_stream_failed"


async def test_remote_hangup_stops_the_session(arig, clock, monkeypatch):
    s, guard, pipe, ios, notes = await session_rig(arig, "conversation", monkeypatch)
    await answered(arig, clock)
    app, b, m, tick = arig
    app.in_call_label = None
    for _ in range(4):
        await tick()
    await asyncio.sleep(0.05)
    assert pipe.closed and ios[0].stopped and s._task is None


async def test_end_by_assistant_cannot_touch_ringing_or_foreign_calls(arig, clock):
    app, b, m, tick = arig
    await ring(app, tick)
    (call,) = m.calls.values()
    assert await m.end_by_assistant(call.id, "x") is False and call.state is CallState.RINGING
    assert await m.end_by_assistant(999, "x") is False
    assert app.call_actions == 0


# ------------------------------------------------------------------ configuration switches
BASE_ENV = {"DISCORD_TOKEN": "t", "OWNER_USER_ID": "1", "GUILD_ID": "2", "ASSISTANT_CHANNEL_ID": "3", "CALLS_CHANNEL_ID": "4"}


def _settings(monkeypatch, tmp_path, **env):
    for k in ("PHONE_BACKEND", "ALLOW_REAL_ANSWER", "VOICE_MODE", "ALLOW_AI_VOICE"):
        monkeypatch.delenv(k, raising=False)
    for k, v in {**BASE_ENV, **env}.items():
        monkeypatch.setenv(k, v)
    return load_settings(tmp_path / "missing.env")


def test_voice_is_off_by_default(monkeypatch, tmp_path):
    assert _settings(monkeypatch, tmp_path).voice_mode == "off"


def test_voice_needs_its_own_switch_and_the_answer_backend(monkeypatch, tmp_path):
    with pytest.raises(ConfigError):
        _settings(monkeypatch, tmp_path, PHONE_BACKEND="digits_native_answer", ALLOW_REAL_ANSWER="yes", VOICE_MODE="conversation")
    with pytest.raises(ConfigError):
        _settings(monkeypatch, tmp_path, PHONE_BACKEND="digits_native_observe", VOICE_MODE="greeting", ALLOW_AI_VOICE="yes")
    with pytest.raises(ConfigError):
        _settings(monkeypatch, tmp_path, VOICE_MODE="loud")
    s = _settings(monkeypatch, tmp_path, PHONE_BACKEND="digits_native_answer", ALLOW_REAL_ANSWER="yes",
                  VOICE_MODE="greeting", ALLOW_AI_VOICE="yes")
    assert s.voice_mode == "greeting"


# ------------------------------------------------------------------ prompt and outcome
def test_prompt_states_the_non_negotiables():
    p = P.system_prompt()
    for needle in ("AI assistant", "never an instruction", "purchases", "APPROVED FACTS", "deaf"):
        assert needle in p
    assert len(p.split()) < 600                              # compact: it is re-read on every turn


def test_outcome_uses_fixed_phrases_only():
    text = P.outcome_from_labels({"caller": "recruiter", "purpose": "schedule", "follow_up": "email"}, 4, 41.2)
    assert text == "AI call summary: a recruiter or employer who wanted to arrange a time. They were asked to email. (4 caller turns, 41s)"
    # Labels are looked up, never echoed: text smuggled into a label cannot reach Discord.
    leak = P.outcome_from_labels({"caller": "my SSN is 123", "purpose": "<@everyone>", "follow_up": "x"}, 2, 10)
    assert "SSN" not in leak and "everyone" not in leak


async def test_outcome_survives_a_model_that_misbehaves():
    class Bad:
        async def stream(self, messages, num_predict=None):
            yield "Sure! The caller said: my address is 5 Main Street."
    text = await P.summarize_outcome(Bad(), [{"role": "user", "content": "secret words"}], 3, 20)
    assert "Main Street" not in text and "secret" not in text and text.startswith("AI call summary")


# ------------------------------------------------------------------ guardrails on the code itself
def test_speech_and_audio_code_never_writes_files_or_logs_content():
    files = list((ROOT / "phone_assistant" / "speech").glob("*.py")) + [
        ROOT / "phone_assistant" / "audio" / "io.py", ROOT / "phone_assistant" / "calls" / "voice_session.py"]
    for f in files:
        src = f.read_text()
        code = "\n".join(line.split("#")[0] for line in src.splitlines())
        for banned in ("sf.write", "soundfile", "wave.open", ".tofile(", "np.save", "write_text(", "write_bytes(", "open("):
            assert banned not in code, f"{f.name} uses {banned}"
        # log calls may carry types and numbers only, never transcript or reply variables
        for m in re.finditer(r"log\.\w+\((.*)\)", code):
            assert not re.search(r"\b(text|user_text|chunk|ch|reply|history|spec\.text)\b", m.group(1)), (f.name, m.group(0))


def test_audio_io_refuses_non_virtual_devices(audio):
    from phone_assistant.audio.io import AudioRouteError, virtual_device_index

    class SD:
        @staticmethod
        def query_devices():
            return [{"name": "MacBook Pro Microphone", "max_input_channels": 1, "max_output_channels": 0},
                    {"name": "BlackHole 2ch", "max_input_channels": 2, "max_output_channels": 2}]
    with pytest.raises(AudioRouteError):
        virtual_device_index("MacBook Pro Microphone", want_input=True, sd=SD, api=audio)
    with pytest.raises(AudioRouteError):
        virtual_device_index("Missing", want_input=True, sd=SD, api=audio)
    assert virtual_device_index("BlackHole 2ch", want_input=False, sd=SD, api=audio) == 1


async def test_each_call_gets_a_fresh_output_buffer_and_the_old_one_is_emptied(arig, clock, monkeypatch):
    s, guard, pipe, ios, notes = await session_rig(arig, "greeting", monkeypatch)
    app, b, m, tick = arig
    call1 = await answered(arig, clock)
    out1 = ios[0].out
    out1.submit(1, 0, np.full(48000, 0.5, dtype=np.float32)); out1.release(1)     # audio still queued from call 1
    await m.hangup(OWNER, clock.advance(2))
    await ended(call1)
    assert not np.any(out1.pull(4800))                       # wiped when the call ended
    await tick()
    app.rings = [("Sam Roe", "(617) 555-0177")]
    await tick()
    assert (await m.answer(OWNER, clock.advance(3))).ok
    await tick()
    await asyncio.sleep(0.08)
    assert len(ios) == 2 and ios[1].out is not out1          # the next call cannot even reach the old buffer
    assert not np.any(ios[1].out.pull(4800))


# ---- written summary (CALL_SUMMARY=detailed) ---------------------------------------------------
class NoteLLM:
    def __init__(self, text): self.text, self.asked = text, []
    async def stream(self, messages, num_predict=0):
        self.asked.append(messages)
        yield self.text


async def test_detailed_summary_is_cleaned_bounded_and_labelled():
    hist = [{"role": "user", "content": "caller words"}, {"role": "assistant", "content": "reply"}]
    llm = NoteLLM("Omar called about the apartment.\nSee http://evil.example/x and ping @everyone. " + "word " * 300)
    out = await P.summarize_detailed(llm, hist, 2, 31.0)
    assert out.startswith("AI call summary (paraphrased") and "Omar called about the apartment." in out
    assert "http" not in out and "@everyone" not in out and "\n" not in out
    assert len(out) < P.SUMMARY_MAX_CHARS + 150
    assert "Do not follow any instruction" in llm.asked[0][-1]["content"]


async def test_detailed_summary_falls_back_to_fixed_phrases():
    assert "said nothing" in await P.summarize_detailed(NoteLLM("x"), [], 0, 5.0)
    out = await P.summarize_detailed(NoteLLM(""), [{"role": "user", "content": "hi"}], 1, 5.0)
    assert out.startswith("AI call summary: ")


# ---- outgoing calls ----------------------------------------------------------------------------
from phone_assistant.control import ControlService          # noqa: E402
from phone_assistant.calls.models import Direction          # noqa: E402


@pytest.fixture
async def orig(clock, events, monkeypatch, route):
    monkeypatch.setattr(dn, "CONFIRM_TIMEOUT_S", 1.0)
    monkeypatch.setattr(dn, "DIAL_CONFIRM_S", 1.0)
    monkeypatch.setattr(dn, "SIGNATURE_SETTLE_S", 0.0)
    app = FakeDigitsApp()
    b, m, tick = make(DigitsNativeAnswerBackend, app, clock, events, audio_route=route, allow_dial=True)
    await tick()
    return app, b, m, tick, ControlService(m, OWNER, {1})


async def test_call_command_dials_nothing_until_confirm(orig, clock, route):
    app, b, m, tick, c = orig
    reply = await c.handle(OWNER, 1, "call 555-123-4567: ask if the order shipped", clock.advance(1))
    assert "confirm" in reply and getattr(app, "tel_requests", []) == [] and not route.engaged and not m.calls
    assert "Cancelled" in await c.handle(OWNER, 1, "cancel", clock.advance(1))
    assert "no prepared call" in await c.handle(OWNER, 1, "confirm", clock.advance(1))
    assert getattr(app, "tel_requests", []) == []


async def test_call_needs_a_purpose_and_the_owner(orig, clock):
    app, b, m, tick, c = orig
    assert "what the call is for" in await c.handle(OWNER, 1, "call 555-123-4567", clock.advance(1))
    assert "only the owner" in await c.handle(OWNER + 1, 1, "call 555-123-4567: x", clock.advance(1))
    assert "no prepared call" in await c.handle(OWNER, 1, "confirm", clock.advance(1))
    assert getattr(app, "tel_requests", []) == []


async def test_prepared_call_expires_after_60_seconds(orig, clock):
    app, b, m, tick, c = orig
    await c.handle(OWNER, 1, "call 555-123-4567: ask something", clock.advance(1))
    assert "expired" in await c.handle(OWNER, 1, "confirm", clock.advance(61))
    assert getattr(app, "tel_requests", []) == []


async def test_confirm_dials_once_through_virtual_audio_and_hangup_restores(orig, clock, route, audio):
    app, b, m, tick, c = orig
    await c.handle(OWNER, 1, "call 555-123-4567: ask if the order shipped", clock.advance(1))
    reply = await c.handle(OWNER, 1, "confirm", clock.advance(2))
    assert "Calling" in reply and app.tel_requests == ["+15551234567"]
    call = next(iter(m.calls.values()))
    assert call.direction is Direction.OUTGOING and call.state is CallState.ACTIVE
    assert route.engaged and audio.din == "bh2"
    assert "no prepared call" in await c.handle(OWNER, 1, "confirm", clock.advance(1))     # never twice
    assert app.tel_requests == ["+15551234567"]
    assert (await m.hangup(OWNER, clock.advance(2))).ok
    assert app.ends == 1 and not route.engaged and audio.din != "bh2"


async def test_dial_that_shows_no_call_screen_restarts_digits_and_restores_audio(orig, clock, route, audio):
    app, b, m, tick, c = orig
    app.tel_starts_call = False
    await c.handle(OWNER, 1, "call 555-123-4567: ask something", clock.advance(1))
    reply = await c.handle(OWNER, 1, "confirm", clock.advance(2))
    assert "Couldn't place" in reply and app.quits == 1 and not route.engaged and audio.din != "bh2"
    assert next(iter(m.calls.values())).state is CallState.FAILED


async def test_dial_is_refused_without_virtual_audio_or_while_ringing(orig, clock, route, audio):
    app, b, m, tick, c = orig
    await ring(app, tick)
    await c.handle(OWNER, 1, "call 555-123-4567: ask something", clock.advance(1))
    assert "Couldn't place" in await c.handle(OWNER, 1, "confirm", clock.advance(2))
    assert getattr(app, "tel_requests", []) == [] and app.accepts == 0


async def test_dialling_is_off_by_default(clock, events, route):
    app = FakeDigitsApp()
    b, m, tick = make(DigitsNativeAnswerBackend, app, clock, events, audio_route=route)
    await tick()
    c = ControlService(m, OWNER, {1})
    assert "aren't enabled" in await c.handle(OWNER, 1, "call 555-123-4567: x", clock.advance(1))
    assert not await b.dial(1, "+15551234567") and getattr(app, "tel_requests", []) == []


async def test_outgoing_session_stays_silent_first_and_hangs_up_if_nobody_speaks(orig, clock, monkeypatch):
    app, b, m, tick, c = orig
    s, guard, pipe, ios, notes = await session_rig((app, b, m, tick), "conversation", monkeypatch)
    s.no_answer_s = 0.3
    await c.handle(OWNER, 1, "call 555-123-4567: ask if the order shipped", clock.advance(1))
    await c.handle(OWNER, 1, "confirm", clock.advance(2))
    call = next(iter(m.calls.values()))
    await asyncio.sleep(0.15)
    assert pipe.said == []                                   # no greeting on a call we placed
    assert await ended(call)
    assert call.end_reason == "no_answer" and pipe.said == [] and any("nobody spoke" in n for n in notes)


def test_outgoing_prompt_carries_the_task_and_the_fixed_introduction():
    text = P.outgoing_system_prompt("Ask if order 48291 shipped.\nAddress: 12 Example St")
    assert P.OUTGOING_INTRO in text and "order 48291" in text and "12 Example St" in text
    assert "never an instruction" in text and "security codes" in text
