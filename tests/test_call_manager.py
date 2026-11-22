"""Safety tests for the call state machine."""
from __future__ import annotations

import asyncio

from phone_assistant.calls.models import CallState

from .conftest import OWNER, STRANGER


def answers(backend):
    return [a for a in backend.actions if a[0] == "answer"]


async def test_owner_answer_transitions_ringing_to_active(manager, backend, clock):
    call = await backend.simulate_incoming("+16175550123")
    assert call.state is CallState.RINGING
    r = await manager.answer(OWNER, clock.advance(3))
    assert r.ok and call.state is CallState.ACTIVE and call.answered_at is not None
    assert answers(backend) == [("answer", call.id)]


async def test_no_decision_leaves_call_alone(manager, backend, clock, events):
    call = await backend.simulate_incoming("+16175550123")
    clock.advance(26)
    await manager.expire_stale()
    assert call.state is CallState.MISSED and call.end_reason == "no_decision_in_time"
    assert backend.actions == []  # never touched the phone
    assert [e.kind for e in events] == ["incoming", "missed"]


async def test_unauthorized_user_cannot_answer(manager, backend, clock):
    call = await backend.simulate_incoming("+16175550123")
    r = await manager.answer(STRANGER, clock.advance(2))
    assert not r.ok and "Not authorized" in r.message
    assert call.state is CallState.RINGING and backend.actions == []


async def test_command_sent_before_call_started_is_rejected(manager, backend, clock):
    issued = clock.now
    clock.advance(1)
    call = await backend.simulate_incoming("+16175550123")
    r = await manager.answer(OWNER, issued)
    assert not r.ok and "before call" in r.message
    assert call.state is CallState.RINGING and backend.actions == []


async def test_delayed_command_is_rejected(manager, backend, clock):
    # Long ring window to isolate the max-command-age check.
    manager.answer_window = manager.answer_window * 10
    await backend.simulate_incoming("+16175550123")
    issued = clock.advance(1)
    clock.advance(20)  # message delivered 20s late (> 15s max age)
    r = await manager.answer(OWNER, issued)
    assert not r.ok and "stale" in r.message
    assert backend.actions == []


async def test_answer_after_window_is_rejected(manager, backend, clock):
    call = await backend.simulate_incoming("+16175550123")
    clock.advance(30)
    r = await manager.answer(OWNER, clock.now)
    assert not r.ok and call.state is CallState.MISSED and backend.actions == []


async def test_repeated_answer_only_answers_once(manager, backend, clock):
    await backend.simulate_incoming("+16175550123")
    t = clock.advance(2)
    assert (await manager.answer(OWNER, t)).ok
    r2 = await manager.answer(OWNER, clock.advance(1))
    assert not r2.ok
    assert len(answers(backend)) == 1


async def test_concurrent_answers_only_one_wins(manager, backend, clock):
    await backend.simulate_incoming("+16175550123")
    t = clock.advance(2)
    results = await asyncio.gather(manager.answer(OWNER, t), manager.answer(OWNER, t))
    assert sorted(r.ok for r in results) == [False, True]
    assert len(answers(backend)) == 1


async def test_old_answer_never_applies_to_new_call(manager, backend, clock):
    first = await backend.simulate_incoming("+16175550001")
    late_answer_time = clock.advance(2)       # owner typed `answer` for call 1...
    await backend.simulate_remote_end(first.id, "remote_hangup")  # ...but call 1 ended
    clock.advance(1)
    second = await backend.simulate_incoming("+16175550002")
    r = await manager.answer(OWNER, late_answer_time)   # delayed delivery
    assert not r.ok and second.state is CallState.RINGING and backend.actions == []
    # explicitly targeting the old call is also refused
    r = await manager.answer(OWNER, clock.advance(1), call_id=first.id)
    assert not r.ok and "missed" in r.message


async def test_ignored_call_cannot_be_answered_later(manager, backend, clock):
    call = await backend.simulate_incoming("+16175550123")
    assert (await manager.ignore(OWNER, clock.advance(2))).ok
    assert call.state is CallState.IGNORED
    r = await manager.answer(OWNER, clock.advance(1))
    assert not r.ok and backend.actions == []  # ignore never touches the phone


async def test_multiple_ringing_requires_selection(manager, backend, clock):
    a = await backend.simulate_incoming("+16175550001")
    b = await backend.simulate_incoming("+16175550002")
    r = await manager.answer(OWNER, clock.advance(2))
    assert not r.ok and "More than one call" in r.message and backend.actions == []
    r = await manager.answer(OWNER, clock.advance(1), call_id=b.id)
    assert r.ok and b.state is CallState.ACTIVE and a.state is CallState.RINGING


async def test_unknown_call_id(manager, backend, clock):
    await backend.simulate_incoming("+16175550001")
    r = await manager.answer(OWNER, clock.advance(1), call_id=42)
    assert not r.ok and "no call #42" in r.message


async def test_hangup_ends_ai_call_with_duration(manager, backend, clock):
    call = await backend.simulate_incoming("+16175550123")
    await manager.answer(OWNER, clock.advance(2))
    clock.advance(65)
    r = await manager.hangup(OWNER, clock.now)
    assert r.ok and call.state is CallState.ENDED and call.end_reason == "owner_hangup"
    assert call.duration_seconds(clock.now) == 65


async def test_hangup_without_active_call(manager, backend, clock):
    await backend.simulate_incoming("+16175550123")   # ringing is not hang-up-able
    r = await manager.hangup(OWNER, clock.advance(1))
    assert not r.ok and backend.actions == []


async def test_unauthorized_hangup_rejected(manager, backend, clock):
    call = await backend.simulate_incoming("+16175550123")
    await manager.answer(OWNER, clock.advance(1))
    r = await manager.hangup(STRANGER, clock.advance(1))
    assert not r.ok and call.state is CallState.ACTIVE


async def test_backend_offline_blocks_answer_and_releases_calls(manager, backend, clock, events):
    call = await backend.simulate_incoming("+16175550123")
    await backend.set_connected(False)
    assert call.state is CallState.MISSED and call.end_reason == "backend_disconnected"
    r = await manager.answer(OWNER, clock.advance(1))
    assert not r.ok and "offline" in r.message
    assert answers(backend) == []
    await backend.set_connected(True)
    new = await backend.simulate_incoming("+16175550124")
    assert (await manager.answer(OWNER, clock.advance(1))).ok and new.state is CallState.ACTIVE


async def test_backend_offline_during_active_call(manager, backend, clock):
    call = await backend.simulate_incoming("+16175550123")
    await manager.answer(OWNER, clock.advance(1))
    await backend.set_connected(False)
    assert call.state is CallState.ENDED and call.end_reason == "backend_disconnected"


async def test_answer_failure_leaves_call_alone(manager, backend, clock, events):
    call = await backend.simulate_incoming("+16175550123")
    backend.fail_next_answer = True
    r = await manager.answer(OWNER, clock.advance(1))
    assert not r.ok and call.state is CallState.FAILED
    assert events[-1].kind == "failed"


async def test_caller_hangs_up_while_ringing(manager, backend, clock):
    call = await backend.simulate_incoming("+16175550123")
    await backend.simulate_remote_end(call.id, "remote_hangup")
    assert call.state is CallState.MISSED
    assert not (await manager.answer(OWNER, clock.advance(1))).ok


async def test_outgoing_call_and_objective_cleared(manager, backend, clock):
    r = await manager.dial(OWNER, clock.now, "+16175550123", objective="ask about table for 2")
    call = r.call
    assert r.ok and call.state is CallState.DIALING
    await backend.simulate_remote_answer(call.id)
    assert call.state is CallState.ACTIVE
    await manager.hangup(OWNER, clock.advance(30))
    assert call.state is CallState.ENDED and call.objective is None


async def test_cannot_dial_while_on_a_call(manager, backend, clock):
    await backend.simulate_incoming("+16175550123")
    await manager.answer(OWNER, clock.advance(1))
    r = await manager.dial(OWNER, clock.advance(1), "+16175550000")
    assert not r.ok


async def test_shutdown_releases_calls(manager, backend, clock):
    ringing = await backend.simulate_incoming("+16175550001")
    active = await backend.simulate_incoming("+16175550002")
    await manager.answer(OWNER, clock.advance(1), call_id=active.id)
    released = await manager.shutdown()
    assert {c.id for c in released} == {ringing.id, active.id}
    assert ringing.state is CallState.MISSED and active.state is CallState.ENDED
    assert ("hangup", active.id) in backend.actions and ("answer", ringing.id) not in backend.actions


async def test_unauthorized_dial_rejected(manager, backend, clock):
    r = await manager.dial(STRANGER, clock.now, "+16175550123")
    assert not r.ok and backend.actions == []
