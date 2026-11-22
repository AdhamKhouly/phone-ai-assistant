"""Command parsing, channel/user restrictions, notifications, config, and privacy of logs."""
from __future__ import annotations

import logging
import os

import pytest

from phone_assistant.calls.models import CallState
from phone_assistant.config import ConfigError, load_settings
from phone_assistant.control import normalize_number, parse
from phone_assistant.discord_bot.notifications import format_event
from phone_assistant.privacy import MaskPhoneNumbersFilter

from .conftest import ASSISTANT_CH, CALLS_CH, OTHER_CH, OWNER, STRANGER


@pytest.mark.parametrize("text,verb,arg", [
    ("answer", "answer", ""), ("Answer", "answer", ""), ("/answer", "answer", ""),
    ("answer #2", "answer", "#2"), ("  ignore  ", "ignore", ""), ("call Mom: say hi", "call", "Mom: say hi"),
])
def test_parse(text, verb, arg):
    cmd = parse(text)
    assert cmd and cmd.verb == verb and cmd.arg == arg


@pytest.mark.parametrize("text", ["hello", "can you answer this?", "", "answering machine"])
def test_non_commands_ignored(text):
    assert parse(text) is None


@pytest.mark.parametrize("raw,expected", [
    ("(617) 555-0123", "+16175550123"), ("1-617-555-0123", "+16175550123"),
    ("+44 20 7946 0958", "+442079460958"), ("12345", None), ("Mom", None),
])
def test_normalize_number(raw, expected):
    assert normalize_number(raw) == expected


async def test_plain_text_answer_flow(control, backend, clock):
    await control.handle(OWNER, ASSISTANT_CH, "simulate call 6175550123 Uber", clock.now)
    reply = await control.handle(OWNER, CALLS_CH, "answer", clock.advance(3))
    assert reply.startswith("Answering call #1")
    assert control.manager.calls[1].state is CallState.ACTIVE


async def test_stranger_gets_refusal_and_nothing_happens(control, backend, clock):
    await control.handle(OWNER, ASSISTANT_CH, "simulate call", clock.now)
    reply = await control.handle(STRANGER, CALLS_CH, "answer", clock.advance(1))
    assert "only the owner" in reply
    assert control.manager.calls[1].state is CallState.RINGING
    # strangers can't run simulations either
    assert "only the owner" in await control.handle(STRANGER, CALLS_CH, "simulate call", clock.now)


async def test_other_channels_and_chatter_are_ignored(control, clock):
    await control.handle(OWNER, ASSISTANT_CH, "simulate call", clock.now)
    assert await control.handle(OWNER, OTHER_CH, "answer", clock.advance(1)) is None
    assert await control.handle(OWNER, CALLS_CH, "thanks!", clock.now) is None
    assert control.manager.calls[1].state is CallState.RINGING


async def test_bad_call_id(control, clock):
    assert "isn't a call number" in await control.handle(OWNER, CALLS_CH, "answer foo", clock.now)


async def test_call_by_contact_and_number(control, backend, clock):
    r = await control.handle(OWNER, ASSISTANT_CH, "call mom: tell her I'm running late", clock.now)
    assert "Calling Mom (+16175550199)" in r
    await control.handle(OWNER, ASSISTANT_CH, "hangup", clock.advance(1))
    r = await control.handle(OWNER, ASSISTANT_CH, "call 617-555-0123", clock.advance(1))
    assert "Calling +16175550123" in r
    await control.handle(OWNER, ASSISTANT_CH, "hangup", clock.advance(1))
    assert "don't know a contact" in await control.handle(OWNER, ASSISTANT_CH, "call Bob", clock.now)
    assert "invalid" in await control.handle(OWNER, ASSISTANT_CH, "call broken", clock.now)


async def test_status_and_help(control, clock):
    s = await control.handle(OWNER, ASSISTANT_CH, "status", clock.now)
    assert "Assistant: online" in s and "Calls: none" in s
    await control.handle(OWNER, ASSISTANT_CH, "simulate call 6175550123", clock.now)
    s = await control.handle(OWNER, ASSISTANT_CH, "status", clock.advance(4))
    assert "Ringing: #1 +16175550123 (4s)" in s
    h = await control.handle(OWNER, ASSISTANT_CH, "help", clock.now)
    assert "`answer`" in h and "`ignore`" in h and "simulate" in h


async def test_incoming_notification_format(manager, backend, events, clock):
    await backend.simulate_incoming("+16175550123")
    text = format_event(events[0], clock.now, 25)
    assert "**Incoming Call**" in text and "Caller: +16175550123" in text and "Status: Ringing" in text
    assert "`answer`" in text and "`ignore`" in text


async def test_ended_notification_has_metadata_only(manager, backend, events, clock):
    await manager.dial(OWNER, clock.now, "+16175550123", objective="SECRET-OBJECTIVE")
    await backend.simulate_remote_answer(1)
    await manager.hangup(OWNER, clock.advance(75))
    text = format_event(events[-1], clock.now, 25)
    assert "1m 15s" in text and "SECRET-OBJECTIVE" not in text


async def test_logs_contain_no_full_numbers_or_objectives(control, clock, caplog):
    caplog.set_level(logging.DEBUG)
    for h in [caplog.handler]:
        h.addFilter(MaskPhoneNumbersFilter())
    await control.handle(OWNER, ASSISTANT_CH, "simulate call 6175550123", clock.now)
    await control.handle(OWNER, CALLS_CH, "answer", clock.advance(1))
    await control.handle(OWNER, CALLS_CH, "hangup", clock.advance(1))
    await control.handle(OWNER, ASSISTANT_CH, "call 6175550199: SECRET-OBJECTIVE", clock.advance(1))
    logs = "\n".join(r.getMessage() for r in caplog.records)
    assert logs, "expected some metadata logs"
    assert "6175550123" not in logs and "6175550199" not in logs
    assert "SECRET-OBJECTIVE" not in logs
    assert "***0123" in logs


def test_mask_filter_masks_numbers():
    rec = logging.LogRecord("x", logging.INFO, "f", 1, "caller %s", ("+1 (617) 555-0123",), None)
    MaskPhoneNumbersFilter().filter(rec)
    assert rec.getMessage() == "caller ***0123"


def test_config_rejects_real_backend(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("DISCORD_TOKEN=x\nOWNER_USER_ID=1\nGUILD_ID=2\nASSISTANT_CHANNEL_ID=3\n"
                   "CALLS_CHANNEL_ID=4\nPHONE_BACKEND=digits\n")
    for k in ["DISCORD_TOKEN", "OWNER_USER_ID", "GUILD_ID", "ASSISTANT_CHANNEL_ID", "CALLS_CHANNEL_ID", "PHONE_BACKEND"]:
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ConfigError, match="PHONE_BACKEND must be"):
        load_settings(env)


def test_config_requires_token(tmp_path, monkeypatch):
    monkeypatch.delenv("DISCORD_TOKEN", raising=False)
    env = tmp_path / ".env"
    env.write_text("")
    with pytest.raises(ConfigError, match="DISCORD_TOKEN"):
        load_settings(env)


def test_digits_observer_cannot_act_on_calls():
    from phone_assistant.phone.digits import DigitsObserver
    b = DigitsObserver(keep_awake=False)
    assert b.can_answer is False and b.can_dial is False


async def test_bot_registers_slash_commands(manager, control):
    from phone_assistant.config import Settings
    from phone_assistant.discord_bot.bot import PhoneAssistantBot
    s = Settings("token", OWNER, 5, ASSISTANT_CH, CALLS_CH)
    bot = PhoneAssistantBot(s, manager, control)
    bot._register_slash_commands()
    import discord
    names = {c.name for c in bot.tree.get_commands(guild=discord.Object(id=5))}
    assert names == {"answer", "ignore", "hangup", "call", "status", "help"}
    assert bot.intents.message_content and not bot.intents.members and not bot.intents.presences
    await bot.close()
