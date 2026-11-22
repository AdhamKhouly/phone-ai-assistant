"""Run with:  .venv/bin/python -m phone_assistant"""
from __future__ import annotations

import logging
import sys

import discord

from .calls.manager import CallManager
from .config import ConfigError, load_settings
from .control import ControlService
from .discord_bot.bot import PhoneAssistantBot
from .phone.simulated import SimulatedPhoneBackend
from .privacy import setup_logging


def main() -> int:
    setup_logging()
    try:
        settings = load_settings()
    except ConfigError as e:
        logging.error("%s", e)
        return 2
    route_cfg = None
    try:   # a crashed earlier run may have left the Mac's audio on a virtual device: put it back
        from .audio.routing import restore_stale
        restore_stale()
    except Exception as e:
        logging.warning("could not check the audio devices: %s", type(e).__name__)
    if settings.phone_backend == "digits_native_answer":
        from .audio.routing import CallAudioRoute, RouteConfig
        from .phone.digits_native import DigitsNativeAnswerBackend, default_watchdog_cmd
        route_cfg = RouteConfig(mic=settings.audio_mic_device, capture=settings.audio_capture_device)
        route = CallAudioRoute(route_cfg)
        for problem in route.problems():
            logging.warning("virtual audio: %s. Calls will NOT be answered until this is fixed.", problem)
        backend = DigitsNativeAnswerBackend(   # presses Accept ONLY via CallManager.answer
            autolaunch=settings.digits_autolaunch,
            watchdog_cmd=default_watchdog_cmd(settings.max_call_s),
            audio_route=route, allow_dial=settings.allow_outgoing)                 # no virtual audio path -> the call is not answered
    elif settings.phone_backend == "digits_native_observe":
        from .phone.digits_native import DigitsNativeObserver
        backend = DigitsNativeObserver(autolaunch=settings.digits_autolaunch)  # cannot act on calls
    elif settings.phone_backend == "digits_answer":
        from .phone.digits_answer import DigitsAnswerBackend
        backend = DigitsAnswerBackend()  # clicks Accept ONLY via CallManager.answer
    elif settings.phone_backend == "digits_observe":
        from .phone.digits import DigitsObserver
        backend = DigitsObserver()  # observe-only: cannot answer, dial, click or type
    else:
        backend = SimulatedPhoneBackend()
    manager = CallManager(backend, settings.owner_user_id,
                          answer_window_s=settings.answer_window_s,
                          max_command_age_s=settings.max_command_age_s,
                          max_call_s=settings.max_call_s)
    control = ControlService(manager, settings.owner_user_id,
                             {settings.assistant_channel_id, settings.calls_channel_id},
                             settings.load_contacts())
    bot = PhoneAssistantBot(settings, manager, control)
    if route_cfg is not None:
        # Guards the microphone path on every assistant-answered call; speaks only if VOICE_MODE says so.
        from .calls.voice_session import VoiceSession
        VoiceSession(manager, mode=settings.voice_mode, route_cfg=route_cfg, notify=bot._post_calls,
                     max_call_s=settings.max_call_s, greeting=settings.voice_greeting, summary=settings.call_summary)
    try:
        bot.run(settings.discord_token, log_handler=None)  # keep our privacy-filtered logging
    except discord.LoginFailure:
        logging.error("Discord rejected DISCORD_TOKEN. Reset the token in the Developer Portal and update .env.")
        return 3
    except discord.PrivilegedIntentsRequired:
        logging.error("Enable 'Message Content Intent' for the bot in the Discord Developer Portal (Bot tab).")
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
