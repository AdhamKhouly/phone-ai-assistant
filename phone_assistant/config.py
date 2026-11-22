"""Configuration from environment / .env. Secrets never live in source code."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


# simulated               test calls only
# digits_native_observe   native DIGITS Mac app -> Discord, observe-only (cannot answer)
# digits_native_answer    native DIGITS Mac app with manual `answer` (needs ALLOW_REAL_ANSWER=yes)
# digits_observe / digits_answer   the older Chrome (web client) versions, kept as a fallback
BACKENDS = ("simulated", "digits_native_observe", "digits_native_answer", "digits_observe", "digits_answer")
ANSWERING_BACKENDS = ("digits_native_answer", "digits_answer")
VOICE_MODES = ("off", "greeting", "conversation")


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    discord_token: str
    owner_user_id: int
    guild_id: int
    assistant_channel_id: int
    calls_channel_id: int
    phone_backend: str = "simulated"
    answer_window_s: float = 25.0
    max_command_age_s: float = 15.0
    max_call_s: float = 60.0   # AI-handled calls are force-ended after this; 0 = no limit (ends on hang-up or silence)
    digits_autolaunch: bool = True   # native backends reopen the DIGITS app if it isn't running
    voice_mode: str = "off"          # off | greeting | conversation (what the assistant does on an answered call)
    allow_outgoing: bool = False     # outgoing calls on the owner's confirmed `call` command (ALLOW_OUTGOING=yes)
    call_summary: str = "categories"  # categories | detailed (paraphrased note sent to #calls)
    voice_greeting: str = ""         # optional fixed opening line; empty = the built-in greeting
    audio_mic_device: str = "BlackHole 2ch"       # virtual device DIGITS uses as its microphone
    audio_capture_device: str = "BlackHole 16ch"  # virtual device DIGITS plays the caller into
    contacts_file: Path = ROOT / "config" / "contacts.json"

    def load_contacts(self) -> dict[str, str]:
        if not self.contacts_file.exists():
            return {}
        data = json.loads(self.contacts_file.read_text())
        if not isinstance(data, dict):
            raise ConfigError("contacts.json must be an object of {name: number}")
        return {str(k): str(v) for k, v in data.items()}


def _req_int(name: str) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw.isdigit():
        raise ConfigError(f"{name} is missing or not a numeric Discord ID. See README > Setup.")
    return int(raw)


def load_settings(env_file: Path | None = None) -> Settings:
    load_dotenv(env_file or ROOT / ".env")
    token = os.environ.get("DISCORD_TOKEN", "").strip()
    if not token:
        raise ConfigError("DISCORD_TOKEN is missing. Copy .env.example to .env and fill it in.")
    backend = os.environ.get("PHONE_BACKEND", "simulated").strip().lower()
    if backend not in BACKENDS:
        raise ConfigError("PHONE_BACKEND must be one of: " + ", ".join(BACKENDS) + ".")
    if backend in ANSWERING_BACKENDS and os.environ.get("ALLOW_REAL_ANSWER", "").strip().lower() != "yes":
        raise ConfigError(f"PHONE_BACKEND={backend} also requires ALLOW_REAL_ANSWER=yes "
                          "(second safety switch). Refusing to start.")
    voice_mode = os.environ.get("VOICE_MODE", "off").strip().lower()
    allow_outgoing = os.environ.get("ALLOW_OUTGOING", "").strip().lower() == "yes"
    if allow_outgoing and not (backend == "digits_native_answer" and voice_mode == "conversation"):
        raise ConfigError("ALLOW_OUTGOING=yes needs PHONE_BACKEND=digits_native_answer and VOICE_MODE=conversation.")
    call_summary = os.environ.get("CALL_SUMMARY", "categories").strip().lower()
    if call_summary not in ("categories", "detailed"):
        raise ConfigError("CALL_SUMMARY must be 'categories' or 'detailed'.")
    if voice_mode not in VOICE_MODES:
        raise ConfigError("VOICE_MODE must be one of: " + ", ".join(VOICE_MODES) + ".")
    if voice_mode != "off":
        if backend != "digits_native_answer":
            raise ConfigError("VOICE_MODE other than 'off' only works with PHONE_BACKEND=digits_native_answer.")
        if os.environ.get("ALLOW_AI_VOICE", "").strip().lower() != "yes":
            raise ConfigError(f"VOICE_MODE={voice_mode} also requires ALLOW_AI_VOICE=yes (separate safety switch). "
                              "Refusing to start.")
    return Settings(
        discord_token=token,
        owner_user_id=_req_int("OWNER_USER_ID"),
        guild_id=_req_int("GUILD_ID"),
        assistant_channel_id=_req_int("ASSISTANT_CHANNEL_ID"),
        calls_channel_id=_req_int("CALLS_CHANNEL_ID"),
        phone_backend=backend,
        answer_window_s=float(os.environ.get("ANSWER_WINDOW_SECONDS", "25")),
        max_command_age_s=float(os.environ.get("MAX_COMMAND_AGE_SECONDS", "15")),
        max_call_s=max(0.0, float(os.environ.get("MAX_CALL_SECONDS", "60"))),
        digits_autolaunch=os.environ.get("DIGITS_AUTOLAUNCH", "yes").strip().lower() != "no",
        voice_mode=voice_mode,
        call_summary=call_summary,
        allow_outgoing=allow_outgoing,
        voice_greeting=os.environ.get("VOICE_GREETING", "").strip(),
        audio_mic_device=os.environ.get("AUDIO_MIC_DEVICE", "BlackHole 2ch").strip(),
        audio_capture_device=os.environ.get("AUDIO_CAPTURE_DEVICE", "BlackHole 16ch").strip(),
    )
