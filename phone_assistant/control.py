"""Transport-independent command handling.

The Discord layer turns every message / slash command into
`handle(author_id, channel_id, text, issued_at)`. Keeping this free of discord.py
makes the security rules easy to test.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from .calls.manager import CallManager
from .phone.simulated import SimulatedPhoneBackend

HELP_TEXT = """**Phone assistant commands** (only the owner can use these)
`answer` — let the AI answer the call that is ringing now (`answer 3` to pick call #3)
`ignore` — leave the ringing call alone (normal ring + voicemail continue)
`call <number or contact>: <what the call is for>` — prepare an outgoing call; nothing is dialled until you type `confirm`
`confirm` / `cancel` — place or drop the prepared call (it expires after 60 seconds)
`hangup` — end the AI-handled call
`status` — is the assistant online, is a call active
`help` — this message
If you do nothing, the call is left alone."""

SIM_HELP = """Simulation (Phase 1 testing only):
`simulate call [number] [name]` · `simulate end [id]` · `simulate pickup [id]` · `simulate offline` · `simulate online` · `simulate failanswer`"""

COMMANDS = {"answer", "ignore", "call", "confirm", "cancel", "hangup", "status", "help", "simulate"}
CONFIRM_WINDOW_S = 60.0
_ID_RE = re.compile(r"^#?(\d{1,4})$")


@dataclass
class Command:
    verb: str
    arg: str


def parse(text: str) -> Command | None:
    text = text.strip()
    if not text:
        return None
    text = text.lstrip("/!")
    verb, _, arg = text.partition(" ")
    verb = verb.lower()
    if verb not in COMMANDS:
        return None
    return Command(verb, arg.strip())


def parse_call_id(arg: str) -> int | None | str:
    """'' -> None, '3' / '#3' -> 3, anything else -> error string."""
    if not arg:
        return None
    m = _ID_RE.match(arg)
    return int(m.group(1)) if m else f"`{arg}` isn't a call number. Use e.g. `#2` or `2`."


def normalize_number(raw: str) -> str | None:
    """US-centric normalization to E.164. Returns None if it doesn't look like a phone number."""
    if re.search(r"[A-Za-z]", raw):
        return None
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 10:
        return "+1" + digits
    if len(digits) == 11 and digits.startswith("1"):
        return "+" + digits
    if raw.strip().startswith("+") and 8 <= len(digits) <= 15:
        return "+" + digits
    return None


class ControlService:
    def __init__(self, manager: CallManager, owner_id: int, allowed_channel_ids: set[int],
                 contacts: dict[str, str] | None = None) -> None:
        self.manager = manager
        self.owner_id = owner_id
        self.allowed_channel_ids = allowed_channel_ids
        self.contacts = {k.lower(): (k, v) for k, v in (contacts or {}).items()}
        self._pending: tuple | None = None    # (number, name, objective, prepared_at); memory only

    @property
    def simulation_enabled(self) -> bool:
        return isinstance(self.manager.backend, SimulatedPhoneBackend)

    async def handle(self, author_id: int, channel_id: int, text: str, issued_at: datetime) -> str | None:
        """Return a reply, or None when the message is not for us (ordinary chat, other channels)."""
        if channel_id not in self.allowed_channel_ids:
            return None
        cmd = parse(text)
        if cmd is None:
            return None
        if author_id != self.owner_id:
            return "Sorry, only the owner can control the phone assistant."

        m = self.manager
        if cmd.verb == "help":
            return HELP_TEXT + ("\n\n" + SIM_HELP if self.simulation_enabled else "")
        if cmd.verb == "status":
            return m.status_text()
        if cmd.verb in ("answer", "ignore", "hangup"):
            cid = parse_call_id(cmd.arg)
            if isinstance(cid, str):
                return cid
            fn = {"answer": m.answer, "ignore": m.ignore, "hangup": m.hangup}[cmd.verb]
            return (await fn(author_id, issued_at, cid)).message
        if cmd.verb == "call":
            return await self._call(cmd.arg, author_id, issued_at)
        if cmd.verb == "cancel":
            had, self._pending = self._pending is not None, None
            return "Cancelled. Nothing was dialled." if had else "There is no prepared call to cancel."
        if cmd.verb == "confirm":
            pending, self._pending = self._pending, None
            if pending is None:
                return "There is no prepared call. Use `call <number>: <what the call is for>` first."
            number, name, objective, prepared_at = pending
            if (issued_at - prepared_at).total_seconds() > CONFIRM_WINDOW_S:
                return "That prepared call expired (60 seconds). Nothing was dialled. Send the `call` command again."
            return (await m.dial(author_id, issued_at, number, name, objective)).message
        if cmd.verb == "simulate":
            return await self._simulate(cmd.arg)
        return None

    async def _call(self, arg: str, author_id: int, issued_at: datetime) -> str:
        target, _, objective = arg.partition(":")
        target, objective = target.strip(), objective.strip() or None
        if not target:
            return "Usage: `call <number or contact>` (optionally `: objective`)."
        name = None
        number = normalize_number(target)
        if number is None:
            entry = self.contacts.get(target.lower())
            if entry is None:
                return f"I don't know a contact called `{target}` and it isn't a valid phone number."
            name, number = entry[0], normalize_number(entry[1])
            if number is None:
                return f"The saved number for `{target}` is invalid. Fix config/contacts.json."
        if self.simulation_enabled:
            return (await self.manager.dial(author_id, issued_at, number, name, objective)).message
        if not self.manager.backend.can_dial:
            return "Outgoing calls aren't enabled (observe-only mode, or ALLOW_OUTGOING is off)."
        if not objective:
            return ("Tell the assistant what the call is for: `call <number>: <purpose and any details it may share>`. "
                    "Nothing was dialled.")
        self._pending = (number, name, objective, issued_at)
        who = f"{name} ({number})" if name else number
        return (f"Ready to call **{who}**. The AI will say it is calling for you, do what you wrote after the colon, "
                f"and may share only the details you wrote there.\nType `confirm` within 60 seconds to dial, or `cancel`.")

    async def _simulate(self, arg: str) -> str:
        if not self.simulation_enabled:
            return "Simulation commands are only available with the simulated phone backend."
        backend: SimulatedPhoneBackend = self.manager.backend  # type: ignore[assignment]
        sub, _, rest = arg.partition(" ")
        sub = sub.lower()
        if sub == "call":
            parts = rest.split(maxsplit=1)
            number = normalize_number(parts[0]) if parts else "+15555550123"
            if number is None:
                return "Invalid simulated number."
            call = await backend.simulate_incoming(number, parts[1] if len(parts) > 1 else None)
            return f"Simulated incoming call #{call.id}."
        if sub in ("end", "pickup"):
            cid = parse_call_id(rest.strip())
            if isinstance(cid, str):
                return cid
            if cid is None:
                live = self.manager.ringing() + self.manager.active()
                if not live:
                    return "No live call to simulate on."
                cid = live[-1].id
            if sub == "end":
                await backend.simulate_remote_end(cid, "simulated_remote_end")
                return f"Simulated: the other side ended call #{cid}."
            await backend.simulate_remote_answer(cid)
            return f"Simulated: the other side picked up call #{cid}."
        if sub == "offline":
            await backend.set_connected(False)
            return "Simulated: phone backend disconnected."
        if sub == "online":
            await backend.set_connected(True)
            return "Simulated: phone backend reconnected."
        if sub == "failanswer":
            backend.fail_next_answer = True
            return "Simulated: the next answer attempt will fail."
        return SIM_HELP
