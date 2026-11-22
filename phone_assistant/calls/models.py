"""Call data model. Holds METADATA ONLY — never audio, transcripts, or conversation content."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Direction(str, Enum):
    INCOMING = "incoming"
    OUTGOING = "outgoing"


class CallState(str, Enum):
    RINGING = "ringing"      # incoming, waiting for the owner's decision
    ANSWERING = "answering"  # owner said `answer`; backend is picking up
    DIALING = "dialing"      # outgoing, waiting for the other side
    ACTIVE = "active"        # AI is handling the call
    IGNORED = "ignored"      # owner said `ignore`; phone keeps ringing -> normal voicemail
    MISSED = "missed"        # stopped ringing without AI involvement (voicemail / answered elsewhere / timeout)
    ENDED = "ended"          # AI-handled call finished
    FAILED = "failed"        # backend error; call left alone

    @property
    def is_final(self) -> bool:
        return self in {CallState.IGNORED, CallState.MISSED, CallState.ENDED, CallState.FAILED}


@dataclass
class Call:
    id: int
    direction: Direction
    remote_number: str
    started_at: datetime                 # ring start (incoming) or dial time (outgoing)
    remote_name: str | None = None
    state: CallState = CallState.RINGING
    answered_at: datetime | None = None
    ended_at: datetime | None = None
    end_reason: str | None = None
    ring_stopped_at: datetime | None = None   # when the phone actually stopped ringing (incoming)
    # Owner-provided scope for outgoing calls. Kept in memory only, never logged.
    objective: str | None = field(default=None, repr=False)

    @property
    def caller_label(self) -> str:
        return f"{self.remote_name} ({self.remote_number})" if self.remote_name else self.remote_number

    def duration_seconds(self, now: datetime) -> int | None:
        if self.answered_at is None:
            return None
        end = self.ended_at or now
        return int((end - self.answered_at).total_seconds())
