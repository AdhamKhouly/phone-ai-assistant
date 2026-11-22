"""Interface for the future conversation engine. Not implemented in Phase 1."""
from __future__ import annotations

from typing import Protocol


class ConversationEngine(Protocol):
    def start(self, direction: str, objective: str | None) -> str:
        """Return the opening line. Holds per-call context in memory only."""

    def reply(self, caller_utterance: str) -> str:
        """Return the next spoken reply. Must not persist the utterance anywhere."""

    def close(self) -> None:
        """Discard all in-memory conversation state."""
