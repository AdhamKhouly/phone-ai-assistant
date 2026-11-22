"""Privacy-safe logging.

Rule: logs contain call METADATA only (call id, state, masked number, timing).
Never audio, transcripts, conversation text, or outgoing-call objectives.
As defense in depth, a filter masks anything that looks like a phone number.
Logs go to stderr only (no log files) and there is no telemetry.
"""
from __future__ import annotations

import logging
import re

_PHONE_RE = re.compile(r"\+?\d[\d\-\s().]{6,}\d")


def _mask(match: re.Match) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    return "***" + digits[-4:]


class MaskPhoneNumbersFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        masked = _PHONE_RE.sub(_mask, msg)
        if masked != msg:
            record.msg, record.args = masked, None
        return True


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.addFilter(MaskPhoneNumbersFilter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # discord.py's gateway logs are noisy at INFO; they never include message content at WARNING.
    logging.getLogger("discord").setLevel(logging.WARNING)
