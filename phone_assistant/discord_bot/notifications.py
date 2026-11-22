"""Turns call events into #calls messages. Metadata only — never conversation content."""
from __future__ import annotations

from datetime import datetime

from ..calls.manager import CallEvent

REASONS = {
    "owner_hangup": "you hung up",
    "owner_hangup_backend_error": "you hung up (backend didn't confirm, check your phone)",
    "remote_hangup": "the other side hung up",
    "simulated_remote_end": "the other side hung up (simulated)",
    "no_decision_in_time": "no decision in time, left to your phone/voicemail",
    "backend_disconnected": "phone connection lost, call left alone",
    "answer_failed": "the assistant couldn't pick up, left to your phone/voicemail",
    "dial_failed": "the call couldn't be placed",
    "assistant_shutdown": "assistant shut down, call left to your phone",
    "stopped_ringing": "voicemail, caller hung up, or answered on your iPhone",
    "max_duration": "safety time limit reached, assistant hung up",
    "max_duration_backend_error": "safety time limit reached; hang-up NOT confirmed, check your phone",
}


def _fmt_duration(seconds: int | None) -> str:
    if seconds is None:
        return "-"
    m, s = divmod(seconds, 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def format_event(event: CallEvent, now: datetime, answer_window_s: float,
                 answer_enabled: bool = True) -> str | None:
    c = event.call
    if event.kind == "backend":
        return ("Phone connection restored." if event.detail == "connected"
                else "⚠️ Phone connection lost (DIGITS signed out, not running, or Mac asleep). "
                     "The assistant can't see calls until it's back; your phone works normally.")
    if c is None:
        return None
    reason = REASONS.get(c.end_reason or "", c.end_reason or "")
    if event.kind == "incoming" and not answer_enabled:
        return (f"📞 **Incoming Call** (#{c.id})\n"
                f"Caller: {c.caller_label}\n"
                f"Status: Ringing\n\n"
                f"Observe-only mode: the AI can't answer yet. Your phone and voicemail handle this call.")
    if event.kind == "incoming":
        return (f"📞 **Incoming Call** (#{c.id})\n"
                f"Caller: {c.caller_label}\n"
                f"Status: Ringing\n\n"
                f"Reply `answer` to let the AI handle it, or `ignore` to leave it alone. "
                f"With no reply within ~{int(answer_window_s)}s it goes to your phone/voicemail as usual.")
    if event.kind == "active":
        return f"🤖 Call #{c.id} with {c.caller_label}: AI is on the call. `hangup` to end it."
    if event.kind == "dialing":
        return f"📤 Call #{c.id}: dialing {c.caller_label}…"
    if event.kind == "stopped":
        secs = int((c.ring_stopped_at - c.started_at).total_seconds()) if c.ring_stopped_at else None
        return (f"🔕 Call #{c.id} from {c.caller_label} stopped ringing"
                + (f" after {secs}s" if secs is not None else "") + " (voicemail, caller hung up, or answered on your iPhone).")
    if event.kind == "missed" and c.end_reason == "no_decision_in_time":
        return (f"⏱ Answer window closed for call #{c.id} from {c.caller_label}. "
                f"It's left to your phone/voicemail; I'll post when it stops ringing.")
    if event.kind == "missed":
        return f"Call #{c.id} from {c.caller_label} stopped ringing ({reason or 'ended'})."
    if event.kind == "ended":
        return (f"✅ Call #{c.id} with {c.caller_label} ended. "
                f"Duration: {_fmt_duration(c.duration_seconds(now))}. Reason: {reason}.")
    if event.kind == "failed":
        return f"❌ Call #{c.id} ({c.caller_label}): {reason}."
    return None  # 'ignored' is already confirmed by the command reply
