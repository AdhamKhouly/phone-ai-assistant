"""Small text heuristics for turn-taking. They only look at the words; nothing is stored."""
from __future__ import annotations

import re

# If the caller's words end like this, they are probably mid-thought: wait longer before replying.
_TRAILING = {
    "and", "but", "or", "so", "because", "cause", "if", "when", "while", "that", "which", "who", "then", "also",
    "the", "a", "an", "my", "your", "his", "her", "their", "our", "this", "these", "those", "some",
    "to", "of", "for", "with", "about", "from", "at", "in", "on", "by", "like", "than", "as",
    "is", "are", "was", "were", "am", "be", "will", "would", "could", "should", "can", "do", "does", "did",
    "have", "has", "had", "i", "i'm", "we", "we're", "it's", "um", "uh", "er", "hmm", "well", "just", "not",
    "what", "where", "how", "why",
}
# Pure acknowledgements: the caller is listening, not taking the floor.
_BACKCHANNEL = {
    "ok", "okay", "right", "yeah", "yep", "yup", "sure", "mm", "mhm", "mmhmm", "hmm", "uhhuh", "uh", "huh",
    "i", "see", "got", "it", "alright", "cool", "great", "good", "nice", "oh", "ah", "gotcha", "true", "exactly",
    "mm-hmm", "uh-huh",
}
# These short words DO take the floor, even though they are short.
_FLOOR_TAKING = {"no", "wait", "stop", "hold", "sorry", "actually", "but", "excuse", "hang", "yes", "what", "pardon"}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9'\-]+", text.lower())


def looks_complete(text: str) -> bool:
    """False when the utterance trails off ("I was calling about the..."), so we keep listening."""
    t = text.strip()
    if not t:
        return True
    if t.endswith(("...", "…", ",", "-", "—")):
        return False
    words = _words(t)
    if not words:
        return True
    return words[-1].strip("'-") not in _TRAILING


def is_backchannel(text: str) -> bool:
    """'okay', 'mm-hmm', 'right, got it' ... said while the assistant is talking."""
    words = _words(text)
    if not words or len(words) > 3:
        return False
    if any(w in _FLOOR_TAKING for w in words):
        return False
    return all(w in _BACKCHANNEL or w.replace("-", "") in _BACKCHANNEL for w in words)
