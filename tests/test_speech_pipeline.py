"""Conversation pipeline logic, with fake models and a fake clock. No audio hardware, no real
models, no real time: every frame advances the clock by exactly 32 ms."""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from phone_assistant.speech.audio_out import AudioOut
from phone_assistant.speech.chunker import PhraseChunker, speakable
from phone_assistant.speech.interfaces import FRAME, SAMPLE_RATE
from phone_assistant.speech.pipeline import FALLBACK_LINE, ConversationPipeline, PipelineConfig
from phone_assistant.speech.turns import is_backchannel, looks_complete

FRAME_S = FRAME / SAMPLE_RATE
SPEECH = np.full(FRAME, 0.5, dtype=np.float32)
SILENCE = np.zeros(FRAME, dtype=np.float32)


class FakeVAD:
    def reset(self): pass
    def prob(self, frame): return 1.0 if frame[0] > 0.1 else 0.0


class FakeSTT:
    name = "fake"
    def __init__(self, texts): self.texts, self.calls = list(texts), 0
    def transcribe(self, audio):
        self.calls += 1
        return self.texts.pop(0) if self.texts else ""


class FakeLLM:
    name = "fake"
    def __init__(self, replies, fail=False): self.replies, self.calls, self.fail, self.seen = list(replies), 0, fail, []
    async def stream(self, messages, num_predict=None):
        self.calls += 1
        self.seen.append(messages)
        if self.fail:
            raise RuntimeError("model crashed")
        reply = self.replies.pop(0) if self.replies else "Okay."
        for word in reply.split(" "):
            await asyncio.sleep(0)
            yield word + " "


class FakeTTS:
    """Each synthesized chunk is 0.5 s of a constant unique to that chunk, so tests can tell
    exactly which chunk a played sample came from."""
    name, sample_rate = "fake", SAMPLE_RATE
    def __init__(self): self.calls, self.texts = 0, []
    def synthesize(self, text):
        self.calls += 1
        self.texts.append(text)
        return np.full(SAMPLE_RATE // 2, self.calls / 100.0, dtype=np.float32)


class Rig:
    def __init__(self, stt_texts, llm_replies, llm_fail=False, **cfg):
        self.t = 0.0
        self.out = AudioOut(SAMPLE_RATE, clock=lambda: self.t)
        self.stt, self.llm, self.tts = FakeSTT(stt_texts), FakeLLM(llm_replies, llm_fail), FakeTTS()
        self.signals = []
        self.pipe = ConversationPipeline(vad=FakeVAD(), stt=self.stt, llm=self.llm, tts=self.tts, out=self.out,
                                         system_prompt="SYS", cfg=PipelineConfig(chunk_pause_s=0.0, **cfg),
                                         clock=lambda: self.t, on_event=self.signals.append)
        self.heard = []          # every sample the caller would hear, in order

    async def feed(self, frame, seconds):
        for _ in range(round(seconds / FRAME_S)):
            self.pipe.feed(frame)
            for _ in range(6):                      # let worker threads and tasks make progress
                await asyncio.sleep(0.002)
            self.heard.append(self.out.pull(FRAME))
            self.t += FRAME_S

    async def speak(self, seconds): await self.feed(SPEECH, seconds)
    async def quiet(self, seconds): await self.feed(SILENCE, seconds)

    def heard_values(self):
        """Chunk markers heard, in order. Only steady stretches count, so the short fade ramps
        (which pass through other values) and silence are ignored."""
        x = np.round(np.concatenate(self.heard) * 100, 3) if self.heard else np.zeros(0)
        vals, i = [], 0
        while i < len(x):
            j = i
            while j < len(x) and x[j] == x[i]:
                j += 1
            v = float(x[i])
            if j - i >= 200 and v >= 1 and v == int(v) and (not vals or vals[-1] != int(v)):
                vals.append(int(v))
            i = j
        return vals

    def names(self): return [n for _, n, _ in self.pipe.events]


# ------------------------------------------------------------------ basic turn
async def test_one_turn_is_answered_and_recorded_in_memory():
    r = Rig(["Is Adham available?"], ["He prefers text or email. Can I take a message?"])
    await r.speak(1.0)
    await r.quiet(3.0)
    assert r.stt.calls == 1 and r.llm.calls == 1
    assert r.heard_values() == [1, 2]                         # two sentences, in order
    assert [m["role"] for m in r.pipe.history] == ["user", "assistant"]
    assert r.llm.seen[0][0] == {"role": "system", "content": "SYS"}
    await r.pipe.close()


async def test_nothing_is_spoken_before_the_turn_is_confirmed():
    r = Rig(["Is Adham available?"], ["Yes. He is."])
    await r.speak(1.0)
    await r.quiet(0.5)                                        # reply is prepared but held (< 0.6 s)
    assert r.llm.calls == 1 and r.tts.calls >= 1              # work happened speculatively...
    assert r.heard_values() == []                             # ...but the caller heard nothing
    await r.quiet(0.3)
    assert r.heard_values() != []
    await r.pipe.close()


async def test_reply_starts_at_the_endpoint_threshold_not_later():
    r = Rig(["Is Adham available?"], ["Yes. He is."])
    await r.speak(1.0)
    end_of_speech = r.t
    await r.quiet(2.0)
    gen = [g for _, n, g in r.pipe.events if n == "commit"][0]
    delay = r.out.first_audio_at[gen] - end_of_speech
    assert 0.55 <= delay <= 0.75                              # the 0.6 s threshold plus one or two frames
    await r.pipe.close()


# ------------------------------------------------------------------ caller keeps talking
async def test_caller_resuming_discards_the_prepared_reply():
    r = Rig(["I wanted to ask.", "I wanted to ask. Is he free Thursday?"], ["WRONG reply.", "Right reply."])
    await r.speak(1.0)
    await r.quiet(0.45)                                       # short pause: speculation runs
    await r.speak(1.0)                                        # ...but they continue
    await r.quiet(3.0)
    assert r.stt.calls == 2 and r.llm.calls == 2
    assert r.tts.texts[-1] == "Right reply."
    heard = r.heard_values()
    assert len(heard) == 1 and heard[0] == r.tts.texts.index("Right reply.") + 1   # the wrong one was never heard
    assert [m["content"] for m in r.pipe.history if m["role"] == "user"] == ["I wanted to ask. Is he free Thursday?"]
    assert "speculation_cancelled" in r.names()
    await r.pipe.close()


async def test_trailing_off_gets_more_time():
    r = Rig(["I was calling about the", "I was calling about the invoice."], ["Sure."])
    await r.speak(1.0)
    await r.quiet(1.2)                                        # past 0.6 s but the sentence is unfinished
    assert r.heard_values() == [] and r.pipe.history == []
    await r.speak(0.6)
    await r.quiet(2.0)
    assert r.pipe.history[0]["content"] == "I was calling about the invoice."
    await r.pipe.close()


async def test_trailing_off_is_still_answered_after_the_longer_wait():
    r = Rig(["I was calling about the"], ["Could you tell me a bit more?"])
    await r.speak(1.0)
    await r.quiet(3.0)
    assert r.heard_values() == [1]
    await r.pipe.close()


async def test_noise_is_ignored():
    r = Rig([""], ["Should not be said."])
    await r.speak(0.4)
    await r.quiet(2.0)
    assert r.llm.calls == 0 and r.heard_values() == [] and r.pipe.history == []
    await r.pipe.close()


# ------------------------------------------------------------------ interruptions
LONG = "One one one. Two two two. Three three three. Four four four. Five five five. Six six six."


async def test_barge_in_stops_the_assistant_and_answers_the_new_request():
    r = Rig(["Tell me about him.", "No wait, what's his email?"], [LONG, "Please text this number."])
    await r.speak(1.0)
    await r.quiet(1.6)                                        # assistant is now mid-answer
    assert r.out.audible
    before = len(r.heard)
    await r.speak(1.0)                                        # caller talks over it
    during = np.concatenate(r.heard[before:])
    silent_from = int(0.45 * SAMPLE_RATE)                     # 0.3 s to be sure + fade + one frame
    assert np.all(np.abs(during[silent_from:]) < 1e-6)        # the assistant went quiet promptly
    await r.quiet(3.0)
    assert r.tts.texts[-1] == "Please text this number."
    new_marker = r.tts.texts.index("Please text this number.") + 1
    heard = r.heard_values()
    assert heard[-1] == new_marker
    assert not {4, 5, 6} & set(heard)                         # the cancelled answer's later sentences never played
    roles = [(m["role"], m["content"]) for m in r.pipe.history]
    assert roles[1][0] == "assistant" and roles[1][1].endswith("—")      # recorded as cut off
    assert roles[2] == ("user", "No wait, what's his email?")
    assert "bargein" in r.names() and "interrupted" in r.names()
    await r.pipe.close()


async def test_backchannel_does_not_derail_the_assistant():
    r = Rig(["Tell me about him.", "Okay."], [LONG])
    await r.speak(1.0)
    await r.quiet(1.6)
    await r.speak(0.5)                                        # "okay"
    await r.quiet(5.0)
    assert r.llm.calls == 1                                   # no new reply was generated
    assert r.heard_values() == [1, 2, 3, 4, 5, 6]             # the whole answer, in order, nothing repeated
    assert "backchannel" in r.names() and "interrupted" not in r.names()
    assert [m["role"] for m in r.pipe.history] == ["user", "assistant"]
    await r.pipe.close()


async def test_a_cough_while_the_assistant_talks_resumes_it():
    r = Rig(["Tell me about him.", ""], [LONG])
    await r.speak(1.0)
    await r.quiet(1.6)
    await r.speak(0.5)
    await r.quiet(5.0)
    assert r.heard_values() == [1, 2, 3, 4, 5, 6] and r.llm.calls == 1
    await r.pipe.close()


async def test_stale_audio_never_plays_in_a_later_turn():
    r = Rig(["First question.", "Second question entirely."], [LONG, "Second answer."])
    await r.speak(1.0)
    await r.quiet(1.2)
    await r.speak(1.2)
    await r.quiet(4.0)
    heard = r.heard_values()
    second = r.tts.texts.index("Second answer.") + 1
    cut = heard.index(second)
    assert all(v == second for v in heard[cut:])              # once the new turn starts, only its audio
    await r.pipe.close()


async def test_greeting_is_spoken_and_can_be_interrupted():
    r = Rig(["Hi, quick question."], ["Go ahead."])
    await r.pipe.say("Hello there. I am an assistant. How can I help?")
    await r.quiet(0.4)
    assert r.out.audible
    await r.speak(0.8)
    await r.quiet(2.5)
    assert r.pipe.history[0]["role"] == "assistant" and r.pipe.history[0]["content"].endswith("—")
    assert r.tts.texts[-1] == "Go ahead."
    await r.pipe.close()


# ------------------------------------------------------------------ failures and silence
async def test_model_failure_says_a_fallback_line_then_asks_to_end_the_call():
    r = Rig(["Hello?"], [], llm_fail=True)
    await r.speak(0.8)
    await r.quiet(8.0)
    assert FALLBACK_LINE.split(".")[0] in " ".join(r.tts.texts)
    assert "idle_timeout" in r.signals
    await r.pipe.close()


async def test_silent_caller_gets_one_prompt_then_the_call_is_released():
    r = Rig([], [], idle_prompt_s=2.0, idle_hangup_s=5.0)
    await r.pipe.say("Hello.")
    await r.quiet(4.0)
    assert r.tts.texts == ["Hello.", "Are you still there?"] and "idle_timeout" not in r.signals
    await r.quiet(6.0)
    assert r.signals.count("idle_timeout") == 1 and r.tts.calls == 2
    await r.pipe.close()


# ------------------------------------------------------------------ privacy
async def test_close_wipes_conversation_and_audio():
    r = Rig(["My number is 617 555 0142."], ["Thanks, I'll pass that on."])
    await r.speak(1.0)
    await r.quiet(1.0)
    history, utt_frames = r.pipe.history, r.pipe._preroll
    await r.pipe.close()
    assert history == [] and len(utt_frames) == 0 and r.pipe._utt == []
    assert r.out.pending_seconds() == 0 and not r.out.audible
    r.pipe.feed(SPEECH)                                       # frames after close are ignored
    assert r.stt.calls == 1


async def test_events_and_logs_hold_no_words(caplog):
    import logging
    with caplog.at_level(logging.DEBUG):
        r = Rig(["My secret is pineapple."], ["Understood, pineapple noted."])
        await r.speak(1.0)
        await r.quiet(3.0)
        blob = repr(list(r.pipe.events)) + "\n".join(rec.getMessage() for rec in caplog.records)
        await r.pipe.close()
    assert "pineapple" not in blob.lower() and "secret" not in blob.lower()


# ------------------------------------------------------------------ small pieces
def test_chunker_cuts_the_first_phrase_early_then_whole_sentences():
    c = PhraseChunker()
    out = []
    for piece in "Sure, I can help with that, no problem. He prefers email. Dr. Lee said 3.5 is fine. Bye".split(" "):
        out += c.feed(piece + " ")
    out += c.flush()
    assert out[0] == "Sure, I can help with that,"              # early cut once 4 words exist
    assert out[1:] == ["no problem.", "He prefers email.", "Dr. Lee said 3.5 is fine.", "Bye"]


def test_speakable_strips_markup():
    assert speakable("**Sure!** Here's a [link](http://x.y) and `code`.") == "Sure! Here's a and code."


def test_turn_heuristics():
    assert looks_complete("Is he available?") and looks_complete("Okay.")
    assert not looks_complete("I was calling about the") and not looks_complete("So, um,") and not looks_complete("and then...")
    for b in ["Okay.", "Mm-hmm.", "Right, got it.", "Yeah", "uh-huh", "I see."]:
        assert is_backchannel(b), b
    for nb in ["No.", "Wait.", "Yes.", "Hold on", "Actually", "What's his email?", "Okay but when", ""]:
        assert not is_backchannel(nb), nb


def test_audio_out_only_plays_the_released_generation():
    out = AudioOut(1000, fade_ms=10)
    out.submit(1, 0, np.full(100, 0.1, np.float32))
    assert not out.audible and not out.pull(50).any()          # held
    out.submit(2, 0, np.full(100, 0.2, np.float32))
    out.release(2)                                             # generation 1 is dropped, never heard
    x = np.concatenate([out.pull(50) for _ in range(4)])
    assert {round(float(v), 3) for v in x[x != 0]} == {0.2}
    out.finish(2)
    assert out.idle(2)


def test_audio_out_pause_fades_and_resume_continues_without_loss():
    out = AudioOut(1000, fade_ms=10)
    out.submit(1, 0, np.full(1000, 0.5, np.float32)); out.finish(1); out.release(1)
    a = out.pull(100)
    out.pause()
    b = out.pull(100)                                          # 10-sample fade, then silence
    assert b[0] > b[9] >= 0 and not b[10:].any() and not out.pull(100).any()
    out.resume()
    rest = np.concatenate([out.pull(100) for _ in range(10)])
    assert int(np.count_nonzero(a)) + 10 + int(np.count_nonzero(rest)) >= 998      # nothing skipped
    assert out.idle(1)


def test_audio_out_counts_gaps_only_while_a_reply_is_unfinished():
    out = AudioOut(1000)
    out.submit(1, 0, np.full(100, 0.5, np.float32)); out.release(1)
    out.pull(100); out.pull(100)                               # starved: more audio was promised
    assert out.gap_events == 1 and out.gap_seconds == pytest.approx(0.1)
    out.submit(1, 1, np.full(100, 0.5, np.float32)); out.finish(1)
    out.pull(100); out.pull(100)                               # finished: silence afterwards is not a gap
    assert out.gap_events == 1


# ------------------------------------------------------------------ context handling
async def test_history_is_trimmed_in_blocks_not_every_turn():
    r = Rig([], [], max_history_messages=8)
    r.pipe.history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(8)]
    old = r.pipe.history[0]
    r.pipe._trim_history()
    assert len(r.pipe.history) == 8                           # at the limit: untouched, the model's cached start stays valid
    r.pipe.history.append({"role": "user", "content": "m8"})
    r.pipe._trim_history()
    assert [m["content"] for m in r.pipe.history] == ["m5", "m6", "m7", "m8"]
    assert old["content"] == ""                               # dropped turns are wiped, not just unlinked
    await r.pipe.close()


async def test_priming_the_model_speaks_nothing_and_records_nothing():
    r = Rig([], ["x"])
    await r.pipe.prime("Hello, I'm the assistant.")
    await r.quiet(0.5)
    assert r.llm.calls == 1 and r.tts.calls == 0 and r.heard_values() == [] and r.pipe.history == []
    assert [m["role"] for m in r.llm.seen[0]] == ["system", "assistant", "user"]
    await r.pipe.close()


async def test_an_empty_model_reply_still_gets_a_spoken_answer():
    r = Rig(["Is Adham available?"], ["   "])
    await r.speak(1.0)
    await r.quiet(3.0)
    assert r.tts.texts == ["Sorry, could you say that again?"] and r.heard_values() == [1]
    assert "empty_reply" in r.names()
    await r.pipe.close()


async def test_a_model_that_hangs_is_treated_as_failed_not_left_in_silence():
    r = Rig(["Is Adham available?"], [], llm_first_token_timeout_s=0.05)

    async def hang(messages, num_predict=None):
        await asyncio.sleep(30)
        yield "never"
    r.llm.stream = hang
    await r.speak(1.0)
    await r.quiet(7.0)
    assert r.tts.texts[0] == FALLBACK_LINE and "idle_timeout" in r.signals
    await r.pipe.close()


# ------------------------------------------------------------------ call ends mid-speech
async def test_caller_hanging_up_while_the_reply_is_being_made_leaves_nothing_to_play():
    r = Rig(["Is Adham available?"], ["One. Two. Three. Four. Five. Six."])
    await r.speak(1.0)
    await r.quiet(0.9)                         # reply released and still being synthesized / played
    assert r.pipe.assistant_active
    await r.pipe.close()                       # the call ended
    before = r.tts.calls
    r.heard.clear()
    for _ in range(40):                        # the audio device keeps pulling for a while
        await asyncio.sleep(0.002)
        r.heard.append(r.out.pull(FRAME))
    assert not np.any(np.concatenate(r.heard)) # pure silence after the call
    assert r.tts.calls <= before + 1           # at most the chunk already in flight; nothing new started
    assert r.pipe.history == [] and r.pipe._utt == []
    r.pipe.feed(SPEECH)                        # a closed pipeline ignores further audio
    assert r.names().count("speech_start") == 1
