# tests

239 tests. They need no phone, no Discord, no network and no models, and take a couple of minutes.

```bash
.venv/bin/pip install -r requirements-voice.txt -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

Everything outside the code under test is faked: a fake DIGITS app with a fake accessibility tree, a fake Core Audio, fake speech engines, and a clock that the tests move by hand.

| File | Tests | Covers |
|---|---|---|
| `test_digits_native.py` | 76 | The native DIGITS backend: spotting a ring, answering once, hanging up, the watchdog, relaunching the app |
| `test_audio_and_voice.py` | 49 | Audio routing, the microphone guard, the voice session, call summaries, outgoing calls |
| `test_control_and_privacy.py` | 29 | Command parsing, who is allowed to send commands, number masking in logs |
| `test_speech_pipeline.py` | 27 | Turn-taking, interruptions, phrase chunking, stale audio never being played |
| `test_call_manager.py` | 23 | The call state machine and the answer window |
| `test_manual_answer.py` | 23 | The older Chrome backend, against a local fake page |
| `test_digits_observer.py` | 12 | Parsing caller details in the Chrome backend |

Most of these are about what must *not* happen. A few I'd point at:

- `test_press_sites_are_exactly_accept_and_end` reads the backend's source and fails if anything other than Accept and End can be pressed.
- The stale-command tests replay an old `answer` message against a new call and check it is refused.
- The privacy tests check log output for full phone numbers and for anything that was said.
- The summary tests check that what reaches Discord is cleaned, length-capped and labelled as a paraphrase, and that it falls back to fixed phrases if the model fails.

`test_manual_answer.py` uses Playwright with your installed Chrome. If Chrome isn't there, those tests are the ones that will complain.

Two tests look at your real `.env` if one exists, reading only a few named settings, and skip themselves if it doesn't. On a fresh clone you'll see 237 passed, 2 skipped.
