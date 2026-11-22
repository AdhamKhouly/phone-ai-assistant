# Safety and privacy notes

A program that can pick up my phone and talk as my representative needs tighter rules than most side projects. This page lists the rules, where they are enforced, and the places where the guarantee is weaker than it sounds.

## What can trigger a real phone action

| Action | The only trigger | Extra checks |
|---|---|---|
| Answer a call | My `answer` message in Discord | From my user ID, in one of two channels, sent after the ring started, under 15 s old, inside the 22 s answer window, and the same caller still ringing |
| Place a call | My `call ...`, then my `confirm` | A purpose is required. The prepared call lives in memory and expires after 60 s |
| End a call | My `hangup`, the other side, or a guard | The assistant can only end a call it answered or placed |
| Decline a call | Nothing | The Reject button is never used. `ignore` just leaves the call alone |

If the bot is off, crashed or confused, the phone behaves exactly as it did before this project existed. It never changes forwarding, ring time or Do Not Disturb.

No automatic retries anywhere. One press of Accept, one dial attempt. If it fails, I get told and the call is left to voicemail.

## What is kept

Nothing about what was said.

- No audio files. `.gitignore` blocks audio extensions as a backstop, and no code path writes one.
- No transcripts. Speech and replies sit in bounded memory buffers during the call and are cleared when it ends.
- Logs go to the terminal only, with phone numbers masked. There are no log files and no telemetry.
- Error messages are built from exception type names, not their text, so a failure can't leak a sentence into a log.
- The benchmarks use synthetic voices and store timings and counts only.

The one thing that leaves the Mac is the Discord message to me. With `CALL_SUMMARY=categories` that is a line built from fixed phrases. With `detailed`, which I use, the local model writes up to four sentences in its own words. Links are stripped and mentions are defused before it's sent, and the note is not saved locally. I chose this knowing that a paraphrase of the call ends up on Discord's servers.

## The microphone, honestly

The claim I can make: on every real call so far, the only audio devices DIGITS opened were the two virtual ones, and no physical microphone was running.

The claim I can't make: that the built-in microphone is impossible to open. macOS grants microphone permission per app, not per device. Once DIGITS is allowed to use a microphone, it is technically able to open any of them.

So the protection is two layers, not a wall:

1. For the length of the call, the Mac's default input is a virtual device, and that is what DIGITS picks.
2. `MicGuard` checks every 0.25 s what DIGITS actually has open. If it is anything other than the virtual microphone, or if macOS can't say and a physical input is live, the call is ended.

Worst case, a real microphone could be live for a fraction of a second before the call is cut. That hasn't happened, but I would rather say it than pretend otherwise.

## The model is not trusted either

The language model is a small local one and can be wrong, so the rules don't rely on it behaving.

- It has no tools. It can't see files, press buttons, dial, or end a call by saying something. All it produces is text that becomes speech.
- Caller speech is always conversation, never instruction. The prompt says so, and since the model can't do anything, a successful "ignore your rules" gets the caller a strange sentence and nothing more.
- On incoming calls it may only state facts from an approved list. On outgoing calls it may only use what I typed for that call.
- It is told to say it is an AI and not me, and to say nothing about my health beyond one sentence: that I'm deaf and handle things in writing.
- The summary it writes is cleaned and length-capped before it reaches Discord.

It still makes mistakes. In offline tests of outgoing calls, an early prompt invented a phone number and a pizza crust. The current one hasn't in repeated runs, but "hasn't in my tests" is the strength of that claim.

## If something goes wrong

| Failure | What happens |
|---|---|
| Bot crashes during a call | The watchdog process quits DIGITS, which drops the call, and restores the Mac's audio devices |
| Virtual device missing | The call is not answered. I'm told why |
| DIGITS never opens the virtual mic | The AI says nothing and the call is ended after 20 s |
| Ollama down or the model errors | A short spoken fallback line, then the call ends |
| Audio left on BlackHole after a hard crash | Fixed at the next start, or with `tools/audio_check.py --restore` |
| Discord token leaks | Reset it in the developer portal. Commands are only accepted from one user ID, so the token alone can't answer or place a call |

## Not covered

- The watchdog and the microphone guard have only fired in tests, never on a real call.
- Anyone with access to my unlocked Mac or my Discord account can use this. That is the trust boundary.
- I have not had this reviewed by anyone who does security for a living.
- Recording laws differ by place. This doesn't record, and the AI announces itself, but check your own situation before pointing something like this at strangers.
