# tools

Small scripts for checking the setup before trusting it with a call. None of them answers, dials or presses anything in the phone app.

| Script | Use it when | What it touches |
|---|---|---|
| `preflight.py` | Right before a real call | Nothing. Prints a checklist: settings, permissions, DIGITS state, audio devices, model files |
| `native_check.py` | DIGITS seems out of sync with the bot | Nothing. One reading of the app state, or `--watch 60` to print changes for a minute |
| `audio_check.py` | Setting up BlackHole, or after a crash | Read-only by default. `--signal` sends a test tone through the two virtual devices. `--restore` puts the Mac's default devices back |
| `device_path_test.py` | You want to hear the pipeline work without a phone | Opens the two virtual devices only. Runs a greeting and a short synthetic conversation with stand-ins for DIGITS |
| `ax_dump.py` | DIGITS updated and the bot stopped recognising a button | Nothing. Dumps the app's accessibility tree (roles and labels) |

```bash
.venv/bin/python tools/preflight.py
```

A healthy run ends with `0 check(s) need attention`. Lines marked `[!!]` say what to fix.

`ax_dump.py --full` prints text values too, which can include caller names and numbers if a call is on screen. The default prints control labels only.
