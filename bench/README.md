# bench

The offline benchmarks behind [docs/benchmarks.md](../docs/benchmarks.md). They use synthetic voices, touch no phone, and save timings and counts only.

```bash
.venv/bin/pip install -r requirements-bench.txt

.venv/bin/python -m bench.run_stt        # speech recognition engines, by line condition and accent
.venv/bin/python -m bench.run_tts        # voices: time to first phrase, intelligibility after a phone line
.venv/bin/python -m bench.run_llm        # models: speed, memory, and the 11 rule checks
.venv/bin/python -m bench.run_e2e        # the whole pipeline against a scripted caller, in real time
.venv/bin/python -m bench.make_charts    # redraws the SVG charts in assets/ from results/
```

| File | What it is |
|---|---|
| `corpus.py` | The scripted sentences and the simulated phone line (band-limit, 8 kHz, G.711, optional noise) |
| `run_e2e.py` | The one that matters. A fake caller and the real pipeline exchange 32 ms frames, so the delays are what a caller would hear |
| `results/*.json` | Output of each run. `e2e-final-qwen3-8b.json` is the configuration in use |
| `make_charts.py` | Standard library only. Writes the two charts as plain SVG |

The language model benchmarks need Ollama running with the models pulled. A full end-to-end run takes 15 to 20 minutes per model and will make the laptop warm.

One caveat on the results folder: `e2e-phi3.json` and `e2e-llama32-3b.json` were recorded with an earlier wording of the rules, so they aren't strictly like-for-like with the two `e2e-final-*` files.

Two of the scripted caller sentences were reworded slightly after the results were recorded. The change is a few words and should not move the numbers, but a re-run will not match to the millisecond.
