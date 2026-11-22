"""Draws the two charts used in the READMEs from bench/results/*.json.

No plotting library: it writes plain SVG, so it runs with the standard library alone.

    python -m bench.make_charts
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"
OUT = ROOT / "assets"

STYLE = """
<style>
  .bg{fill:#fcfcfb} .t1{fill:#0b0b0b} .t2{fill:#52514e} .grid{stroke:#e6e5e0} .band{fill:#f0efec}
  .s1{fill:#2a78d6} .s2{fill:#eb6834} .link{stroke:#c3c2b7} .ring{stroke:#fcfcfb}
  text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
  @media (prefers-color-scheme: dark){
    .bg{fill:#1a1a19} .t1{fill:#f0efec} .t2{fill:#c3c2b7} .grid{stroke:#383835} .band{fill:#262624}
    .s1{fill:#3987e5} .s2{fill:#d95926} .link{stroke:#52514e} .ring{stroke:#1a1a19}
  }
</style>"""


def _svg(w: int, h: int, title: str, body: list[str]) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{title}">{STYLE}\n<rect class="bg" width="{w}" height="{h}" rx="8"/>\n'
            + "\n".join(body) + "\n</svg>\n")


def _bar(x: float, y: float, w: float, h: float, r: float = 4) -> str:
    """Horizontal bar: square at the baseline (left), rounded at the data end (right)."""
    if w <= 0:
        return ""
    r = min(r, w)
    return (f'<path class="s1" d="M{x},{y} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{h - 2 * r} '
            f'a{r},{r} 0 0 1 -{r},{r} h-{w - r:.1f} z"/>')


def delay_buckets() -> str:
    s = json.loads((RESULTS / "e2e-final-qwen3-8b.json").read_text())["summary"]
    rows = [("Under 1 s", s["under_1s"]), ("1 to 2 s", s["1_to_2s"]),
            ("2 to 3 s", s["2_to_3s"]), ("Over 3 s", s["over_3s"])]
    total = s["turns"]
    w, h, left, top, row_h, bar_h = 720, 250, 110, 78, 38, 20
    scale = (w - left - 130) / max(n for _, n in rows)
    body = [f'<text class="t1" x="24" y="34" font-size="17" font-weight="600">How long the caller waits for a reply</text>',
            f'<text class="t2" x="24" y="56" font-size="13">{total} turns of scripted conversation, qwen3:8b. '
            f'Median {s["median_s"]:.1f} s, slowest {s["max_s"]:.1f} s.</text>',
            f'<line class="grid" x1="{left}" y1="{top - 6}" x2="{left}" y2="{top + row_h * len(rows) - 6}"/>']
    for i, (label, n) in enumerate(rows):
        y = top + i * row_h
        body.append(f'<text class="t2" x="{left - 12}" y="{y + 15}" font-size="13" text-anchor="end">{label}</text>')
        body.append(_bar(left, y, n * scale, bar_h))
        body.append(f'<text class="t1" x="{left + n * scale + 8}" y="{y + 15}" font-size="13">'
                    f'{n} turn{"" if n == 1 else "s"} ({round(100 * n / total)}%)</text>')
    return _svg(w, h, "Reply delay, grouped by second", body)


def model_delays() -> str:
    runs = [("llama3.2:3b", "e2e-llama32-3b.json"), ("phi3 (3.8B)", "e2e-phi3.json"),
            ("qwen3:8b  (in use)", "e2e-final-qwen3-8b.json"), ("gemma3:4b", "e2e-final-gemma3-4b.json"),
            ("gemma3:4b, no head start", "e2e-gemma3-4b-nospec.json")]
    w, left, top, row_h = 720, 200, 104, 34
    h = top + row_h * len(runs) + 44
    x0, x1, vmax = left, w - 40, 2.5

    def X(v: float) -> float:
        return x0 + (x1 - x0) * v / vmax

    bottom = top + row_h * len(runs) - 8
    body = ['<text class="t1" x="24" y="34" font-size="17" font-weight="600">Reply delay by language model</text>',
            '<text class="t2" x="24" y="56" font-size="13">Seconds from the caller going quiet to the first sound '
            'of the reply. Lower is better.</text>',
            # legend
            f'<circle class="s1" cx="30" cy="78" r="5"/><text class="t2" x="42" y="82" font-size="13">Median</text>',
            f'<circle class="s2" cx="112" cy="78" r="5"/><text class="t2" x="124" y="82" font-size="13">95th percentile</text>',
            f'<rect class="band" x="{X(1):.1f}" y="{top - 10}" width="{X(2) - X(1):.1f}" height="{bottom - top + 10}"/>',
            f'<text class="t2" x="{(X(1) + X(2)) / 2:.1f}" y="{top - 16}" font-size="12" text-anchor="middle">target: 1 to 2 s</text>']
    for v in (0, 0.5, 1, 1.5, 2, 2.5):
        body.append(f'<line class="grid" x1="{X(v):.1f}" y1="{top - 10}" x2="{X(v):.1f}" y2="{bottom}"/>')
        body.append(f'<text class="t2" x="{X(v):.1f}" y="{bottom + 18}" font-size="12" text-anchor="middle">{v:g} s</text>')
    for i, (label, fname) in enumerate(runs):
        s = json.loads((RESULTS / fname).read_text())["summary"]
        y = top + i * row_h + 8
        weight = ' font-weight="600"' if "in use" in label else ""
        body.append(f'<text class="t1" x="{left - 14}" y="{y + 4}" font-size="13" text-anchor="end"{weight}>{label}</text>')
        body.append(f'<line class="link" x1="{X(s["median_s"]):.1f}" y1="{y}" x2="{X(s["p95_s"]):.1f}" y2="{y}" stroke-width="2"/>')
        body.append(f'<circle class="s1 ring" cx="{X(s["median_s"]):.1f}" cy="{y}" r="6" stroke-width="2"/>')
        body.append(f'<circle class="s2 ring" cx="{X(s["p95_s"]):.1f}" cy="{y}" r="6" stroke-width="2"/>')
    return _svg(w, h, "Median and 95th percentile reply delay for each language model", body)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    (OUT / "reply-delay.svg").write_text(delay_buckets())
    (OUT / "model-delays.svg").write_text(model_delays())
    print("wrote assets/reply-delay.svg and assets/model-delays.svg")
