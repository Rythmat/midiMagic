"""Compare the rule-based, neural and hybrid engines with the ranking metrics.

Anchor prompts use lead tracks from XMIDI *test-split* files (never seen in training).

    python scripts/evaluate_engines.py --data data/xmidi --xmidi-root XMIDI_Dataset --n-anchors 8
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from midicomposer.generation.pipeline import GenerationRequest, run_generation  # noqa: E402
from midicomposer.midi_io import Score, Track, load_midi, score_to_bytes  # noqa: E402

METRICS = ["total", "quality", "genre", "emotion", "key_consistency", "harmonic_compatibility",
           "rhythmic_compatibility", "structure", "density"]
META_PROMPTS = [("pop", "happy", "D"), ("rock", "angry", "E minor"), ("jazz", "lazy", "Bb"), ("country", "warm", "G"),
                ("classical", "sad", "A minor"), ("traditional", "romantic", "F")]


def anchor_files(data_dir: Path, n: int, seed: int) -> list[dict]:
    recs = []
    with (data_dir / "manifest.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if '"split": "test"' not in line or '"usable_for_training": true' not in line:
                continue
            r = json.loads(line)
            leads = [t for t in r["tracks"] if t["role"] == "lead" and t["n_notes"] >= 40]
            if leads and r["n_bars"] >= 24 and not r["root_ambiguous"]:
                recs.append({"path": r["path"], "genre": r["genre"], "emotion": r["emotion"], "lead": leads[0]["track_id"]})
    random.Random(seed).shuffle(recs)
    return recs[:n]


def lead_excerpt(path: Path, track_id: int, n_bars: int = 16) -> Score:
    """The densest n-bar window of the lead track (XMIDI is all 4/4)."""
    s = load_midi(path)
    t = s.track_by_id(track_id)
    per_bar = np.bincount([int(n.start // 4) for n in t.notes])
    sums = np.convolve(per_bar, np.ones(n_bars, dtype=int), mode="valid") if len(per_bar) >= n_bars else per_bar.sum(keepdims=True)
    start_bar = int(np.argmax(sums))
    a, b = start_bar * 4.0, (start_bar + n_bars) * 4.0
    notes = [type(n)(n.start - a, min(n.duration, b - n.start), n.pitch, n.velocity) for n in t.notes if a <= n.start < b]
    return Score(tracks=[Track(0, t.name, t.program, t.channel, False, notes, role="lead")], tempos=s.tempos[:1],
                 time_signatures=s.time_signatures[:1], ticks_per_beat=s.ticks_per_beat)


def summarize(rows: list[dict]) -> dict:
    out = {}
    for eng in sorted({r["engine"] for r in rows}):
        rs = [r for r in rows if r["engine"] == eng]
        out[eng] = {m: round(float(np.mean([r[m] for r in rs])), 3) for m in METRICS}
        out[eng]["valid_rate"] = round(float(np.mean([r["valid"] for r in rs])), 3)
        out[eng]["seconds"] = round(float(np.mean([r["seconds"] for r in rs])), 2)
        out[eng]["n"] = len(rs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/xmidi")
    ap.add_argument("--xmidi-root", default="XMIDI_Dataset")
    ap.add_argument("--n-anchors", type=int, default=8)
    ap.add_argument("--engines", default="rule,neural,hybrid")
    ap.add_argument("--out", default="reports/engine_comparison")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    engines = a.engines.split(",")
    rows = []
    for genre, emotion, key in META_PROMPTS:
        for eng in engines:
            t = time.time()
            state = run_generation(GenerationRequest(genre=genre, emotion=emotion, key=key, bars=16, n_variations=2,
                                                     seed=a.seed, engine=eng,
                                                     generated_roles=["lead_melody", "chord_bed", "bass_line", "drums"]))
            rows.append(_row("metadata", f"{genre}/{emotion}/{key}", eng, state, time.time() - t))
            print(rows[-1], flush=True)
    for rec in anchor_files(Path(a.data), a.n_anchors, a.seed):
        anchor = score_to_bytes(lead_excerpt(Path(a.xmidi_root) / rec["path"], rec["lead"]))
        for eng in engines:
            t = time.time()
            state = run_generation(GenerationRequest(mode="midi", input_role="lead", genre=rec["genre"], emotion=rec["emotion"],
                                                     n_variations=2, seed=a.seed, engine=eng,
                                                     generated_roles=["chord_bed", "bass_line", "drums"]),
                                   input_bytes=anchor, input_name=rec["path"])
            rows.append(_row("anchor", rec["path"], eng, state, time.time() - t))
            print(rows[-1], flush=True)
    summary = {"metadata": summarize([r for r in rows if r["kind"] == "metadata"]),
               "anchor": summarize([r for r in rows if r["kind"] == "anchor"])}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    lines = ["# Engine comparison", "",
             "Best-ranked candidate (of 2) per prompt; mean over prompts. Anchor prompts are 16-bar lead-track excerpts",
             "from XMIDI **test-split** files (unseen in training). All metrics come from `midicomposer.ranking`.", ""]
    for kind, s in summary.items():
        lines += [f"## {kind} prompts", "", "| metric | " + " | ".join(s) + " |", "|---|" + "---:|" * len(s)]
        for m in METRICS + ["valid_rate", "seconds", "n"]:
            lines.append(f"| {m} | " + " | ".join(str(s[e][m]) for e in s) + " |")
        lines.append("")
    Path(a.out + ".md").write_text("\n".join(lines))
    print("\n".join(lines))


def _row(kind, prompt, eng, state, secs):
    c = state.candidates[0]
    row = {"kind": kind, "prompt": prompt, "engine": eng, "seconds": round(secs, 2), "valid": c.validation["passed"],
           "total": c.scores["total"], "quality": c.scores["quality"], "genre": c.scores["genre"],
           "emotion": c.scores["emotion"]}
    row.update({k: c.scores["components"][k] for k in METRICS if k in c.scores["components"]})
    return row


if __name__ == "__main__":
    main()
