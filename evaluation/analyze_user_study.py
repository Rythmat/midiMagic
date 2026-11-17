"""Summarise evaluation/user_study_results.csv into evaluation/results/user_study.md.

    python evaluation/analyze_user_study.py [path/to/results.csv]

*_done cells: yes / helped / no. SUS and Q cells: 1-5. Empty cells are skipped.
"""
from __future__ import annotations

import csv
import statistics as stats
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = ["T1", "T2", "T3", "T4", "T5"]
QUESTIONS = {"Q1": "Music matched the requested mood", "Q2": "App understood my prompt",
             "Q3": "Added parts fit my melody", "Q4": "Critic feedback was useful", "Q5": "Separate stems are useful",
             "Q6": "More control than a finished-audio tool"}


def sus(row) -> float | None:
    """Standard SUS: odd items score-1, even items 5-score, sum x 2.5."""
    try:
        vals = [int(row[f"SUS{i}"]) for i in range(1, 11)]
    except (KeyError, ValueError):
        return None
    return 2.5 * sum((v - 1) if i % 2 == 0 else (5 - v) for i, v in enumerate(vals))


def summary(values: list[float]) -> str:
    if not values:
        return "–"
    sd = f" ± {stats.stdev(values):.2f}" if len(values) > 1 else ""
    return f"{stats.mean(values):.2f}{sd}"


def main(path: Path) -> str:
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8")) if r.get("participant")]
    if not rows:
        return "No participants recorded yet."
    out = [f"# User study results\n\n{len(rows)} participants "
           f"({sum(r.get('musical_training', '').lower() == 'yes' for r in rows)} with musical training, "
           f"{sum(r.get('daw_experience', '').lower() == 'yes' for r in rows)} with DAW experience).\n"]
    scores = [s for s in (sus(r) for r in rows) if s is not None]
    out.append(f"**SUS score: {summary(scores)}** (0–100; ~68 is average usability), n={len(scores)}.\n")
    out.append("| task | completed alone | completed with help | not completed | time (s), mean ± sd |\n|---|---|---|---|---|")
    for t in TASKS:
        done = [r.get(f"{t}_done", "").strip().lower() for r in rows if r.get(f"{t}_done")]
        secs = [float(r[f"{t}_seconds"]) for r in rows if r.get(f"{t}_seconds")]
        n = max(1, len(done))
        out.append(f"| {t} | {done.count('yes') / n:.0%} | {done.count('helped') / n:.0%} | {done.count('no') / n:.0%} "
                   f"| {summary(secs)} |")
    out.append("\n| question (1–5) | mean ± sd | n |\n|---|---|---|")
    for q, text in QUESTIONS.items():
        vals = [float(r[q]) for r in rows if r.get(q)]
        out.append(f"| {q}. {text} | {summary(vals)} | {len(vals)} |")
    notes = [(r["participant"], r.get("most_confusing", ""), r.get("suggestion", "")) for r in rows]
    out.append("\n## Comments\n")
    out += [f"* **{p}** — confusing: {c or '–'}; suggestion: {s or '–'}" for p, c, s in notes]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "user_study_results.csv"
    md = main(src)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "user_study.md").write_text(md, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(md)
