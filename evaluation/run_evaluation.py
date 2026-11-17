"""MIDI Magic evaluation (proposal slide 7). Writes evaluation/results/results.{json,md}.

    python evaluation/run_evaluation.py                 # all tests, rule + hybrid engines (~2-4 min on a laptop CPU)
    python evaluation/run_evaluation.py --engines rule  # faster

Test 1  Melodic lines in isolation: key detection, then chords/bass/drums generated around each fixed melody.
Test 2  Prompts and chords in isolation: (a) text-planner accuracy against hand-labelled settings,
        (b) does the requested emotion reach the music? (emotion classifier on the output, vs. no emotion given),
        (c) melody + bass generated over fixed chord progressions.
Test 3  The whole pipeline through the API, as the GUI drives it: prompt -> plan -> generate -> critic ->
        build on this -> stems. Latency and success of every step, and what the critic says.
        The user-study half of test 3 is in evaluation/user_study.md.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.test_sets import MELODIES, PROGRESSIONS, PROMPTS, line_to_score, progression_to_score  # noqa: E402
from midicomposer.agents import critique, plan_from_text, read_emotion  # noqa: E402
from midicomposer.api.app import XMIDI_DEFAULT_EMOTIONS, XMIDI_DEFAULT_GENRES  # noqa: E402
from midicomposer.classifiers import load_scorer  # noqa: E402
from midicomposer.generation.pipeline import DEFAULT_CLASSIFIER, GenerationRequest, run_generation  # noqa: E402
from midicomposer.midi_io import score_to_bytes  # noqa: E402
from midicomposer.root import detect_root  # noqa: E402

GENRES, EMOTIONS = XMIDI_DEFAULT_GENRES, XMIDI_DEFAULT_EMOTIONS
PLAN_FIELDS = ["genre", "emotion", "key", "scale_mode", "tempo", "bars", "density"]


def best(state) -> dict:
    """Metrics of the top-ranked candidate."""
    c = state.candidates[0]
    sc = c.scores
    comp = sc.get("components", {})
    return {"total": sc["total"], "quality": sc["quality"], "emotion_fit": sc["emotion"], "genre_fit": sc["genre"],
            "harmonic": comp.get("harmonic_compatibility"), "rhythmic": comp.get("rhythmic_compatibility"),
            "key_consistency": comp.get("key_consistency"), "structure": comp.get("structure"),
            "valid": bool(c.validation.get("passed")),
            "heard_emotions": list(((sc.get("classifier_top") or {}).get("emotion") or {}).keys())}


def mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(float(np.mean(vals)), 3) if vals else None


# ----------------------------------------------------------------------------- test 1

def test1(engines) -> dict:
    clf = load_scorer(DEFAULT_CLASSIFIER)
    rows = []
    for name, notes, meter, tempo, pc, fam in MELODIES:
        score = line_to_score(notes, meter, tempo)
        root = detect_root(score)
        style = read_emotion(score, root, clf, EMOTIONS)
        emo = style["emotion"][0]["label"] if style["emotion"] else "happy"
        genre = style["genre"][0]["label"] if style["genre"] else "pop"
        for eng in engines:
            t0 = time.time()
            st = run_generation(GenerationRequest(mode="midi", input_role="lead", engine=eng, n_variations=3, seed=0,
                                                  genre=genre, emotion=emo,
                                                  generated_roles=["chord_bed", "bass_line", "drums"]),
                                input_bytes=score_to_bytes(score), input_name=f"{name}.mid")
            rows.append({"melody": name, "engine": eng, "root_ok": root.root_pitch_class == pc,
                         "mode_ok": root.mode_family == fam, "root_confidence": round(root.root_confidence, 2),
                         "detected": f"{root.root_name} {root.mode}", "read_emotion": emo, "read_genre": genre,
                         "seconds": round(time.time() - t0, 2), **best(st)})
            print(f"  test1 {name:16} {eng:6} total={rows[-1]['total']:.3f} harm={rows[-1]['harmonic']:.3f}")
    summary = {eng: {"root_accuracy": mean([dict(r, x=float(r["root_ok"])) for r in rows if r["engine"] == eng], "x"),
                     "mode_accuracy": mean([dict(r, x=float(r["mode_ok"])) for r in rows if r["engine"] == eng], "x"),
                     **{k: mean([r for r in rows if r["engine"] == eng], k)
                        for k in ("total", "harmonic", "rhythmic", "key_consistency", "structure", "seconds")},
                     "valid_rate": mean([dict(r, x=float(r["valid"])) for r in rows if r["engine"] == eng], "x")}
               for eng in engines}
    return {"rows": rows, "summary": summary}


# ----------------------------------------------------------------------------- test 2

def test2a(prompts) -> dict:
    rows, per_field = [], {f: [0, 0] for f in PLAN_FIELDS}
    for prompt, expected in prompts:
        got = plan_from_text(prompt, GENRES, EMOTIONS)["settings"]
        ok = {}
        for f, v in expected.items():
            ok[f] = got.get(f) == v
            per_field[f][0] += ok[f]
            per_field[f][1] += 1
        rows.append({"prompt": prompt, "expected": expected, "got": {f: got.get(f) for f in expected},
                     "all_correct": all(ok.values()), "n_correct": sum(ok.values()), "n_expected": len(ok)})
    n_ok, n_all = sum(r["n_correct"] for r in rows), sum(r["n_expected"] for r in rows)
    return {"rows": rows, "field_accuracy": round(n_ok / n_all, 3),
            "prompt_exact_match": round(sum(r["all_correct"] for r in rows) / len(rows), 3),
            "per_field": {f: {"correct": c, "total": t, "accuracy": round(c / t, 3)} for f, (c, t) in per_field.items() if t}}


def _musical_traits(state) -> dict:
    c = state.candidates[0]
    notes = [n for t in c.score.tracks for n in t.notes]
    return {"tempo": state.plan.tempo, "velocity": float(np.mean([n.velocity for n in notes])),
            "notes_per_bar": len(notes) / state.plan.n_bars,
            "minor": 1.0 if state.plan.mode in ("minor", "dorian", "phrygian", "locrian", "harmonic_minor") else 0.0,
            "heard": list(((c.scores.get("classifier_top") or {}).get("emotion") or {}).keys())}


def test2b(genres=("pop", "rock"), seeds=(0, 1, 2)) -> dict:
    """Emotion sweep: only the emotion changes (key C, mode/tempo automatic), so any difference comes from it."""
    from midicomposer.styles import EMOTIONS as PROFILES

    def gen(genre, emotion, seed):
        return _musical_traits(run_generation(GenerationRequest(mode="metadata", genre=genre, emotion=emotion,
                                                                key="C", bars=16, n_variations=1, seed=seed)))
    rows, neutral = [], []
    for g in genres:
        for seed in seeds:
            neutral.append(gen(g, None, seed))
            for e in EMOTIONS:
                rows.append({"genre": g, "seed": seed, "emotion": e, "valence": PROFILES[e].valence,
                             "arousal": PROFILES[e].arousal, **gen(g, e, seed)})
                rows[-1]["hit"] = e in rows[-1]["heard"]
    per_emotion = {e: {"top3_hit_rate": round(float(np.mean([r["hit"] for r in rows if r["emotion"] == e])), 3),
                       "neutral_top3_hit_rate": round(float(np.mean([e in n["heard"] for n in neutral])), 3),
                       **{k: round(float(np.mean([r[k] for r in rows if r["emotion"] == e])), 2)
                          for k in ("tempo", "velocity", "notes_per_bar", "minor")}} for e in EMOTIONS}
    corr = lambda x, y: round(float(np.corrcoef([r[x] for r in rows], [r[y] for r in rows])[0, 1]), 3)
    print(f"  test2b {len(rows)} generations")
    return {"per_emotion": per_emotion, "n_generations": len(rows) + len(neutral),
            "top3_hit_rate": round(float(np.mean([r["hit"] for r in rows])), 3),
            "neutral_top3_hit_rate": round(float(np.mean([v["neutral_top3_hit_rate"] for v in per_emotion.values()])), 3),
            "chance_top3": round(3 / len(EMOTIONS), 3),
            "correlations": {"arousal~tempo": corr("arousal", "tempo"), "arousal~velocity": corr("arousal", "velocity"),
                             "arousal~notes_per_bar": corr("arousal", "notes_per_bar"),
                             "valence~minor": corr("valence", "minor")}}


def test2c(engines) -> dict:
    rows = []
    for name, chords, pc, mode, tempo, genre, emo in PROGRESSIONS:
        score = progression_to_score(chords, pc, mode, tempo)
        root = detect_root(score)
        for eng in engines:
            t0 = time.time()
            st = run_generation(GenerationRequest(mode="midi", input_role="chordal", engine=eng, n_variations=3,
                                                  seed=0, genre=genre, emotion=emo,
                                                  generated_roles=["lead_melody", "bass_line"]),
                                input_bytes=score_to_bytes(score), input_name=f"{name}.mid")
            rows.append({"progression": name, "engine": eng, "root_ok": root.root_pitch_class == pc,
                         "detected": f"{root.root_name} {root.mode}", "seconds": round(time.time() - t0, 2),
                         **best(st)})
            print(f"  test2c {name:24} {eng:6} harm={rows[-1]['harmonic']:.3f}")
    summary = {eng: {k: mean([r for r in rows if r["engine"] == eng], k)
                     for k in ("total", "harmonic", "rhythmic", "key_consistency", "seconds")} for eng in engines}
    for eng in engines:
        summary[eng]["root_accuracy"] = round(np.mean([r["root_ok"] for r in rows if r["engine"] == eng]), 3)
        summary[eng]["valid_rate"] = round(np.mean([r["valid"] for r in rows if r["engine"] == eng]), 3)
    return {"rows": rows, "summary": summary}


# ----------------------------------------------------------------------------- test 3

def test3(n_prompts=6) -> dict:
    from fastapi.testclient import TestClient
    from midicomposer.api.app import create_app

    def wait(c, job):
        for _ in range(1200):
            j = c.get(f"/generation/{job}").json()
            if j["status"] in ("done", "error"):
                return j
            time.sleep(0.05)
        raise TimeoutError(job)

    steps = ["plan", "generate", "critique", "build_on", "generate_more", "stems"]
    runs, critic_notes = [], []
    with TestClient(create_app()) as c:
        sid = c.post("/session").json()["session_id"]
        h = {"X-Session-Id": sid}
        for prompt, _ in PROMPTS[:n_prompts]:
            run, t = {"prompt": prompt}, time.time
            try:
                t0 = t(); s = c.post("/plan-from-text", json={"prompt": prompt}, headers=h).json()["settings"]
                run["plan"] = t() - t0
                body = {"mode": "metadata", "n_variations": 3, "engine": "rule",
                        **{k: s[k] for k in ("genre", "emotion", "key", "scale_mode", "tempo", "bars", "form",
                                             "instrumentation", "density") if s.get(k) is not None},
                        "generated_roles": s["generated_roles"]}
                t0 = t(); j = wait(c, c.post("/generate", json=body, headers=h).json()["job_id"]); run["generate"] = t() - t0
                assert j["status"] == "done", j.get("error")
                job = j["result"]["job_id"]
                t0 = t(); cr = c.get(f"/generation/{job}/critique", headers=h).json(); run["critique"] = t() - t0
                critic_notes += cr["notes"]
                t0 = t(); up = c.post(f"/generation/{job}/as-input", json={"candidate": 0}, headers=h).json()
                run["build_on"] = t() - t0
                t0 = t()
                j2 = wait(c, c.post("/generate", headers=h, json={"mode": "midi", "upload_id": up["upload_id"],
                                                                  "input_role": "full_mix", "generated_roles": ["pad"],
                                                                  "n_variations": 2}).json()["job_id"])
                run["generate_more"] = t() - t0
                assert j2["status"] == "done", j2.get("error")
                t0 = t(); z = c.get(f"/generation/{j2['result']['job_id']}/stems", headers=h); run["stems"] = t() - t0
                assert z.status_code == 200
                run["ok"] = True
            except Exception as e:                      # record the failure and keep going
                run["ok"], run["error"] = False, str(e)
            runs.append(run)
            print(f"  test3 {'ok ' if run['ok'] else 'ERR'} {prompt[:50]}")
    latency = {s: round(float(np.mean([r[s] for r in runs if s in r])), 3) for s in steps}
    kinds = Counter(n["kind"] for n in critic_notes)
    sev = Counter(n["severity"] for n in critic_notes)
    return {"runs": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in runs],
            "success_rate": round(sum(r["ok"] for r in runs) / len(runs), 3), "mean_latency_seconds": latency,
            "critic": {"notes_per_result": round(len(critic_notes) / max(1, len(runs)), 2), "by_kind": dict(kinds),
                       "by_severity": dict(sev),
                       "with_action": round(sum(n["action"] is not None for n in critic_notes) / max(1, len(critic_notes)), 3)}}


# ----------------------------------------------------------------------------- report

def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join("" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)) for v in r) + " |"
            for r in rows]
    return "\n".join(out)


def heldout_section(h) -> str:
    if not h:
        return ("\n**Held-out set: not run yet.** Ask a teammate who has not read `agents.py` to write 10-15 prompts "
                "with their intended settings into `evaluation/heldout_prompts.json` (format in "
                "`evaluation/README.md`), then rerun this script.\n")
    return (f"\n**Held-out set** (`evaluation/heldout_prompts.json`, {len(h['rows'])} prompts written by someone who "
            f"had not seen the planner): field accuracy {h['field_accuracy']:.0%}, prompts fully correct "
            f"{h['prompt_exact_match']:.0%}.\n\n"
            + table(["prompt", "expected", "got", "all correct"],
                    [[r["prompt"], json.dumps(r["expected"]), json.dumps(r["got"]), r["all_correct"]] for r in h["rows"]]))


def markdown(res: dict) -> str:
    t1, t2a, t2b, t2c, t3 = res["test1"], res["test2a"], res["test2b"], res["test2c"], res["test3"]
    md = [f"# MIDI Magic evaluation results\n\nGenerated {res['meta']['date']} on {res['meta']['machine']} by "
          "`python evaluation/run_evaluation.py`. Engines: " + ", ".join(res["meta"]["engines"]) + ". Seed 0, 3 "
          "candidates per request; metrics are for the top-ranked candidate (0-1, higher is better).\n",
          "## Test 1: melodic lines in isolation\n",
          "Six fixed melodies (`evaluation/test_sets.py`). The app detects key and emotion, then generates chords, "
          "bass and drums around the locked melody.\n",
          table(["engine", "key root acc.", "mode acc.", "harmonic", "rhythmic", "key consist.", "structure",
                 "total", "valid", "s/request"],
                [[e, s["root_accuracy"], s["mode_accuracy"], s["harmonic"], s["rhythmic"], s["key_consistency"],
                  s["structure"], s["total"], s["valid_rate"], s["seconds"]] for e, s in t1["summary"].items()]),
          "\n", table(["melody", "engine", "detected key", "root ok", "mode ok", "read emotion", "harmonic", "total"],
                      [[r["melody"], r["engine"], r["detected"], r["root_ok"], r["mode_ok"], r["read_emotion"],
                        r["harmonic"], r["total"]] for r in t1["rows"]]),
          "\n## Test 2a: text planner accuracy\n",
          f"**Development set** (12 prompts, `PROMPTS` in `test_sets.py`): field accuracy "
          f"{t2a['field_accuracy']:.0%}, prompts fully correct {t2a['prompt_exact_match']:.0%} (planner: "
          f"`{res['meta']['text_planner']}`). Caution: these prompts were written by the same person who wrote the "
          "planner's keyword lists, so this number is optimistic. The held-out set below is the fair measure.\n",
          table(["field", "correct", "total", "accuracy"],
                [[f, v["correct"], v["total"], v["accuracy"]] for f, v in t2a["per_field"].items()]),
          "\n", table(["prompt", "expected", "got", "all correct"],
                      [[r["prompt"], json.dumps(r["expected"]), json.dumps(r["got"]), r["all_correct"]]
                       for r in t2a["rows"]]),
          heldout_section(res.get("test2a_heldout")),
          "\n## Test 2b: does the requested emotion reach the music?\n",
          f"Emotion sweep: every emotion × 2 genres (pop, rock) × 3 seeds, key C, everything else automatic, "
          f"plus the same requests with no emotion as a baseline ({t2b['n_generations']} generations). (1) The "
          "trained emotion classifier (48% test accuracy over 11 emotions) listens to each result; a hit means the "
          "requested emotion is in its top 3. (2) Correlation between each emotion's arousal/valence and measurable "
          "musical traits.\n",
          table(["condition", "top-3 hit rate"], [["emotion requested", t2b["top3_hit_rate"]],
                                                  ["no emotion (baseline)", t2b["neutral_top3_hit_rate"]],
                                                  ["chance", t2b["chance_top3"]]]),
          "\n", table(["relationship", "Pearson r"], [[k, v] for k, v in t2b["correlations"].items()]),
          "\n", table(["emotion", "top-3 hit", "baseline hit", "tempo", "velocity", "notes/bar", "minor share"],
                      [[e, v["top3_hit_rate"], v["neutral_top3_hit_rate"], v["tempo"], v["velocity"],
                        v["notes_per_bar"], v["minor"]] for e, v in t2b["per_emotion"].items()]),
          "\n## Test 2c: chord progressions in isolation\n",
          "Four fixed progressions as the locked input; the app generates a lead melody and bass line over them.\n",
          table(["engine", "key root acc.", "harmonic", "rhythmic", "key consist.", "total", "valid", "s/request"],
                [[e, s["root_accuracy"], s["harmonic"], s["rhythmic"], s["key_consistency"], s["total"],
                  s["valid_rate"], s["seconds"]] for e, s in t2c["summary"].items()]),
          "\n", table(["progression", "engine", "detected key", "root ok", "harmonic", "total"],
                      [[r["progression"], r["engine"], r["detected"], r["root_ok"], r["harmonic"], r["total"]]
                       for r in t2c["rows"]]),
          "\n## Test 3: whole pipeline through the API\n",
          f"Prompt → plan → generate → critic → build on this (+ pad) → stems, for {len(t3['runs'])} prompts. "
          f"**Success rate {t3['success_rate']:.0%}.**\n",
          table(["step", "mean seconds"], [[k, v] for k, v in t3["mean_latency_seconds"].items()]),
          f"\nCritic: {t3['critic']['notes_per_result']} notes per result; by kind {t3['critic']['by_kind']}; "
          f"by severity {t3['critic']['by_severity']}; {t3['critic']['with_action']:.0%} carry a one-click fix.\n",
          "The user-study half of test 3 (people using the GUI) is in `evaluation/user_study.md`.\n"]
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="rule,hybrid")
    ap.add_argument("--out", default=str(ROOT / "evaluation" / "results"))
    a = ap.parse_args()
    engines = a.engines.split(",")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = ROOT / "evaluation" / "data"                  # the test inputs as MIDI files (user study, DAW checks)
    for sub in ("melodies", "progressions"):
        (data / sub).mkdir(parents=True, exist_ok=True)
    for name, notes, meter, tempo, *_ in MELODIES:
        (data / "melodies" / f"{name}.mid").write_bytes(score_to_bytes(line_to_score(notes, meter, tempo)))
    for name, chords, pc, mode, tempo, *_ in PROGRESSIONS:
        (data / "progressions" / f"{name}.mid").write_bytes(score_to_bytes(progression_to_score(chords, pc, mode, tempo)))
    print("Test 1"); r1 = test1(engines)
    print("Test 2a"); r2a = test2a(PROMPTS)
    heldout = ROOT / "evaluation" / "heldout_prompts.json"
    r2a_held = None
    if heldout.exists():
        r2a_held = test2a([(x["prompt"], x["expected"]) for x in json.loads(heldout.read_text(encoding="utf-8"))])
    print("Test 2b"); r2b = test2b()
    print("Test 2c"); r2c = test2c(engines)
    print("Test 3"); r3 = test3()
    res = {"meta": {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "engines": engines,
                    "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
                    "text_planner": "keywords", "critic": "rules"},
           "test1": r1, "test2a": r2a, "test2a_heldout": r2a_held, "test2b": r2b, "test2c": r2c, "test3": r3}
    (out / "results.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    (out / "results.md").write_text(markdown(res), encoding="utf-8")
    print(f"wrote {out / 'results.md'}")


if __name__ == "__main__":
    main()
