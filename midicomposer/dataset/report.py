"""Dataset statistics report (JSON + Markdown)."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _quantiles(vals: list[float]) -> dict:
    if not vals:
        return {}
    a = np.asarray(vals, dtype=float)
    return {k: round(float(np.percentile(a, q)), 3) for k, q in
            (("min", 0), ("p05", 5), ("p25", 25), ("median", 50), ("p75", 75), ("p95", 95), ("max", 100))}


def build_report(records: list[dict], label_vocab: dict, meta: dict) -> dict:
    ok = [r for r in records if r.get("status") == "ok"]
    usable = [r for r in records if r.get("usable_for_training")]
    status = Counter(r.get("status") for r in records)
    splits = Counter(r.get("split") for r in usable)
    split_by_genre = defaultdict(Counter)
    split_by_emotion = defaultdict(Counter)
    cross = defaultdict(Counter)
    for r in usable:
        split_by_genre[r.get("genre")][r["split"]] += 1
        split_by_emotion[r.get("emotion")][r["split"]] += 1
    for r in records:
        if r.get("emotion") and r.get("genre"):
            cross[r["emotion"]][r["genre"]] += 1
    tempos = [r["tempo_bpm"] for r in ok]
    tempo_counter = Counter(round(t) for t in tempos)
    instruments = Counter(t["instrument"] for r in ok for t in r.get("tracks", []))
    track_names = Counter(t["name"].lower() for r in ok for t in r.get("tracks", []))
    roles = Counter(t["role"] for r in ok for t in r.get("tracks", []))
    confs = [r["root_confidence"] for r in ok if r.get("root_confidence") is not None]
    report = {
        "meta": meta,
        "summary": {
            "files_discovered": meta.get("n_discovered"),
            "files_processed": len(records),
            "status_counts": dict(status),
            "usable_for_training": len(usable),
            "exact_or_musical_duplicates": sum(1 for r in records if r.get("is_duplicate")),
            "duplicate_kinds": dict(Counter(r.get("duplicate_kind") for r in records if r.get("is_duplicate"))),
            "label_parse_status_all_files": label_vocab.get("parse_status"),
            "n_emotion_classes": len(label_vocab["emotions"]),
            "n_genre_classes": len(label_vocab["genres"]),
        },
        "labels_all_discovered_files": {
            "emotions": {e["label"]: e["count"] for e in label_vocab["emotions"]},
            "genres": {g["label"]: g["count"] for g in label_vocab["genres"]},
            "raw_spellings": {"emotions": {e["label"]: e["raw_spellings"] for e in label_vocab["emotions"]},
                              "genres": {g["label"]: g["raw_spellings"] for g in label_vocab["genres"]}},
        },
        "labels_processed": {
            "emotions": dict(Counter(r.get("emotion") for r in records).most_common()),
            "genres": dict(Counter(r.get("genre") for r in records).most_common()),
            "emotion_x_genre": {e: dict(c) for e, c in cross.items()},
        },
        "splits": {"usable_by_split": dict(splits), "by_genre": {g: dict(c) for g, c in split_by_genre.items()},
                   "by_emotion": {e: dict(c) for e, c in split_by_emotion.items()}},
        "root": {
            "root_distribution": dict(Counter(r.get("root_name") for r in ok).most_common()),
            "mode_distribution": dict(Counter(r.get("mode") for r in ok).most_common()),
            "confidence_quantiles": _quantiles(confs),
            "ambiguous": sum(1 for r in ok if r.get("root_ambiguous")),
            "undetermined": sum(1 for r in ok if r.get("root_pitch_class") is None),
            "methods": dict(Counter(r.get("root_detection_method") for r in ok)),
            "files_with_key_signature": sum(1 for r in ok if r.get("has_key_signature")),
        },
        "timing": {
            "tempo_quantiles": _quantiles(tempos),
            "most_common_tempos": dict(tempo_counter.most_common(10)),
            "share_default_119_120_bpm": round(sum(1 for t in tempos if 118.5 <= t <= 120.5) / max(1, len(tempos)), 4),
            "meters": dict(Counter(r.get("meter") for r in ok)),
            "files_with_tempo_changes": sum(1 for r in ok if r.get("n_tempo_changes", 0) > 0),
            "duration_seconds_quantiles": _quantiles([r["duration_seconds"] for r in ok]),
            "bars_quantiles": _quantiles([r["n_bars"] for r in ok]),
        },
        "tracks": {
            "tracks_per_file_quantiles": _quantiles([r["n_tracks"] for r in ok]),
            "notes_per_file_quantiles": _quantiles([r["n_notes"] for r in ok]),
            "instrument_counts": dict(instruments.most_common()),
            "track_name_counts": dict(track_names.most_common(40)),
            "detected_role_counts": dict(roles.most_common()),
            "files_with_drums": sum(1 for r in ok if any(t["is_drum"] for t in r.get("tracks", []))),
        },
        "failures": [{"path": r["path"], "status": r.get("status"), "error": r.get("error")}
                     for r in records if r.get("status") != "ok"][:200],
    }
    return report


def write_report(out: Path, records: list[dict], label_vocab: dict, meta: dict) -> dict:
    rep = build_report(records, label_vocab, meta)
    (out / "dataset_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    (out / "dataset_report.md").write_text(render_markdown(rep), encoding="utf-8")
    return rep


def _table(d: dict, k1: str = "label", k2: str = "count") -> str:
    lines = [f"| {k1} | {k2} |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in d.items()]
    return "\n".join(lines)


def render_markdown(rep: dict) -> str:
    s = rep["summary"]
    m = rep["meta"]
    lab = rep["labels_all_discovered_files"]
    out = [
        "# XMIDI dataset report", "",
        f"- Source: `{m.get('xmidi_root')}`",
        f"- Files discovered: **{s['files_discovered']}**; processed: **{s['files_processed']}**"
        + (f" (dev subset, `--max-files {m.get('max_files')}`)" if m.get("max_files") else " (all files)"),
        f"- Usable for training (parsed, labelled, non-duplicate, root determined): **{s['usable_for_training']}**",
        f"- Status counts: {s['status_counts']}",
        f"- Duplicates removed from training: {s['exact_or_musical_duplicates']} {s['duplicate_kinds']}",
        f"- Filename parse status (all files): {s['label_parse_status_all_files']}",
        f"- Processing time: {m.get('seconds')} s", "",
        f"## Emotion classes ({s['n_emotion_classes']}, all discovered files)", "", _table(lab["emotions"]), "",
        f"## Genre classes ({s['n_genre_classes']}, all discovered files)", "", _table(lab["genres"]), "",
        "## Emotion x genre (processed files)", "",
    ]
    cross = rep["labels_processed"]["emotion_x_genre"]
    genres = sorted({g for c in cross.values() for g in c})
    out.append("| emotion | " + " | ".join(genres) + " |")
    out.append("|---|" + "---:|" * len(genres))
    for e in sorted(cross):
        out.append(f"| {e} | " + " | ".join(str(cross[e].get(g, 0)) for g in genres) + " |")
    out += ["", "## Splits (usable files)", "", _table(rep["splits"]["usable_by_split"], "split"), "",
            "## Root / key", "",
            f"- Root confidence quantiles: {rep['root']['confidence_quantiles']}",
            f"- Ambiguous roots: {rep['root']['ambiguous']}; undetermined: {rep['root']['undetermined']}",
            f"- Files with a key-signature event: {rep['root']['files_with_key_signature']}",
            f"- Detection methods: {rep['root']['methods']}", "",
            _table(rep["root"]["root_distribution"], "root"), "", _table(rep["root"]["mode_distribution"], "mode"), "",
            "## Timing", "",
            f"- Tempo quantiles: {rep['timing']['tempo_quantiles']}",
            f"- Share of files at the default ~119-120 BPM: {rep['timing']['share_default_119_120_bpm']}"
            " (tempo is therefore a weak signal in XMIDI)",
            f"- Meters: {rep['timing']['meters']}",
            f"- Duration (s) quantiles: {rep['timing']['duration_seconds_quantiles']}",
            f"- Bars quantiles: {rep['timing']['bars_quantiles']}", "",
            "## Tracks", "",
            f"- Tracks per file: {rep['tracks']['tracks_per_file_quantiles']}",
            f"- Files with drums: {rep['tracks']['files_with_drums']}", "",
            _table(rep["tracks"]["detected_role_counts"], "detected role"), "",
            _table(rep["tracks"]["instrument_counts"], "instrument"), ""]
    if rep["failures"]:
        out += ["## Failures (first 200)", "", "| path | status | error |", "|---|---|---|"]
        out += [f"| {f['path']} | {f['status']} | {str(f['error'])[:120]} |" for f in rep["failures"]]
    return "\n".join(out) + "\n"
