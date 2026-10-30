"""XMIDI dataset preparation: discovery, label parsing, hashing, analysis, dedup, splits, manifest.

    python -m midicomposer.cli prepare --xmidi-root PATH --out data/xmidi [--max-files N]

Outputs (in --out):
    manifest.jsonl          one JSON record per discovered file (incl. failures)
    manifest.csv            flat view of the most useful columns
    features.npy            classifier features for usable files (rows align with features_index.json)
    label_vocab.json        discovered emotion/genre classes with counts and raw spellings
    dataset_report.json/.md statistics report
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..features import score_features
from ..midi_io import MidiLoadError, load_midi
from ..roles import detect_roles
from ..root import detect_root
from ..theory import gm_name
from .filenames import LabelVocab, build_vocab, discover_midi_files, display_label, parse_filename
from .report import write_report

ENV_ROOT = "XMIDI_ROOT"


@dataclass
class PrepareConfig:
    xmidi_root: str
    out_dir: str = "data/xmidi"
    max_files: int | None = None
    workers: int = max(1, (os.cpu_count() or 2) - 2)
    split_salt: str = "xmidi-v1"
    train_frac: float = 0.8
    val_frac: float = 0.1
    min_root_confidence: float = 0.4
    root_overrides: str | None = None      # JSON/CSV: filename or id -> key ("D minor")
    label_hints: dict | None = None
    chunksize: int = 32


def resolve_xmidi_root(arg: str | None) -> Path:
    root = arg or os.environ.get(ENV_ROOT)
    if not root:
        raise SystemExit(f"No XMIDI root given. Pass --xmidi-root or set the {ENV_ROOT} environment variable.")
    p = Path(root)
    if not p.is_dir():
        raise SystemExit(f"XMIDI root {p} is not a directory.")
    return p


def load_root_overrides(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if p.suffix.lower() == ".json":
        return {str(k): str(v) for k, v in json.loads(p.read_text()).items()}
    out = {}
    with p.open(newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["file"].strip()] = row["key"].strip()
    return out


def _hash_notes(items: list[tuple]) -> str:
    h = hashlib.sha1()
    for it in sorted(items):
        h.update(repr(it).encode())
    return h.hexdigest()


def process_file(args: tuple[str, str, str | None]) -> dict:
    """Worker: hash + parse + analyse one file. Never raises; errors are recorded."""
    path, rel, override = args
    rec: dict = {"path": rel}
    try:
        data = Path(path).read_bytes()
    except OSError as e:
        return {**rec, "status": "unreadable", "error": str(e)}
    rec["size_bytes"] = len(data)
    rec["content_sha256"] = hashlib.sha256(data).hexdigest()
    if not data:
        return {**rec, "status": "empty_file", "error": "zero-byte file"}
    try:
        score = load_midi(data)
    except MidiLoadError as e:
        return {**rec, "status": "corrupted", "error": str(e)[:300]}
    except Exception as e:  # pragma: no cover - defensive
        return {**rec, "status": "corrupted", "error": f"{type(e).__name__}: {e}"[:300]}
    notes_all = [n for t in score.tracks for n in t.notes]
    if not notes_all:
        return {**rec, "status": "no_notes", "error": "no note events"}
    pitched = [(n, t) for t in score.tracks if not t.is_drum for n in t.notes]
    min_p = min((n.pitch for n, _ in pitched), default=0)
    music_items, trans_items = [], []
    for t in score.tracks:
        for n in t.notes:
            key = (round(n.start * 12), round(n.duration * 12), t.is_drum)
            music_items.append(key + (n.pitch,))
            trans_items.append(key + ((n.pitch if t.is_drum else n.pitch - min_p),))
    rec["music_hash"] = _hash_notes(music_items)
    rec["transposition_hash"] = _hash_notes(trans_items)

    root = detect_root(score, override=override)
    grid = score.bar_grid()
    roles = detect_roles(score)
    rec.update({
        "status": "ok",
        "load_warnings": score.load_warnings,
        "duration_seconds": round(score.duration_seconds, 3),
        "total_beats": round(score.end_beat, 3),
        "n_bars": grid.n_bars,
        "tempo_bpm": round(score.initial_bpm, 3),
        "n_tempo_changes": len(score.tempos) - 1,
        "meter": score.initial_time_signature.label,
        "n_meter_changes": len(score.time_signatures) - 1,
        "ticks_per_beat": score.ticks_per_beat,
        "has_key_signature": bool(score.key_signatures),
        "root_name": root.root_name,
        "root_pitch_class": root.root_pitch_class,
        "root_reference_midi": root.root_reference_midi,
        "mode": root.mode,
        "mode_family": root.mode_family,
        "root_confidence": root.root_confidence,
        "root_detection_method": root.root_detection_method,
        "root_ambiguous": root.ambiguous,
        "root_requires_override": root.requires_override,
        "root_top_candidates": root.top_candidates if root.ambiguous else root.top_candidates[:2],
        "root_override_applied": override,
        "n_tracks": len(score.tracks),
        "n_notes": len(notes_all),
        "tracks": [{
            "track_id": t.track_id, "name": t.name, "program": t.program,
            "instrument": gm_name(t.program, t.is_drum), "channel": t.channel, "is_drum": t.is_drum,
            "n_notes": len(t.notes), "role": roles[t.track_id]["role"],
            "pitch_min": min((n.pitch for n in t.notes), default=None),
            "pitch_max": max((n.pitch for n in t.notes), default=None),
        } for t in score.tracks],
    })
    rec["_features"] = score_features(score, root).tolist()
    return rec


class _UnionFind:
    def __init__(self):
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def assign_split(group_key: str, salt: str, train_frac: float, val_frac: float) -> str:
    u = int(hashlib.sha256(f"{salt}:{group_key}".encode()).hexdigest()[:15], 16) / float(16 ** 15)
    if u < train_frac:
        return "train"
    if u < train_frac + val_frac:
        return "val"
    return "test"


def select_files(files: list[Path], root: Path, max_files: int | None) -> list[Path]:
    """Deterministic subset for dev mode: order by a hash of the relative path (spreads across labels)."""
    if not max_files or max_files >= len(files):
        return files
    keyed = sorted(files, key=lambda p: hashlib.sha1(p.relative_to(root).as_posix().encode()).hexdigest())
    return sorted(keyed[:max_files], key=lambda p: p.relative_to(root).as_posix())


def prepare_dataset(cfg: PrepareConfig, progress=None) -> dict:
    t0 = time.time()
    root = resolve_xmidi_root(cfg.xmidi_root)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_files = discover_midi_files(root)
    names = [p.name for p in all_files]

    # Labels are discovered from *all* filenames (cheap), even in --max-files mode.
    vocab: LabelVocab = build_vocab(names, cfg.label_hints)
    parsed_all = {p: parse_filename(p.name, vocab) for p in all_files}
    files = select_files(all_files, root, cfg.max_files)

    overrides = load_root_overrides(cfg.root_overrides)
    jobs = []
    for p in files:
        pn = parsed_all[p]
        ov = overrides.get(p.name) or (overrides.get(pn.file_id) if pn.file_id else None)
        jobs.append((str(p), p.relative_to(root).as_posix(), ov))

    records: list[dict] = []
    if cfg.workers <= 1 or len(jobs) < 64:
        for i, j in enumerate(jobs):
            records.append(process_file(j))
            if progress and i % 200 == 0:
                progress(i, len(jobs))
    else:
        with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
            for i, r in enumerate(ex.map(process_file, jobs, chunksize=cfg.chunksize)):
                records.append(r)
                if progress and i % 1000 == 0:
                    progress(i, len(jobs))

    # Attach labels.
    by_rel = {p.relative_to(root).as_posix(): parsed_all[p] for p in files}
    for r in records:
        pn = by_rel[r["path"]]
        r.update({"filename": pn.filename, "file_id": pn.file_id, "raw_emotion": pn.raw_emotion,
                  "raw_genre": pn.raw_genre, "emotion": pn.emotion, "genre": pn.genre,
                  "label_status": pn.status, "id_length_ok": pn.id_length_ok, "label_notes": pn.notes})

    # Duplicate groups: exact bytes, identical note content, transposition-invariant content.
    uf = _UnionFind()
    for key in ("content_sha256", "music_hash", "transposition_hash"):
        first: dict[str, int] = {}
        for i, r in enumerate(records):
            h = r.get(key)
            if h is None:
                continue
            if h in first:
                uf.union(first[h], i)
            else:
                first[h] = i
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(records)):
        groups[uf.find(i)].append(i)
    for members in groups.values():
        members.sort(key=lambda i: records[i]["path"])
        canon = records[members[0]]
        gkey = canon.get("content_sha256") or canon["path"]
        split = assign_split(gkey, cfg.split_salt, cfg.train_frac, cfg.val_frac)
        for rank, i in enumerate(members):
            r = records[i]
            r["dup_group"] = gkey[:16]
            r["dup_group_size"] = len(members)
            r["is_duplicate"] = rank > 0
            r["duplicate_of"] = canon["path"] if rank > 0 else None
            if rank > 0:
                kinds = [k for k in ("content_sha256", "music_hash", "transposition_hash")
                         if r.get(k) and r.get(k) == canon.get(k)]
                r["duplicate_kind"] = kinds[0] if kinds else "transitive"
            r["split"] = split
    for r in records:
        r["usable_for_training"] = bool(
            r.get("status") == "ok" and not r.get("is_duplicate") and r.get("root_pitch_class") is not None
            and (r.get("root_confidence", 0) >= cfg.min_root_confidence or r.get("root_override_applied"))
            and r.get("label_status") in ("ok", "resolved_by_vocabulary"))

    # Write manifest + features.
    feats, feat_index = [], []
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            f = r.pop("_features", None)
            if f is not None and r["usable_for_training"]:
                feats.append(f)
                feat_index.append(r["path"])
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    np.save(out / "features.npy", np.asarray(feats, dtype=np.float32).reshape(len(feats), -1))
    (out / "features_index.json").write_text(json.dumps(feat_index))
    _write_csv(out / "manifest.csv", records)

    label_vocab = _label_vocab(parsed_all.values())
    (out / "label_vocab.json").write_text(json.dumps(label_vocab, indent=2, ensure_ascii=False))
    meta = {"xmidi_root": str(root), "n_discovered": len(all_files), "n_processed": len(records),
            "max_files": cfg.max_files, "seconds": round(time.time() - t0, 1), "config": {
                k: v for k, v in cfg.__dict__.items() if k not in ("label_hints",)}}
    report = write_report(out, records, label_vocab, meta)
    return {"out_dir": str(out), "n_discovered": len(all_files), "n_processed": len(records),
            "n_usable": sum(r["usable_for_training"] for r in records), "seconds": meta["seconds"],
            "report": report["summary"]}


def _label_vocab(parsed) -> dict:
    emo, gen, raw_e, raw_g, status = Counter(), Counter(), defaultdict(Counter), defaultdict(Counter), Counter()
    for pn in parsed:
        status[pn.status] += 1
        if pn.emotion:
            emo[pn.emotion] += 1
            raw_e[pn.emotion][pn.raw_emotion] += 1
        if pn.genre:
            gen[pn.genre] += 1
            raw_g[pn.genre][pn.raw_genre] += 1
    return {
        "emotions": [{"label": k, "display": display_label(k), "count": c, "raw_spellings": dict(raw_e[k])}
                     for k, c in emo.most_common()],
        "genres": [{"label": k, "display": display_label(k), "count": c, "raw_spellings": dict(raw_g[k])}
                   for k, c in gen.most_common()],
        "parse_status": dict(status),
        "note": "Labels are discovered dynamically from filenames over all discovered files.",
    }


CSV_COLUMNS = ["path", "file_id", "raw_emotion", "raw_genre", "emotion", "genre", "label_status", "status",
               "error", "split", "usable_for_training", "is_duplicate", "duplicate_of", "dup_group",
               "content_sha256", "size_bytes", "duration_seconds", "n_bars", "tempo_bpm", "meter",
               "root_name", "root_pitch_class", "root_reference_midi", "mode", "root_confidence",
               "root_detection_method", "root_ambiguous", "n_tracks", "n_notes", "instruments", "roles"]


def _write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            row = dict(r)
            tr = r.get("tracks") or []
            row["instruments"] = "|".join(t["instrument"] for t in tr)
            row["roles"] = "|".join(t["role"] for t in tr)
            w.writerow(row)


def load_manifest(data_dir: str | Path) -> list[dict]:
    with (Path(data_dir) / "manifest.jsonl").open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
