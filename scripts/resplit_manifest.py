"""Re-split a prepared XMIDI manifest with new train/val/test fractions, without re-parsing any MIDI.

    python scripts/resplit_manifest.py --src data/xmidi --out data/xmidi_60_40 --train 0.57 --val 0.03

Splits are assigned per duplicate group exactly as `prepare` does (salted hash of the group's canonical
file, `assign_split`), so duplicates and transposed copies still never cross splits. Before writing,
the script checks that the source's own fractions reproduce its existing split for every record.
Each output record keeps its previous split as `split_original`. The source folder is not modified.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from midicomposer.dataset.prepare import assign_split  # noqa: E402


def group_key(rec: dict, sha_by_path: dict[str, str | None]) -> str:
    canon = rec.get("duplicate_of") or rec["path"]
    return sha_by_path.get(canon) or canon


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/xmidi")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=float, required=True, help="train fraction (e.g. 0.57)")
    ap.add_argument("--val", type=float, required=True, help="validation fraction (e.g. 0.03); test = the rest")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    if src.resolve() == out.resolve():
        raise SystemExit("--out must differ from --src (the source manifest is never modified)")
    if not (0 < args.train and args.val >= 0 and args.train + args.val < 1):
        raise SystemExit("need 0 < train, 0 <= val, train + val < 1")
    cfg = json.loads((src / "dataset_report.json").read_text(encoding="utf-8"))["meta"]["config"]
    salt, old_train, old_val = cfg["split_salt"], cfg["train_frac"], cfg["val_frac"]

    sha_by_path: dict[str, str | None] = {}
    with (src / "manifest.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            sha_by_path[r["path"]] = r.get("content_sha256")

    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, Counter] = defaultdict(Counter)
    moved = Counter()
    by_genre: dict[str, Counter] = defaultdict(Counter)
    by_emotion: dict[str, Counter] = defaultdict(Counter)
    mismatches = 0
    tmp = out / "manifest.jsonl.tmp"
    with (src / "manifest.jsonl").open(encoding="utf-8") as fin, tmp.open("w", encoding="utf-8") as fout:
        for line in fin:
            r = json.loads(line)
            gkey = group_key(r, sha_by_path)
            if (assign_split(gkey, salt, old_train, old_val) != r["split"]
                    or (r.get("dup_group") and gkey[:16] != r["dup_group"])):
                mismatches += 1
            new = assign_split(gkey, salt, args.train, args.val)
            r["split_original"] = r["split"]
            r["split"] = new
            usable = bool(r.get("usable_for_training"))
            counts["all"][new] += 1
            if usable:
                counts["usable"][new] += 1
                moved[f"{r['split_original']}->{new}"] += 1
                by_genre[r.get("genre")][new] += 1
                by_emotion[r.get("emotion")][new] += 1
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    if mismatches:
        tmp.unlink()
        raise SystemExit(f"{mismatches} records do not reproduce their original split; refusing to write")
    tmp.replace(out / "manifest.jsonl")
    shutil.copy2(src / "label_vocab.json", out / "label_vocab.json")
    n_usable = sum(counts["usable"].values())
    summary = {
        "source": str(src), "salt": salt, "fractions": {"train": args.train, "val": args.val,
                                                        "test": round(1 - args.train - args.val, 6)},
        "note": ("val is carved out of the training portion and is used only for checkpoint selection; "
                 "test is never read during training"),
        "usable_by_split": dict(counts["usable"]),
        "usable_share": {k: round(v / n_usable, 4) for k, v in counts["usable"].items()},
        "all_records_by_split": dict(counts["all"]),
        "usable_moves_from_original_split": dict(moved),
        "usable_by_genre": {k: dict(v) for k, v in sorted(by_genre.items(), key=lambda kv: str(kv[0]))},
        "usable_by_emotion": {k: dict(v) for k, v in sorted(by_emotion.items(), key=lambda kv: str(kv[0]))},
        "original_split_reproduced_for_all_records": True,
    }
    (out / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("fractions", "usable_by_split", "usable_share",
                                              "usable_moves_from_original_split")}, indent=1))


if __name__ == "__main__":
    main()
