"""Held-out test loss of composer checkpoints on the 60/40 test split.

    python scripts/eval_test_loss.py --checkpoint checkpoints/composer_60_40.pt checkpoints/composer_dev.pt

Every checkpoint is scored on the same deterministic examples (masked-track reconstruction, section
infilling and continuation, built exactly like training examples) from test-split files: masked
cross-entropy per target token (nats) and its perplexity. Two subsets:

  all            usable files of the 60/40 test split (deterministic sample of --max-files)
  original_test  files that were already in the original 80/10/10 test split. Neither composer_dev
                 (trained on original train, selected on original val) nor a 60/40 model has trained on
                 or been selected with them, so this is the fair subset for comparing the two.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from midicomposer.model.data import MaskedTrackDataset, _tokenize_one, collate  # noqa: E402
from midicomposer.model.hierarchical import HierarchicalComposer, ModelConfig  # noqa: E402
from midicomposer.model.train import evaluate  # noqa: E402
from midicomposer.model.vocab import Vocab  # noqa: E402

KEEP = ("path", "genre", "emotion", "root_name", "root_pitch_class", "mode", "root_confidence", "tempo_bpm", "meter",
        "tracks", "n_bars")


def test_records(data_dir: Path) -> dict[str, list[dict]]:
    subsets: dict[str, list[dict]] = {"all": [], "original_test": []}
    with (data_dir / "manifest.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if '"split": "test"' not in line or '"usable_for_training": true' not in line:
                continue
            r = json.loads(line)
            rec = {k: r[k] for k in KEEP}
            subsets["all"].append(rec)
            if r.get("split_original") == "test":
                subsets["original_test"].append(rec)
    return subsets


def sample(recs: list[dict], n: int | None, seed: int) -> list[dict]:
    recs = sorted(recs, key=lambda r: r["path"])
    if n and len(recs) > n:
        recs = sorted(random.Random(seed).sample(recs, n), key=lambda r: r["path"])
    return recs


def token_cache(recs, vocab: Vocab, xmidi_root: Path, max_bars_per_file: int, workers: int, path: Path) -> dict:
    paths = [r["path"] for r in recs]
    if path.exists():
        with path.open("rb") as fh:
            cache = pickle.load(fh)
        if cache.get("paths") == paths and cache.get("vocab") == vocab.to_dict():
            return cache
    jobs = [(str(xmidi_root / r["path"]), r, vocab.genres, vocab.emotions, max_bars_per_file) for r in recs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        items = [x for x in ex.map(_tokenize_one, jobs, chunksize=16) if x is not None]
    cache = {"paths": paths, "vocab": vocab.to_dict(), "n_files": len(items), "items": items}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return cache


def load_checkpoint(path: Path, dev: torch.device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    vocab = Vocab(ck["genres"], ck["emotions"])
    if vocab.to_dict() != ck["info"]["vocab"]:
        raise SystemExit(f"{path}: vocabulary differs from the current code; cannot score it fairly")
    model = HierarchicalComposer(ModelConfig(**ck["model_config"]))
    model.load_state_dict(ck["state_dict"])
    return model.to(dev).eval(), vocab, ck


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", nargs="+", required=True)
    ap.add_argument("--data", default="data/xmidi_60_40")
    ap.add_argument("--xmidi-root", default="XMIDI_Dataset")
    ap.add_argument("--cache-dir", default="data/token_cache_60_40")
    ap.add_argument("--max-files", type=int, default=6000, help="test files sampled per subset (0 = all)")
    ap.add_argument("--batches", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="reports/test_eval_60_40.json")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    models = {p: load_checkpoint(Path(p), dev) for p in args.checkpoint}
    first_vocab = next(iter(models.values()))[1]
    for p, (_, v, _) in models.items():
        if v.to_dict() != first_vocab.to_dict():
            raise SystemExit(f"{p} uses a different vocabulary; score it in a separate run")
    tcfgs = {p: ck["train_config"] for p, (_, _, ck) in models.items()}
    window = {tc["window_bars"] for tc in tcfgs.values()}
    max_len = {tc["max_len"] for tc in tcfgs.values()}
    max_bars = {ck["model_config"]["max_bars"] for _, _, ck in models.values()}
    mbpf = {tc.get("max_bars_per_file", 96) for tc in tcfgs.values()}
    if len(window) > 1 or len(max_len) > 1 or len(max_bars) > 1 or len(mbpf) > 1:
        raise SystemExit(f"checkpoints use different example settings {window, max_len, max_bars, mbpf}")
    window, max_len, max_bars, mbpf = window.pop(), max_len.pop(), max_bars.pop(), mbpf.pop()

    subsets = test_records(Path(args.data))
    if not subsets["all"]:
        raise SystemExit(f"no usable test records in {args.data}/manifest.jsonl")
    report = {"data_dir": args.data, "seed": args.seed, "batches": args.batches, "batch_size": args.batch_size,
              "example_settings": {"window_bars": window, "max_len": max_len, "max_bars": max_bars},
              "test_files_in_split": {k: len(v) for k, v in subsets.items()}, "results": {}}
    for name, recs in subsets.items():
        recs = sample(recs, args.max_files or None, args.seed)
        cache = token_cache(recs, first_vocab, Path(args.xmidi_root), mbpf, args.workers,
                            Path(args.cache_dir) / f"test_{name}_{len(recs)}_seed{args.seed}.pkl")
        ds = MaskedTrackDataset(cache, first_vocab, window, max_len, max_bars, args.batches * args.batch_size, args.seed)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=lambda b: collate(b, first_vocab.pad, max_bars), num_workers=0)
        batches = list(dl)  # identical examples for every checkpoint
        n_tok = int(sum(b["loss"][:, 1:].sum().item() for b in batches))
        report["results"][name] = {"files": cache["n_files"], "examples": args.batches * args.batch_size,
                                   "target_tokens": n_tok, "checkpoints": {}}
        for p, (model, _, ck) in models.items():
            loss = evaluate(model, batches, dev, dev.type == "cuda")
            info = ck["info"]
            report["results"][name]["checkpoints"][p] = {
                "test_loss": round(loss, 4), "perplexity": round(float(torch.tensor(loss).exp()), 3),
                "best_val_loss": info.get("best_val_loss"), "step": info.get("step"),
                "train_files": info.get("train_files"), "data_dir": info.get("data_dir")}
            print(f"[{name:13s}] {p}: test loss {loss:.4f} (ppl {float(torch.tensor(loss).exp()):.2f}) "
                  f"on {cache['n_files']} files / {n_tok} target tokens", flush=True)
    report["seconds"] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
