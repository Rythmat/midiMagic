"""Training data: token cache built from the prepared manifest + random-masking example sampler.

Every complete XMIDI piece yields many training examples: a random window of bars, a random
target track, a random subset of the other tracks as context (including none, which trains
metadata-only generation), and a random task (masked-track reconstruction, masked-section
infilling or continuation).
"""
from __future__ import annotations

import json
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..analyzer import detect_sections
from ..midi_io import load_midi
from ..representation import normalize_score
from ..root import RootAnalysis
from ..theory import gm_family
from .tokenizer import ExampleBuilder, TrackTokens, condition_prefix, encode_bar, sample_task, split_into_bars
from .vocab import Vocab, density_bin, form_token, meter_token, section_token, tempo_bin


def select_records(data_dir: str | Path, split: str, max_files: int | None, seed: int = 0) -> list[dict]:
    """Usable manifest records of one split (deterministic subset when max_files is set)."""
    keep = []
    with (Path(data_dir) / "manifest.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if f'"split": "{split}"' not in line or '"usable_for_training": true' not in line:
                continue
            r = json.loads(line)
            keep.append({k: r[k] for k in ("path", "genre", "emotion", "root_name", "root_pitch_class", "mode",
                                           "root_confidence", "tempo_bpm", "meter", "tracks", "n_bars")})
    keep.sort(key=lambda r: r["path"])
    if max_files and len(keep) > max_files:
        rng = random.Random(seed)
        keep = sorted(rng.sample(keep, max_files), key=lambda r: r["path"])
    return keep


def _tokenize_one(args) -> dict | None:
    path, rec, genres, emotions, max_bars_per_file = args
    vocab = Vocab(genres, emotions)
    try:
        score = load_midi(path)
    except Exception:
        return None
    root = RootAnalysis(rec["root_pitch_class"], rec["root_name"], rec["mode"], None, rec["root_confidence"],
                        "manifest", False)
    roles = {t["track_id"]: t["role"] for t in rec["tracks"]}
    piece = normalize_score(score, root=root, roles=roles)
    grid = score.bar_grid()
    n_bars = min(grid.n_bars, max_bars_per_file)
    starts = grid.bar_starts[:n_bars]
    lens = grid.bar_lengths[:n_bars]
    tracks = []
    for nt in piece.tracks:
        notes = [(n.onset, n.duration, n.original_pitch if nt.is_drum else n.relative_pitch, n.velocity) for n in nt.notes]
        per_bar = split_into_bars(notes, starts, lens)
        bars = [np.asarray(encode_bar(per_bar[i], nt.is_drum, starts[i], lens[i], vocab), dtype=np.int16)
                for i in range(n_bars)]
        tracks.append(TrackTokens(nt.role or "lead", nt.program, nt.is_drum, bars))
    try:
        secs = detect_sections(score, grid, rec["root_pitch_class"])
    except Exception:
        secs = [{"label": "A", "start_bar": 0, "end_bar": grid.n_bars}]
    sec_per_bar = ["A"] * n_bars
    for s in secs:
        for b in range(s["start_bar"], min(s["end_bar"], n_bars)):
            sec_per_bar[b] = s["label"]
    n_notes = sum(len(t.notes) for t in score.tracks)
    return {"path": rec["path"], "genre": rec["genre"], "emotion": rec["emotion"], "root": rec["root_name"],
            "mode": rec["mode"], "tempo_tok": f"TEMPO={tempo_bin(rec['tempo_bpm'])}", "meter_tok": meter_token(rec["meter"]),
            "density": density_bin(n_notes / max(1, grid.n_bars)), "form_tok": form_token([s["label"] for s in secs]),
            "sections": sec_per_bar, "n_bars": n_bars, "tracks": tracks}


def build_token_cache(data_dir: str | Path, xmidi_root: str | Path, split: str, out_path: str | Path,
                      vocab: Vocab, max_files: int | None, workers: int = 8, max_bars_per_file: int = 128,
                      seed: int = 0) -> dict:
    out_path = Path(out_path)
    if out_path.exists():
        with out_path.open("rb") as fh:
            cache = pickle.load(fh)
        if cache.get("max_files") == max_files and cache.get("genres") == vocab.genres:
            return cache
    recs = select_records(data_dir, split, max_files, seed)
    jobs = [(str(Path(xmidi_root) / r["path"]), r, vocab.genres, vocab.emotions, max_bars_per_file) for r in recs]
    items = []
    if workers > 1 and len(jobs) > 32:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            items = [x for x in ex.map(_tokenize_one, jobs, chunksize=16) if x is not None]
    else:
        items = [x for x in map(_tokenize_one, jobs) if x is not None]
    cache = {"split": split, "max_files": max_files, "genres": vocab.genres, "emotions": vocab.emotions,
             "n_files": len(items), "items": items}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return cache


def instr_of(tr: TrackTokens) -> str:
    return "drums" if tr.is_drum else gm_family(tr.program)


def make_example(rec: dict, rng: random.Random, builder: ExampleBuilder, vocab: Vocab, window: int, max_len: int):
    n_bars = rec["n_bars"]
    if n_bars < 1:
        return None
    w = min(window, n_bars)
    for _attempt in range(3):
        a = rng.randrange(0, n_bars - w + 1)
        b = a + w
        tracks = [t for t in rec["tracks"] if t.n_notes(a, b) > 0]
        if tracks:
            break
    else:
        return None
    target = rng.choice(tracks)
    others = [t for t in tracks if t is not target]
    task = sample_task(rng)
    fill = None
    if task == "track":
        k = 0 if (rng.random() < 0.15 or not others) else rng.randint(1, len(others))
        context = rng.sample(others, k)
    else:
        context = rng.sample(others, rng.randint(0, len(others))) if others else []
        if w < 2:
            task, fill = "track", None
        elif task == "continuation":
            fill = (a + rng.randint(1, w - 1), b)
        else:
            L = rng.randint(1, max(1, w // 2))
            fa = rng.randint(a, b - L)
            fill = (fa, fa + L)
    while True:
        if context:
            inrole = context[0].role if len(context) == 1 else "full_mix"
        else:
            inrole = target.role if fill else "none"
        prefix = condition_prefix(
            vocab, genre=rec["genre"], emotion=rec["emotion"], root=rec["root"], mode=rec["mode"] or "major",
            tempo_tok=rec["tempo_tok"], meter_tok=rec["meter_tok"], density=rec["density"], form_tok=rec["form_tok"],
            section_tok=section_token(rec["sections"][a]), inrole=inrole, outrole=target.role, instr=instr_of(target))
        ex = builder.build(prefix, context, target, a, b, fill)
        if len(ex.ids) <= max_len:
            return ex
        if context:
            context.remove(max(context, key=lambda t: t.n_notes(a, b)))
        elif b - a > 1:
            b = a + max(1, (b - a) // 2)
            if fill:
                fill = (min(fill[0], b - 1), b) if fill[0] < b else (b - 1, b)
        else:
            return None


class MaskedTrackDataset(Dataset):
    def __init__(self, cache: dict, vocab: Vocab, window: int, max_len: int, max_bars: int,
                 examples_per_epoch: int, seed: int = 0):
        self.items = cache["items"]
        self.vocab = vocab
        self.window = window
        self.max_len = max_len
        self.builder = ExampleBuilder(vocab, max_bars)
        self.n = examples_per_epoch
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        rng = random.Random(self.seed * 1_000_003 + i)
        for _ in range(10):
            rec = self.items[rng.randrange(len(self.items))]
            ex = make_example(rec, rng, self.builder, self.vocab, self.window, self.max_len)
            if ex is not None and sum(ex.loss) > 2:
                return ex
        raise RuntimeError("could not sample a training example")


def collate(batch, pad_id: int, no_bar: int):
    T = max(len(e.ids) for e in batch)
    B = len(batch)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    bars = torch.full((B, T), no_bar, dtype=torch.long)
    segs = torch.zeros((B, T), dtype=torch.long)
    loss = torch.zeros((B, T), dtype=torch.float32)
    for i, e in enumerate(batch):
        n = len(e.ids)
        ids[i, :n] = torch.tensor(e.ids)
        bars[i, :n] = torch.tensor(e.bar_ids)
        segs[i, :n] = torch.tensor(e.seg_ids)
        loss[i, :n] = torch.tensor(e.loss, dtype=torch.float32)
    return {"ids": ids, "bars": bars, "segs": segs, "loss": loss}
