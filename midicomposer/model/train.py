"""Training loop for the hierarchical track generator.

    python -m midicomposer.cli train --config configs/dev_train.yaml [--set max_steps=200]

The checkpoint records exactly what was trained on (file counts per split, steps, tokens seen,
losses) and whether that covers every usable training file - it never claims more.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .. import __version__
from .data import MaskedTrackDataset, build_token_cache, collate
from .hierarchical import HierarchicalComposer, ModelConfig
from .vocab import Vocab

DEFAULTS = {
    "data_dir": "data/xmidi", "xmidi_root": "XMIDI_Dataset", "cache_dir": "data/token_cache",
    "max_train_files": 2000, "max_val_files": 200, "max_bars_per_file": 128, "window_bars": 8,
    "max_len": 1024, "d_model": 256, "n_layers": 6, "n_heads": 8, "d_ff": 1024, "bar_layers": 2, "max_bars": 64,
    "dropout": 0.1, "batch_size": 16, "lr": 6e-4, "weight_decay": 0.05, "warmup_steps": 200, "max_steps": 2000,
    "eval_every": 250, "eval_batches": 20, "grad_clip": 1.0, "seed": 0, "device": "auto", "amp": True,
    "num_workers": 0, "prep_workers": 16, "out": "checkpoints/composer_dev.pt", "log_every": 50,
}


def load_config(path: str | Path | None, overrides: dict | None = None) -> dict:
    cfg = dict(DEFAULTS)
    if path:
        cfg.update(yaml.safe_load(Path(path).read_text()) or {})
    cfg.update(overrides or {})
    return cfg


def _count_usable(data_dir: str, split: str) -> int:
    n = 0
    with (Path(data_dir) / "manifest.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if f'"split": "{split}"' in line and '"usable_for_training": true' in line:
                n += 1
    return n


def _label_lists(data_dir: str) -> tuple[list[str], list[str]]:
    lv = json.loads((Path(data_dir) / "label_vocab.json").read_text())
    return sorted(e["label"] for e in lv["genres"]), sorted(e["label"] for e in lv["emotions"])


def train_from_config(config_path: str | Path | None, overrides: dict | None = None) -> dict:
    cfg = load_config(config_path, overrides)
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    dev = torch.device("cuda" if cfg["device"] == "auto" and torch.cuda.is_available() else
                       ("cpu" if cfg["device"] == "auto" else cfg["device"]))
    genres, emotions = _label_lists(cfg["data_dir"])
    vocab = Vocab(genres, emotions)
    cache_dir = Path(cfg["cache_dir"])
    t0 = time.time()
    print(f"[train] building token caches (train<= {cfg['max_train_files']}, val<= {cfg['max_val_files']} files)", flush=True)
    tr_cache = build_token_cache(cfg["data_dir"], cfg["xmidi_root"], "train",
                                 cache_dir / f"train_{cfg['max_train_files']}.pkl", vocab, cfg["max_train_files"],
                                 cfg["prep_workers"], cfg["max_bars_per_file"], cfg["seed"])
    va_cache = build_token_cache(cfg["data_dir"], cfg["xmidi_root"], "val",
                                 cache_dir / f"val_{cfg['max_val_files']}.pkl", vocab, cfg["max_val_files"],
                                 cfg["prep_workers"], cfg["max_bars_per_file"], cfg["seed"])
    print(f"[train] caches ready in {time.time() - t0:.1f}s: {tr_cache['n_files']} train / {va_cache['n_files']} val files",
          flush=True)

    mcfg = ModelConfig(vocab_size=len(vocab), d_model=cfg["d_model"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
                       d_ff=cfg["d_ff"], bar_layers=cfg["bar_layers"], max_len=cfg["max_len"],
                       max_bars=cfg["max_bars"], dropout=cfg["dropout"])
    model = HierarchicalComposer(mcfg).to(dev)
    print(f"[train] model params: {model.n_params() / 1e6:.2f}M on {dev}", flush=True)
    steps = cfg["max_steps"]
    bs = cfg["batch_size"]
    tr_ds = MaskedTrackDataset(tr_cache, vocab, cfg["window_bars"], cfg["max_len"], cfg["max_bars"], steps * bs, cfg["seed"])
    va_ds = MaskedTrackDataset(va_cache, vocab, cfg["window_bars"], cfg["max_len"], cfg["max_bars"],
                               cfg["eval_batches"] * bs, cfg["seed"] + 1)
    col = lambda b: collate(b, vocab.pad, cfg["max_bars"])  # noqa: E731
    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=False, collate_fn=col, num_workers=cfg["num_workers"])
    va_dl = DataLoader(va_ds, batch_size=bs, shuffle=False, collate_fn=col, num_workers=0)

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg["lr"], betas=(0.9, 0.95))

    def lr_at(step: int) -> float:
        if step < cfg["warmup_steps"]:
            return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
        prog = (step - cfg["warmup_steps"]) / max(1, steps - cfg["warmup_steps"])
        return cfg["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))

    use_amp = cfg["amp"] and dev.type == "cuda"
    history, best_val, tokens_seen, step = [], float("inf"), 0, 0
    out_path = Path(cfg["out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_train = time.time()
    model.train()
    run_loss = run_n = 0.0
    for batch in tr_dl:
        if step >= steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        ids, bars, segs, lmask = (batch[k].to(dev, non_blocking=True) for k in ("ids", "bars", "segs", "loss"))
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(ids[:, :-1], bars[:, :-1], segs[:, :-1])
        loss = _masked_ce(logits.float(), ids[:, 1:], lmask[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        tokens_seen += int(lmask[:, 1:].sum().item())
        run_loss += loss.item()
        run_n += 1
        step += 1
        if step % cfg["log_every"] == 0:
            print(f"[train] step {step}/{steps} loss {run_loss / run_n:.4f} lr {lr_at(step):.2e} "
                  f"{(time.time() - t_train) / step:.3f}s/step", flush=True)
        if step % cfg["eval_every"] == 0 or step == steps:
            val = evaluate(model, va_dl, dev, use_amp)
            history.append({"step": step, "train_loss": round(run_loss / max(run_n, 1), 4), "val_loss": round(val, 4)})
            print(f"[train] eval step {step}: train {run_loss / max(run_n, 1):.4f} val {val:.4f}", flush=True)
            run_loss = run_n = 0.0
            if val < best_val:
                best_val = val
                _save(out_path, model, mcfg, vocab, cfg, history, step, tokens_seen, tr_cache, va_cache, t_train, best_val)
            model.train()
    if not history:
        raise RuntimeError("training produced no evaluation; increase max_steps")
    info = json.loads(out_path.with_suffix(".json").read_text())
    info["final_step"] = step
    info["history"] = history
    out_path.with_suffix(".json").write_text(json.dumps(info, indent=2))
    return info


def _masked_ce(logits, targets, mask):
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
    mask = mask.reshape(-1)
    return (ce * mask).sum() / mask.sum().clamp(min=1)


@torch.no_grad()
def evaluate(model, loader, dev, use_amp) -> float:
    model.eval()
    tot = n = 0.0
    for batch in loader:
        ids, bars, segs, lmask = (batch[k].to(dev) for k in ("ids", "bars", "segs", "loss"))
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(ids[:, :-1], bars[:, :-1], segs[:, :-1])
        ce = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1),
                                               reduction="none")
        m = lmask[:, 1:].reshape(-1)
        tot += float((ce * m).sum())
        n += float(m.sum())
    return tot / max(n, 1)


def _save(path: Path, model, mcfg, vocab, cfg, history, step, tokens_seen, tr_cache, va_cache, t_train, best_val):
    n_usable_train = _count_usable(cfg["data_dir"], "train")
    info = {
        "model_version": f"hierarchical-composer-{__version__}-step{step}",
        "package_version": __version__, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "step": step, "max_steps": cfg["max_steps"], "tokens_seen_target": tokens_seen,
        "best_val_loss": round(best_val, 4), "train_seconds": round(time.time() - t_train, 1),
        "n_params": model.n_params(), "model_config": mcfg.to_dict(),
        "train_files": tr_cache["n_files"], "val_files": va_cache["n_files"],
        "usable_train_files_in_manifest": n_usable_train,
        "trained_on_all_usable_train_files": tr_cache["n_files"] >= n_usable_train,
        "data_dir": cfg["data_dir"], "window_bars": cfg["window_bars"],
        "training_tasks": ["masked-track reconstruction (random context subsets incl. none)",
                           "masked-section infilling", "continuation"],
        "vocab": vocab.to_dict(), "history": history,
        "honesty_note": (f"Trained on {tr_cache['n_files']} of {n_usable_train} usable XMIDI training files "
                         f"for {step} steps. This is a development-scale model, not a fully trained one."
                         if tr_cache["n_files"] < n_usable_train else
                         f"Trained on all {n_usable_train} usable XMIDI training files for {step} steps."),
    }
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "model_config": mcfg.to_dict(),
                "genres": vocab.genres, "emotions": vocab.emotions, "info": info, "train_config": cfg}, path)
    path.with_suffix(".json").write_text(json.dumps(info, indent=2))
