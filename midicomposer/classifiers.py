"""Genre/emotion classifiers over key-invariant score features (used for candidate ranking).

Trained on the *train* split only; accuracy is reported on held-out val/test splits.
Limitation (documented in the README and in every checkpoint): the classifiers learn XMIDI's
labels and statistics. Generated candidates are out-of-distribution for them, so their scores
are one ranking signal among several, never the sole criterion.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .features import FEATURE_NAMES, N_FEATURES

LIMITATION = ("Classifiers are trained on XMIDI's train split and evaluated on held-out splits; scores on "
              "generated music are out-of-distribution estimates and are combined with rule-based quality "
              "metrics rather than used alone.")


class FeatureClassifier(nn.Module):
    def __init__(self, n_in: int, n_genre: int, n_emotion: int, hidden: int = 192, dropout: float = 0.2):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
        self.genre = nn.Linear(hidden, n_genre)
        self.emotion = nn.Linear(hidden, n_emotion)

    def forward(self, x):
        h = self.trunk(x)
        return self.genre(h), self.emotion(h)


def _load_labels(data_dir: Path, wanted: set[str]) -> dict[str, dict]:
    out = {}
    with (data_dir / "manifest.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            # Cheap prefilter: parse only records we need.
            r = json.loads(line)
            if r["path"] in wanted:
                out[r["path"]] = {"split": r["split"], "genre": r["genre"], "emotion": r["emotion"]}
    return out


def train_classifiers(data_dir: str = "data/xmidi", out_path: str = "checkpoints/classifiers.pt",
                      epochs: int = 60, device: str = "auto", seed: int = 0) -> dict:
    torch.manual_seed(seed)
    data_dir = Path(data_dir)
    X = np.load(data_dir / "features.npy")
    index = json.loads((data_dir / "features_index.json").read_text())
    labels = _load_labels(data_dir, set(index))
    genres = sorted({labels[p]["genre"] for p in index})
    emotions = sorted({labels[p]["emotion"] for p in index})
    g = np.array([genres.index(labels[p]["genre"]) for p in index])
    e = np.array([emotions.index(labels[p]["emotion"]) for p in index])
    split = np.array([labels[p]["split"] for p in index])
    tr, va, te = split == "train", split == "val", split == "test"
    mean, std = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - mean) / std
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    model = FeatureClassifier(N_FEATURES, len(genres), len(emotions)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    gw = torch.tensor(len(g[tr]) / (len(genres) * np.bincount(g[tr], minlength=len(genres)).clip(1)), dtype=torch.float32, device=dev)
    ew = torch.tensor(len(e[tr]) / (len(emotions) * np.bincount(e[tr], minlength=len(emotions)).clip(1)), dtype=torch.float32, device=dev)
    lg, le = nn.CrossEntropyLoss(weight=gw), nn.CrossEntropyLoss(weight=ew)
    Xt = torch.tensor(Xn, dtype=torch.float32, device=dev)
    gt, et = torch.tensor(g, device=dev), torch.tensor(e, device=dev)
    tr_idx = torch.tensor(np.where(tr)[0], device=dev)
    best, best_state, t0 = -1.0, None, time.time()
    for ep in range(epochs):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx), device=dev)]
        for i in range(0, len(perm), 1024):
            b = perm[i:i + 1024]
            pg, pe = model(Xt[b])
            loss = lg(pg, gt[b]) + le(pe, et[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        m = _evaluate(model, Xt, gt, et, va)
        score = m["genre_macro_f1"] + m["emotion_macro_f1"]
        if score > best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    metrics = {"val": _evaluate(model, Xt, gt, et, va), "test": _evaluate(model, Xt, gt, et, te),
               "train": _evaluate(model, Xt, gt, et, tr)}
    chance = {"genre_majority": float(np.bincount(g[te]).max() / max(1, te.sum())),
              "emotion_majority": float(np.bincount(e[te]).max() / max(1, te.sum()))}
    ckpt = {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "mean": mean, "std": std,
            "genres": genres, "emotions": emotions, "feature_names": FEATURE_NAMES, "metrics": metrics,
            "majority_baseline": chance, "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
            "data_dir": str(data_dir), "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "train_seconds": round(time.time() - t0, 1), "limitation": LIMITATION}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    report = {k: ckpt[k] for k in ("genres", "emotions", "metrics", "majority_baseline", "n_train", "n_val", "n_test",
                                   "trained_at", "train_seconds", "limitation")}
    Path(out_path).with_suffix(".json").write_text(json.dumps(report, indent=2))
    return report


@torch.no_grad()
def _evaluate(model, X, g, e, mask) -> dict:
    model.eval()
    idx = torch.tensor(np.where(mask)[0], device=X.device)
    if len(idx) == 0:
        return {}
    pg, pe = model(X[idx])
    gp, ep = pg.argmax(1).cpu().numpy(), pe.argmax(1).cpu().numpy()
    gy, ey = g[idx].cpu().numpy(), e[idx].cpu().numpy()
    return {"n": int(len(idx)), "genre_acc": round(float((gp == gy).mean()), 4),
            "emotion_acc": round(float((ep == ey).mean()), 4),
            "genre_macro_f1": round(_macro_f1(gy, gp), 4), "emotion_macro_f1": round(_macro_f1(ey, ep), 4)}


def _macro_f1(y, p) -> float:
    f1s = []
    for c in np.unique(y):
        tp = ((p == c) & (y == c)).sum()
        fp = ((p == c) & (y != c)).sum()
        fn = ((p != c) & (y == c)).sum()
        f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else 0.0


class ClassifierScorer:
    def __init__(self, path: str | Path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.genres, self.emotions = ck["genres"], ck["emotions"]
        self.mean, self.std = ck["mean"], ck["std"]
        self.metrics = ck.get("metrics", {})
        self.model = FeatureClassifier(len(self.mean), len(self.genres), len(self.emotions))
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.path = str(path)

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> dict:
        x = torch.tensor(((features - self.mean) / self.std)[None], dtype=torch.float32)
        pg, pe = self.model(x)
        pg, pe = torch.softmax(pg, -1)[0].numpy(), torch.softmax(pe, -1)[0].numpy()
        return {"genre": dict(zip(self.genres, map(float, pg))), "emotion": dict(zip(self.emotions, map(float, pe)))}


_CACHE: dict[str, ClassifierScorer | None] = {}


def load_scorer(path: str | Path | None) -> ClassifierScorer | None:
    if not path:
        return None
    key = str(path)
    if key not in _CACHE:
        _CACHE[key] = ClassifierScorer(path) if Path(path).exists() else None
    return _CACHE[key]
