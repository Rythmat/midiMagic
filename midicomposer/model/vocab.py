"""Token vocabulary for the conditional track generator.

Genre/emotion tokens are built from the labels discovered in the dataset (label_vocab.json)
and stored in each checkpoint, so the vocabulary is never hard-coded to XMIDI's classes.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..roles import TRACK_ROLES
from ..theory import GM_FAMILIES, MODES, PITCH_NAMES

SPECIALS = ["<PAD>", "<BOS>", "<EOS>", "<SEP>", "<TRACK>", "<TRACK_END>", "<BAR>", "<FILL>", "<FILL_START>",
            "<FILL_END>", "<TARGET>"]
TEMPO_BINS = [60, 76, 92, 108, 118, 122, 136, 160]          # upper edges; >160 -> last bin
METERS = ["4/4", "3/4", "6/8", "2/4", "12/8", "other"]
FORMS = ["A", "AB", "ABA", "AABA", "ABAB", "verse_chorus", "other"]
SECTIONS = ["A", "B", "C", "D", "E", "intro", "verse", "chorus", "bridge", "outro", "other"]
DUR_BINS = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36, 48, 64, 72, 96]   # in 1/12-beat steps
PITCH_MIN, PITCH_MAX = -48, 47                                  # root-relative pitch range
DRUM_MIN, DRUM_MAX = 27, 87
MAX_POS = 96                                                    # 8 quarter notes at 12 steps/quarter
N_VEL = 8
N_DENSITY = 5


@dataclass
class Vocab:
    genres: list[str]
    emotions: list[str]

    def __post_init__(self):
        toks = list(SPECIALS)
        toks += [f"GENRE={g}" for g in self.genres + ["unk"]]
        toks += [f"EMOTION={e}" for e in self.emotions + ["unk"]]
        toks += [f"ROOT={p}" for p in PITCH_NAMES]
        toks += [f"MODE={m}" for m in MODES]
        toks += [f"TEMPO={i}" for i in range(len(TEMPO_BINS))]
        toks += [f"METER={m}" for m in METERS]
        toks += [f"DENSITY={i}" for i in range(N_DENSITY)]
        toks += [f"FORM={f}" for f in FORMS]
        toks += [f"SECTION={s}" for s in SECTIONS]
        toks += [f"INROLE={r}" for r in TRACK_ROLES + ["none"]]
        toks += [f"OUTROLE={r}" for r in TRACK_ROLES]
        toks += [f"INSTR={f}" for f in GM_FAMILIES + ["drums"]]
        toks += [f"ROLE={r}" for r in TRACK_ROLES]
        toks += [f"PROG={p}" for p in range(128)] + ["PROG=drums"]
        toks += [f"POS={i}" for i in range(MAX_POS)]
        toks += [f"PITCH={p}" for p in range(PITCH_MIN, PITCH_MAX + 1)]
        toks += [f"DRUM={d}" for d in range(DRUM_MIN, DRUM_MAX + 1)]
        toks += [f"DUR={d}" for d in DUR_BINS]
        toks += [f"VEL={v}" for v in range(N_VEL)]
        self.tokens = toks
        self.index = {t: i for i, t in enumerate(toks)}

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, tok: str) -> int:
        return self.index[tok]

    def id(self, tok: str, fallback: str | None = None) -> int:
        if tok in self.index:
            return self.index[tok]
        if fallback is not None:
            return self.index[fallback]
        raise KeyError(tok)

    def ids_with_prefix(self, prefix: str) -> list[int]:
        return [i for t, i in self.index.items() if t.startswith(prefix)]

    def to_dict(self) -> dict:
        return {"genres": self.genres, "emotions": self.emotions, "size": len(self.tokens)}

    @property
    def pad(self) -> int:
        return self.index["<PAD>"]


def tempo_bin(bpm: float) -> int:
    for i, edge in enumerate(TEMPO_BINS):
        if bpm <= edge:
            return i
    return len(TEMPO_BINS) - 1


def density_bin(notes_per_bar: float) -> int:
    edges = [6, 12, 20, 32]
    for i, e in enumerate(edges):
        if notes_per_bar <= e:
            return i
    return N_DENSITY - 1


def density_from_plan(density: float) -> int:
    return max(0, min(N_DENSITY - 1, int(density * N_DENSITY)))


def dur_token(steps: float) -> str:
    best = min(DUR_BINS, key=lambda b: (abs(b - steps), b))
    return f"DUR={best}"


def vel_token(velocity: int) -> str:
    return f"VEL={max(0, min(N_VEL - 1, int(velocity) * N_VEL // 128))}"


def vel_value(bin_: int) -> int:
    return int((bin_ + 0.5) * 128 / N_VEL)


def meter_token(label: str) -> str:
    return f"METER={label if label in METERS else 'other'}"


def form_token(letters: list[str]) -> str:
    seq = []
    for l in letters:
        base = l.rstrip("'")[:1]
        if not seq or seq[-1] != base:
            seq.append(base)
    s = "".join(seq)
    for f in ("AABA", "ABAB", "ABA", "AB"):
        if s.startswith(f):
            return f"FORM={f}"
    return "FORM=A" if s == "A" else "FORM=other"


def section_token(label: str | None) -> str:
    if not label:
        return "SECTION=other"
    l = label.rstrip("'").lower()
    if l in ("intro", "verse", "chorus", "bridge", "outro"):
        return f"SECTION={l}"
    u = l[:1].upper()
    return f"SECTION={u}" if u in "ABCDE" else "SECTION=other"
