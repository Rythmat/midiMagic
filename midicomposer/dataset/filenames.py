"""Robust XMIDI filename parsing: XMIDI_<Emotion>_<Genre>_<ID>.midi

Labels are discovered dynamically, never from a fixed list:
  * underscore is the primary field separator; '-' and ' ' join words *inside* a label;
  * names with exactly two body fields are unambiguous and seed a label vocabulary;
  * names with more fields (e.g. 'very_happy_hip_hop') are split using that vocabulary
    (whole-label and word-level evidence). Unresolvable splits are flagged 'ambiguous'.
Raw label text is preserved alongside the normalized (lower snake_case) label.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

MIDI_EXTENSIONS = (".mid", ".midi")
_PREFIX_RE = re.compile(r"^xmidi[_\- ]+", re.IGNORECASE)
_ID_RE = re.compile(r"^[A-Za-z0-9]{4,16}$")
EXPECTED_ID_LEN = 8


def normalize_label(raw: str) -> str:
    words = [w for w in re.split(r"[_\-\s]+", raw.strip().lower()) if w]
    return "_".join(words)


def display_label(norm: str) -> str:
    return " ".join(w.capitalize() for w in norm.split("_"))


@dataclass
class ParsedName:
    filename: str
    file_id: str | None
    raw_emotion: str | None
    raw_genre: str | None
    emotion: str | None
    genre: str | None
    status: str                     # ok | resolved_by_vocabulary | ambiguous | unparsed
    id_length_ok: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LabelVocab:
    emotions: Counter = field(default_factory=Counter)
    genres: Counter = field(default_factory=Counter)

    @property
    def emotion_words(self) -> set[str]:
        return {w for e in self.emotions for w in e.split("_")}

    @property
    def genre_words(self) -> set[str]:
        return {w for g in self.genres for w in g.split("_")}


def _split_body(stem: str) -> tuple[list[str], str | None] | None:
    m = _PREFIX_RE.match(stem)
    if not m:
        return None
    body = stem[m.end():]
    fields = [f for f in body.split("_") if f.strip()]
    if len(fields) < 2:
        return None
    file_id = fields[-1].strip()
    if not _ID_RE.match(file_id):
        return fields, None
    return fields[:-1], file_id


def build_vocab(filenames: list[str], hints: dict | None = None) -> LabelVocab:
    """Seed the label vocabulary from unambiguous names (+ optional user hints)."""
    vocab = LabelVocab()
    for name in filenames:
        sp = _split_body(Path(name).stem)
        if not sp:
            continue
        fields, file_id = sp
        if file_id is None:
            continue
        if len(fields) == 2:
            vocab.emotions[normalize_label(fields[0])] += 1
            vocab.genres[normalize_label(fields[1])] += 1
        elif len(fields) == 1:
            words = [w for w in re.split(r"[\-\s]+", fields[0]) if w]
            if len(words) == 2:
                vocab.emotions[normalize_label(words[0])] += 1
                vocab.genres[normalize_label(words[1])] += 1
    for e in (hints or {}).get("emotions", []):
        vocab.emotions[normalize_label(e)] += 0
    for g in (hints or {}).get("genres", []):
        vocab.genres[normalize_label(g)] += 0
    return vocab


def parse_filename(name: str, vocab: LabelVocab | None = None) -> ParsedName:
    base = Path(name).name
    stem, ext = Path(base).stem, Path(base).suffix.lower()
    notes = []
    if ext not in MIDI_EXTENSIONS:
        notes.append(f"unexpected extension {ext!r}")
    sp = _split_body(stem)
    if not sp:
        return ParsedName(base, None, None, None, None, None, "unparsed", notes=notes + ["does not match XMIDI_<Emotion>_<Genre>_<ID>"])
    fields, file_id = sp
    if file_id is None:
        return ParsedName(base, None, None, None, None, None, "unparsed", notes=notes + ["missing alphanumeric ID"])
    id_ok = len(file_id) == EXPECTED_ID_LEN
    if not id_ok:
        notes.append(f"ID length {len(file_id)} != {EXPECTED_ID_LEN}")

    if len(fields) == 1:
        words = [w for w in re.split(r"[\-\s]+", fields[0]) if w]
        if len(words) != 2:
            return ParsedName(base, file_id, None, None, None, None, "unparsed", id_ok,
                              notes + ["cannot separate emotion and genre"])
        fields = words
    if len(fields) == 2:
        re_, rg = fields[0].strip(), fields[1].strip()
        return ParsedName(base, file_id, re_, rg, normalize_label(re_), normalize_label(rg), "ok", id_ok, notes)

    # More than two fields: choose the split point using vocabulary evidence.
    vocab = vocab or LabelVocab()
    ew, gw = vocab.emotion_words, vocab.genre_words
    scored = []
    for k in range(1, len(fields)):
        emo = normalize_label("_".join(fields[:k]))
        gen = normalize_label("_".join(fields[k:]))
        s = 0.0
        s += 3.0 if emo in vocab.emotions else 0.0
        s += 3.0 if gen in vocab.genres else 0.0
        for w in emo.split("_"):
            s += 1.0 if w in ew else (-1.0 if w in gw else 0.0)
        for w in gen.split("_"):
            s += 1.0 if w in gw else (-1.0 if w in ew else 0.0)
        if emo.split("_")[-1] in ew:
            s += 0.5
        scored.append((s, k))
    scored.sort(key=lambda x: (-x[0], x[1]))
    best_s, best_k = scored[0]
    tie = len(scored) > 1 and abs(scored[1][0] - best_s) < 1e-9
    if tie or best_s <= 0:
        k, status = 1, "ambiguous"
        notes.append(f"could not resolve emotion/genre split of {len(fields)} fields; defaulted to first field as emotion")
    else:
        k, status = best_k, "resolved_by_vocabulary"
    re_, rg = "_".join(fields[:k]), "_".join(fields[k:])
    return ParsedName(base, file_id, re_, rg, normalize_label(re_), normalize_label(rg), status, id_ok, notes)


def discover_midi_files(root: str | Path) -> list[Path]:
    """Recursively find MIDI files (case-insensitive extensions), sorted for determinism."""
    root = Path(root)
    out = []
    import os
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(MIDI_EXTENSIONS):
                out.append(Path(dirpath) / f)
    return sorted(out, key=lambda p: p.relative_to(root).as_posix())
