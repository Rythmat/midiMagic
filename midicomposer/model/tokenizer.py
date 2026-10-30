"""Event tokenization of root-relative multi-track music and example construction.

Sequence layout (one example = one target track or one masked section):

    <BOS> GENRE EMOTION ROOT MODE TEMPO METER DENSITY FORM SECTION INROLE OUTROLE INSTR <SEP>
    ( <TRACK> ROLE PROG (<BAR> events)*  <TRACK_END> )*          context tracks (may contain <FILL>)
    <TARGET> ROLE PROG                                            target header (given)
    (<BAR> events)* <TRACK_END> <EOS>                             track task      } loss is computed
    <FILL_START> (<BAR> events)* <FILL_END> <EOS>                 infill task     } only on these

Bar events: POS=p (once per onset), then per note PITCH=rel|DRUM=n, DUR=d, VEL=v.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from ..representation import STEPS_PER_QUARTER
from .vocab import (DRUM_MAX, DRUM_MIN, MAX_POS, PITCH_MAX, PITCH_MIN, Vocab, dur_token, vel_token)

NO_BAR_OFFSET = 0     # bar id for non-bar tokens is `max_bars` (set by the model config)
SEG_PREFIX, SEG_CONTEXT, SEG_TARGET = 0, 1, 2


def encode_bar(notes: list[tuple[float, float, int, int]], is_drum: bool, bar_start: float, bar_len: float,
               vocab: Vocab) -> list[int]:
    """notes: (start, duration, pitch_or_drum, velocity) with relative pitches for pitched tracks."""
    out: list[int] = []
    last_pos = -1
    for s, d, p, v in sorted(notes, key=lambda n: (n[0], n[2])):
        pos = int(round((s - bar_start) * STEPS_PER_QUARTER))
        if pos < 0 or pos >= MAX_POS or pos >= int(round(bar_len * STEPS_PER_QUARTER)):
            continue
        if is_drum:
            if not DRUM_MIN <= p <= DRUM_MAX:
                continue
            ptok = f"DRUM={p}"
        else:
            while p > PITCH_MAX:
                p -= 12
            while p < PITCH_MIN:
                p += 12
            ptok = f"PITCH={p}"
        if pos != last_pos:
            out.append(vocab[f"POS={pos}"])
            last_pos = pos
        out += [vocab[ptok], vocab[dur_token(d * STEPS_PER_QUARTER)], vocab[vel_token(v)]]
    return out


def split_into_bars(notes, bar_starts: list[float], bar_lens: list[float]) -> list[list]:
    import bisect
    bars: list[list] = [[] for _ in bar_starts]
    for n in notes:
        i = bisect.bisect_right(bar_starts, n[0] + 1e-9) - 1
        if 0 <= i < len(bars) and n[0] < bar_starts[i] + bar_lens[i] - 1e-9:
            bars[i].append(n)
    return bars


def decode_events(ids: list[int], vocab: Vocab, bar_starts: list[float], is_drum: bool,
                  first_bar: int = 0) -> list[tuple[float, float, int, int]]:
    """Inverse of encode_bar over a body of <BAR> events. Returns (start, dur, pitch, velocity)."""
    out = []
    bar = first_bar - 1
    pos = 0
    pending = None
    for i in ids:
        t = vocab.tokens[i]
        if t == "<BAR>":
            bar += 1
            pos = 0
        elif t.startswith("POS="):
            pos = int(t[4:])
        elif t.startswith("PITCH=") or t.startswith("DRUM="):
            pending = [int(t.split("=")[1]), None]
        elif t.startswith("DUR=") and pending is not None:
            pending[1] = int(t[4:])
        elif t.startswith("VEL=") and pending is not None and pending[1] is not None:
            from .vocab import vel_value
            if 0 <= bar < len(bar_starts):
                start = bar_starts[bar] + pos / STEPS_PER_QUARTER
                out.append((start, pending[1] / STEPS_PER_QUARTER, pending[0], vel_value(int(t[4:]))))
            pending = None
    return out


@dataclass
class TrackTokens:
    role: str
    program: int
    is_drum: bool
    bars: list[np.ndarray]          # per-bar event token ids (no <BAR> token)

    def n_notes(self, a: int, b: int) -> int:
        return int(sum(len(x) for x in self.bars[a:b]))


@dataclass
class Example:
    ids: list[int]
    loss: list[int]
    bar_ids: list[int]
    seg_ids: list[int]


def condition_prefix(vocab: Vocab, *, genre: str, emotion: str, root: str, mode: str, tempo_tok: str, meter_tok: str,
                     density: int, form_tok: str, section_tok: str, inrole: str, outrole: str, instr: str) -> list[int]:
    return [vocab["<BOS>"], vocab.id(f"GENRE={genre}", "GENRE=unk"), vocab.id(f"EMOTION={emotion}", "EMOTION=unk"),
            vocab[f"ROOT={root}"], vocab.id(f"MODE={mode}", "MODE=major"), vocab[tempo_tok], vocab[meter_tok],
            vocab[f"DENSITY={density}"], vocab[form_tok], vocab[section_tok], vocab.id(f"INROLE={inrole}", "INROLE=none"),
            vocab.id(f"OUTROLE={outrole}", "OUTROLE=lead"), vocab[f"INSTR={instr}"], vocab["<SEP>"]]


class ExampleBuilder:
    def __init__(self, vocab: Vocab, max_bars: int):
        self.v = vocab
        self.no_bar = max_bars

    def _push(self, ex: Example, ids, loss: int, bar: int, seg: int):
        for i in ids:
            ex.ids.append(int(i))
            ex.loss.append(loss)
            ex.bar_ids.append(bar)
            ex.seg_ids.append(seg)

    def track_header(self, tr: TrackTokens) -> list[int]:
        v = self.v
        return [v[f"ROLE={tr.role}"], v["PROG=drums"] if tr.is_drum else v[f"PROG={tr.program}"]]

    def context_track(self, ex: Example, tr: TrackTokens, a: int, b: int, fill: tuple[int, int] | None = None):
        v = self.v
        self._push(ex, [v["<TRACK>"]] + self.track_header(tr), 0, self.no_bar, SEG_CONTEXT)
        for k, bar in enumerate(range(a, b)):
            if fill and fill[0] <= bar < fill[1]:
                self._push(ex, [v["<BAR>"], v["<FILL>"]], 0, k, SEG_CONTEXT)
            else:
                self._push(ex, [v["<BAR>"]], 0, k, SEG_CONTEXT)
                self._push(ex, tr.bars[bar], 0, k, SEG_CONTEXT)
        self._push(ex, [v["<TRACK_END>"]], 0, self.no_bar, SEG_CONTEXT)

    def build(self, prefix: list[int], context: list[TrackTokens], target: TrackTokens, a: int, b: int,
              fill: tuple[int, int] | None = None, target_body: bool = True) -> Example:
        """Assemble an example over bars [a, b). With `fill`, the target track appears in the context
        with <FILL> placeholders and the body regenerates only bars fill[0]..fill[1]."""
        v = self.v
        ex = Example([], [], [], [])
        self._push(ex, prefix, 0, self.no_bar, SEG_PREFIX)
        for tr in context:
            self.context_track(ex, tr, a, b)
        if fill is not None:
            self.context_track(ex, target, a, b, fill)
        self._push(ex, [v["<TARGET>"]] + self.track_header(target), 0, self.no_bar, SEG_TARGET)
        if not target_body:
            return ex
        if fill is None:
            for k, bar in enumerate(range(a, b)):
                self._push(ex, [v["<BAR>"]], 1, k, SEG_TARGET)
                self._push(ex, target.bars[bar], 1, k, SEG_TARGET)
            self._push(ex, [v["<TRACK_END>"], v["<EOS>"]], 1, self.no_bar, SEG_TARGET)
        else:
            self._push(ex, [v["<FILL_START>"]], 1, self.no_bar, SEG_TARGET)
            for bar in range(fill[0], fill[1]):
                k = bar - a
                self._push(ex, [v["<BAR>"]], 1, k, SEG_TARGET)
                self._push(ex, target.bars[bar], 1, k, SEG_TARGET)
            self._push(ex, [v["<FILL_END>"], v["<EOS>"]], 1, self.no_bar, SEG_TARGET)
        return ex


def sample_task(rng: random.Random) -> str:
    r = rng.random()
    return "track" if r < 0.55 else ("infill" if r < 0.8 else "continuation")
