"""Neural track generation from a trained HierarchicalComposer checkpoint.

Decoding is grammar-constrained so output is always structurally valid: the exact number of
bars, strictly increasing positions inside a bar, pitch tokens restricted to the role's register
(root-relative), drum tokens only for drum tracks, and one note per onset for monophonic roles.
Long pieces are generated window by window (the training window size), each conditioned on the
other tracks in that window.
"""
from __future__ import annotations

from pathlib import Path

import torch

from ..roles import GEN_TO_TRACK_ROLE, MONOPHONIC_GEN_ROLES
from ..representation import STEPS_PER_QUARTER
from ..theory import gm_family
from ..model.hierarchical import HierarchicalComposer, ModelConfig
from ..model.tokenizer import ExampleBuilder, TrackTokens, condition_prefix, decode_events, encode_bar, split_into_bars
from ..model.vocab import (MAX_POS, Vocab, density_from_plan, meter_token, section_token, tempo_bin)
from .rule_based import GenContext, RelNote

_LOADED: dict[str, "NeuralTrackGenerator | None"] = {}


def load_neural_generator(path: str | None) -> "NeuralTrackGenerator | None":
    if not path or not Path(path).exists():
        return None
    if path not in _LOADED:
        try:
            _LOADED[path] = NeuralTrackGenerator(path)
        except Exception:
            _LOADED[path] = None
    return _LOADED[path]


class NeuralTrackGenerator:
    def __init__(self, path: str, device: str = "auto", temperature: float = 0.95, top_p: float = 0.95):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.path = str(path)
        self.vocab = Vocab(ck["genres"], ck["emotions"])
        self.cfg = ModelConfig(**ck["model_config"])
        self.dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                                ("cpu" if device == "auto" else device))
        self.model = HierarchicalComposer(self.cfg)
        self.model.load_state_dict(ck["state_dict"])
        self.model.to(self.dev).eval()
        self.info = ck["info"]
        self.window = int(ck.get("train_config", {}).get("window_bars", 8))
        self.version = self.info.get("model_version", "unknown")
        self.builder = ExampleBuilder(self.vocab, self.cfg.max_bars)
        self.temperature, self.top_p = temperature, top_p
        v = self.vocab
        self.pos_ids = {int(t[4:]): i for t, i in v.index.items() if t.startswith("POS=")}
        self.pitch_ids = {int(t[6:]): i for t, i in v.index.items() if t.startswith("PITCH=")}
        self.drum_ids = {int(t[5:]): i for t, i in v.index.items() if t.startswith("DRUM=")}
        self.dur_ids = v.ids_with_prefix("DUR=")
        self.vel_ids = v.ids_with_prefix("VEL=")

    def describe(self) -> dict:
        keys = ("model_version", "trained_at", "step", "train_files", "usable_train_files_in_manifest",
                "trained_on_all_usable_train_files", "best_val_loss", "n_params", "honesty_note")
        return {"path": self.path, **{k: self.info.get(k) for k in keys}}

    def supports(self, role: str) -> bool:
        return role in GEN_TO_TRACK_ROLE

    # ------------------------------------------------------------------ prompt construction
    def _tracks_for_context(self, ctx: GenContext, starts, lens, exclude_role: str) -> list[TrackTokens]:
        out = []
        n = len(starts)
        for role, program, is_drum, notes in (ctx.anchor_tracks or []):
            bars = split_into_bars([(x.start, x.duration, x.pitch, x.velocity) for x in notes], starts, lens)
            out.append(TrackTokens(role or "lead", program, is_drum,
                                   [encode_bar(bars[i], is_drum, starts[i], lens[i], self.vocab) for i in range(n)]))
        for gen_role, notes in ctx.generated.items():
            if gen_role == exclude_role:
                continue
            is_drum = gen_role == "drums"
            bars = split_into_bars([(x.start, x.duration, x.pitch, x.velocity) for x in notes], starts, lens)
            out.append(TrackTokens(GEN_TO_TRACK_ROLE.get(gen_role, "lead"), ctx.plan.programs.get(gen_role, 0), is_drum,
                                   [encode_bar(bars[i], is_drum, starts[i], lens[i], self.vocab) for i in range(n)]))
        return out

    def _scaffold(self, ctx: GenContext, starts, lens) -> TrackTokens:
        """Hidden block-chord track built from the plan's progression. Used only as conditioning
        (never output) so that independently generated parts share one harmonic plan."""
        notes = []
        for c in ctx.plan.chords:
            for pc in c.pcs[:3]:
                rel = pc if pc <= 7 else pc - 12              # close voicing around the tonic (-5..+7)
                notes.append((c.start, c.end - c.start, rel, 64))
        bars = split_into_bars(notes, starts, lens)
        return TrackTokens("chordal", 0, False,
                           [encode_bar(bars[i], False, starts[i], lens[i], self.vocab) for i in range(len(starts))])

    def generate(self, role: str, ctx: GenContext) -> list[RelNote]:
        plan, grid = ctx.plan, ctx.grid
        n_bars = plan.n_bars
        starts = [grid.bar_start(b) for b in range(n_bars)]
        lens = [grid.bar_length(b) for b in range(n_bars)]
        track_role = GEN_TO_TRACK_ROLE[role]
        is_drum = role == "drums"
        program = plan.programs.get(role, 0)
        target = TrackTokens(track_role, program, is_drum, [[] for _ in range(n_bars)])
        context = self._tracks_for_context(ctx, starts, lens, role)
        if not ctx.anchor_tracks and plan.chords:
            context.insert(0, self._scaffold(ctx, starts, lens))
        lo, hi = plan.registers.get(role, (-24, 24))
        seed = ctx.rng.randrange(2 ** 31)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        out: list[RelNote] = []
        form_tok = "FORM=other"
        for a in range(0, n_bars, self.window):
            b = min(n_bars, a + self.window)
            active_ctx = [t for t in context if t.n_notes(a, b) > 0]
            inrole = active_ctx[0].role if len(active_ctx) == 1 else ("full_mix" if active_ctx else "none")
            prefix = condition_prefix(
                self.vocab, genre=plan.genre, emotion=plan.emotion, root=plan.root_name, mode=plan.mode,
                tempo_tok=f"TEMPO={tempo_bin(plan.tempo)}", meter_tok=meter_token(plan.meter.label),
                density=density_from_plan(plan.density), form_tok=form_tok,
                section_tok=section_token(plan.section_of_bar(a).name), inrole=inrole, outrole=track_role,
                instr="drums" if is_drum else gm_family(program))
            budget = 40 * (b - a) + 8
            while True:
                ex = self.builder.build(prefix, active_ctx, target, a, b, target_body=False)
                if len(ex.ids) + budget <= self.cfg.max_len or not active_ctx:
                    break
                active_ctx.remove(max(active_ctx, key=lambda t: t.n_notes(a, b)))
            body = self._sample(ex, b - a, [lens[i] for i in range(a, b)], is_drum, role in MONOPHONIC_GEN_ROLES,
                                (lo, hi), budget, gen)
            for s, d, p, v in decode_events(body, self.vocab, starts[a:b], is_drum):
                out.append(RelNote(s, max(1 / STEPS_PER_QUARTER, d), p, v))
        return [n for n in out if ctx.active(role, grid.locate(n.start)[0])]

    # ------------------------------------------------------------------ constrained sampling
    @torch.no_grad()
    def _sample(self, ex, n_bars: int, bar_lens: list[float], is_drum: bool, mono: bool, register, budget: int,
                gen: torch.Generator) -> list[int]:
        v, m = self.vocab, self.model
        ids = torch.tensor([ex.ids], device=self.dev)
        bars = torch.tensor([ex.bar_ids], device=self.dev)
        segs = torch.tensor([ex.seg_ids], device=self.dev)
        logits, caches, bar_ctx = m.prefill(ids, bars, segs)
        position = len(ex.ids)
        body: list[int] = []
        bar_tok, end_tok = v["<BAR>"], v["<TRACK_END>"]
        pitch_allowed = ([i for d, i in self.drum_ids.items()] if is_drum else
                         [i for p, i in self.pitch_ids.items() if register[0] <= p <= register[1]])
        state, bar_k, last_pos, notes_in_bar = "need_bar", -1, -1, 0

        def feed(tok: int, bar_id: int):
            nonlocal logits, position
            body.append(tok)
            logits = m.step(tok, bar_id, position, caches, bar_ctx)
            position += 1

        while True:
            if position >= self.cfg.max_len - 2:
                break
            if state == "need_bar":
                if bar_k + 1 >= n_bars:
                    break
                bar_k += 1
                last_pos, notes_in_bar = -1, 0
                feed(bar_tok, bar_k)
                state = "after_bar"
                continue
            allowed: list[int] = []
            steps_in_bar = min(MAX_POS, int(round(bar_lens[bar_k] * STEPS_PER_QUARTER)))
            if state in ("after_bar", "after_vel"):
                if notes_in_bar < 48 and len(body) < budget:
                    allowed += [i for p, i in self.pos_ids.items() if last_pos < p < steps_in_bar]
                    if state == "after_vel" and not mono:
                        allowed += pitch_allowed
                allowed.append(bar_tok if bar_k + 1 < n_bars else end_tok)
            elif state == "after_pos":
                allowed = pitch_allowed
            elif state == "after_pitch":
                allowed = self.dur_ids
            elif state == "after_dur":
                allowed = self.vel_ids
            tok = self._pick(logits[0], allowed, gen)
            t = v.tokens[tok]
            if tok == end_tok:
                break
            if tok == bar_tok:
                state = "need_bar"
                continue
            if t.startswith("POS="):
                last_pos = int(t[4:])
                state = "after_pos"
            elif t.startswith("PITCH=") or t.startswith("DRUM="):
                state = "after_pitch"
                notes_in_bar += 1
            elif t.startswith("DUR="):
                state = "after_dur"
            elif t.startswith("VEL="):
                state = "after_vel"
            feed(tok, bar_k)
        # Pad any bars the budget cut short, so decoding offsets stay aligned.
        while bar_k + 1 < n_bars:
            bar_k += 1
            body.append(bar_tok)
        return body

    def _pick(self, logits: torch.Tensor, allowed: list[int], gen: torch.Generator) -> int:
        if len(allowed) == 1:
            return allowed[0]
        idx = torch.tensor(allowed, device=logits.device)
        lg = logits.float()[idx] / max(self.temperature, 1e-3)
        probs = torch.softmax(lg, dim=-1).cpu()
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, 0)
        keep = cum - sp < self.top_p
        sp = sp * keep
        choice = torch.multinomial(sp / sp.sum(), 1, generator=gen).item()
        return allowed[int(si[choice])]
