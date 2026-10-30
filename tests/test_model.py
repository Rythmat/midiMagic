"""Vocabulary, tokenizer, hierarchical model invariants, training smoke run and neural inference."""
import random
from pathlib import Path

import pytest
import torch

from build_fixtures import tonal_piece
from midicomposer.dataset.prepare import PrepareConfig, prepare_dataset
from midicomposer.generation.pipeline import GenerationRequest, run_generation
from midicomposer.midi_io import Score, score_to_bytes
from midicomposer.model.hierarchical import HierarchicalComposer, ModelConfig
from midicomposer.model.tokenizer import (SEG_TARGET, ExampleBuilder, TrackTokens, condition_prefix, decode_events,
                                          encode_bar)
from midicomposer.model.vocab import Vocab

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def vocab():
    return Vocab(["pop", "rock"], ["happy", "sad"])


def test_vocab_is_built_from_discovered_labels(vocab):
    assert vocab.id("GENRE=pop") != vocab.id("GENRE=rock")
    assert vocab.id("GENRE=metal", "GENRE=unk") == vocab["GENRE=unk"]
    assert "GENRE=jazz" not in vocab.index                     # not hard-coded
    assert len(vocab) == len(set(vocab.tokens))


def test_bar_round_trip(vocab):
    notes = [(0.0, 1.0, 0, 100), (0.0, 1.0, 4, 90), (1.5, 0.5, -5, 70), (3.0, 1.0, 12, 64)]
    ids = encode_bar(notes, False, 0.0, 4.0, vocab)
    toks = [vocab.tokens[i] for i in ids]
    assert toks[:4] == ["POS=0", "PITCH=0", "DUR=12", toks[3]] and toks.count("POS=0") == 1
    back = decode_events([vocab["<BAR>"]] + ids, vocab, [0.0], False)
    assert [(s, d, p) for s, d, p, _ in back] == [(s, d, p) for s, d, p, _ in notes]


def test_drum_tokens_keep_gm_numbers(vocab):
    ids = encode_bar([(0.0, 0.25, 36, 100), (1.0, 0.25, 38, 100)], True, 0.0, 4.0, vocab)
    assert [vocab.tokens[i] for i in ids if vocab.tokens[i].startswith("DRUM=")] == ["DRUM=36", "DRUM=38"]


def _example(vocab, fill=None):
    rng = random.Random(0)
    mk = lambda role, prog, drum=False: TrackTokens(role, prog, drum, [  # noqa: E731
        encode_bar([(b * 4.0 + i, 1.0, rng.randint(-5, 7) if not drum else 36, 80) for i in range(4)], drum, b * 4.0, 4.0, vocab)
        for b in range(4)])
    prefix = condition_prefix(vocab, genre="pop", emotion="happy", root="D", mode="major", tempo_tok="TEMPO=3",
                              meter_tok="METER=4/4", density=2, form_tok="FORM=AB", section_tok="SECTION=A",
                              inrole="lead", outrole="bass", instr="bass")
    return ExampleBuilder(vocab, 16).build(prefix, [mk("lead", 73)], mk("bass", 33), 0, 4, fill)


def test_loss_mask_covers_only_target_body(vocab):
    ex = _example(vocab)
    first_loss = ex.loss.index(1)
    assert vocab.tokens[ex.ids[first_loss - 1]].startswith("PROG=")        # header given, not predicted
    assert all(s == SEG_TARGET for s, l in zip(ex.seg_ids, ex.loss) if l)
    assert vocab.tokens[ex.ids[-1]] == "<EOS>" and ex.loss[-1] == 1
    infill = _example(vocab, fill=(1, 3))
    assert vocab["<FILL>"] in infill.ids and vocab["<FILL_START>"] in infill.ids


def _model(vocab):
    torch.manual_seed(0)
    return HierarchicalComposer(ModelConfig(vocab_size=len(vocab), d_model=32, n_layers=2, n_heads=4, d_ff=64,
                                            bar_layers=1, max_len=512, max_bars=16, dropout=0.0)).eval()


def _tensors(ex):
    return (torch.tensor([ex.ids]), torch.tensor([ex.bar_ids]), torch.tensor([ex.seg_ids]))


def test_decoder_is_causal_and_target_does_not_leak(vocab):
    m = _model(vocab)
    ex = _example(vocab)
    ids, bars, segs = _tensors(ex)
    out = m(ids, bars, segs)
    k = ex.loss.index(1) + 5
    ids2 = ids.clone()
    ids2[0, k:] = vocab["POS=7"]                         # change the *future* target tokens
    out2 = m(ids2, bars, segs)
    assert torch.allclose(out[0, :k], out2[0, :k], atol=1e-5)
    # Bar-level context is computed from context tracks only.
    assert torch.allclose(m.bar_context(ids, bars, segs), m.bar_context(ids2, bars, segs))


def test_kv_cache_matches_full_forward(vocab):
    m = _model(vocab)
    ex = _example(vocab)
    ids, bars, segs = _tensors(ex)
    k = ex.loss.index(1)
    logits, caches, bctx = m.prefill(ids[:, :k], bars[:, :k], segs[:, :k])
    full = m(ids, bars, segs)
    assert torch.allclose(logits[0], full[0, k - 1], atol=1e-4)
    step = m.step(ex.ids[k], ex.bar_ids[k], k, caches, bctx)
    assert torch.allclose(step[0], full[0, k], atol=1e-4)


@pytest.fixture(scope="module")
def trained_smoke(fixture_dataset, tmp_path_factory):
    from midicomposer.model.train import train_from_config
    d = tmp_path_factory.mktemp("smoke")
    prepare_dataset(PrepareConfig(xmidi_root=str(fixture_dataset), out_dir=str(d / "data"), workers=1,
                                  train_frac=0.7, val_frac=0.3))
    info = train_from_config(ROOT / "configs" / "smoke_train.yaml", {
        "data_dir": str(d / "data"), "xmidi_root": str(fixture_dataset), "cache_dir": str(d / "cache"),
        "out": str(d / "smoke.pt")})
    return d / "smoke.pt", info


def test_smoke_training_records_what_it_trained_on(trained_smoke):
    path, info = trained_smoke
    assert path.exists() and info["final_step"] == 12
    assert info["train_files"] <= info["usable_train_files_in_manifest"]
    assert "development-scale" in info["honesty_note"] or "all" in info["honesty_note"]
    assert info["history"] and info["best_val_loss"] > 0


def test_neural_engine_generates_valid_constrained_tracks(trained_smoke):
    path, _ = trained_smoke
    s = tonal_piece(2, "major")
    lead = Score(tracks=[s.tracks[0]], tempos=s.tempos, time_signatures=s.time_signatures)
    r = run_generation(GenerationRequest(mode="midi", input_role="lead", engine="neural", model_path=str(path),
                                         classifier_path=None, n_variations=1, generated_roles=["bass_line", "drums"]),
                       input_bytes=score_to_bytes(lead)).view()
    c = r["candidates"][0]
    assert all(v.startswith("neural:") for v in c["engine_by_role"].values()), c["engine_by_role"]
    assert c["validation"]["passed"], c["validation"]["issues"]
    assert r["manifest"]["model"]["neural_checkpoint"]["honesty_note"]


def test_missing_checkpoint_falls_back_to_rules(tmp_path):
    r = run_generation(GenerationRequest(engine="neural", model_path=str(tmp_path / "none.pt"), classifier_path=None,
                                         bars=8, n_variations=1)).view()
    assert set(r["candidates"][0]["engine_by_role"].values()) == {"rule"}
    assert any("no trained checkpoint" in w for w in r["warnings"])
