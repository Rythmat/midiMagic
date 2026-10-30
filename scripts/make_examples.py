"""Regenerate the example MIDI files (and their generation manifests) in examples/.

    python scripts/make_examples.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_engines import anchor_files, lead_excerpt  # noqa: E402
from midicomposer.generation.pipeline import GenerationRequest, run_generation  # noqa: E402
from midicomposer.midi_io import save_midi, score_to_bytes  # noqa: E402

OUT = Path("examples")


def keep(state, name: str) -> None:
    best = state.candidates[0]
    (OUT / f"{name}.mid").write_bytes(state.midi_bytes(0))
    (OUT / f"{name}.manifest.json").write_text(json.dumps(state.manifest, indent=2, default=str))
    s = best.scores
    print(f"{name}: total {s['total']:.3f} quality {s['quality']:.3f} valid {best.validation['passed']} "
          f"engines {sorted(set(best.engine_used.values()))}")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    meta = [
        ("metadata_pop_happy_D_major", dict(genre="pop", emotion="happy", key="D", bars=32, seed=1)),
        ("metadata_jazz_lazy_Bb_walking_bass", dict(genre="jazz", emotion="lazy", key="Bb", bars=32, seed=2,
                                                    generated_roles=["lead_melody", "chord_bed", "walking_bass", "drums"])),
        ("metadata_classical_sad_A_minor_waltz", dict(genre="classical", emotion="sad", key="A minor", meter="3/4",
                                                      bars=24, seed=3)),
        ("metadata_rock_angry_E_minor_neural", dict(genre="rock", emotion="angry", key="E minor", bars=16, seed=4,
                                                    engine="neural",
                                                    generated_roles=["lead_melody", "chord_bed", "bass_line", "drums"])),
        ("metadata_country_warm_G_hybrid", dict(genre="country", emotion="warm", key="G", bars=16, seed=5,
                                                engine="hybrid", n_variations=4)),
    ]
    for name, kw in meta:
        keep(run_generation(GenerationRequest(**{"n_variations": 3, **kw})), name)

    rec = anchor_files(Path("data/xmidi"), 1, seed=7)[0]
    anchor = lead_excerpt(Path("XMIDI_Dataset") / rec["path"], rec["lead"])
    save_midi(anchor, OUT / "example_input_melody.mid")
    (OUT / "example_input_melody.source.txt").write_text(
        f"16-bar lead-track excerpt from {rec['path']} (XMIDI test split; genre={rec['genre']}, emotion={rec['emotion']}).\n")
    data = score_to_bytes(anchor)
    keep(run_generation(GenerationRequest(mode="midi", input_role="lead", genre=rec["genre"], emotion=rec["emotion"],
                                          generated_roles=["chord_bed", "bass_line", "counterline", "drums"],
                                          n_variations=4, seed=11, engine="hybrid"),
                        input_bytes=data, input_name="example_input_melody.mid"),
         "accompaniment_for_example_melody")
    keep(run_generation(GenerationRequest(mode="midi", input_role="lead", genre=rec["genre"], emotion=rec["emotion"],
                                          generated_roles=["full_composition"], target_key="F", n_variations=3, seed=12),
                        input_bytes=data, input_name="example_input_melody.mid"),
         "full_composition_for_example_melody_transposed_to_F")


if __name__ == "__main__":
    main()
