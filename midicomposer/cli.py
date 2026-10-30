"""Command-line entry point.

    python -m midicomposer.cli prepare   --xmidi-root XMIDI_Dataset --out data/xmidi [--max-files 2000]
    python -m midicomposer.cli train-classifiers --data data/xmidi
    python -m midicomposer.cli train     --config configs/dev_train.yaml
    python -m midicomposer.cli analyze   song.mid [--root "D minor"]
    python -m midicomposer.cli generate  --genre pop --emotion happy --key D --bars 16 --out out.mid
    python -m midicomposer.cli generate  --input melody.mid --input-role lead --roles chord_bed,bass_line,drums --out out.mid
    python -m midicomposer.cli serve     [--host 127.0.0.1 --port 8000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _progress(i: int, n: int) -> None:
    print(f"  processed {i}/{n}", flush=True)


def cmd_prepare(a: argparse.Namespace) -> None:
    from .dataset.prepare import PrepareConfig, prepare_dataset, resolve_xmidi_root
    hints = json.loads(Path(a.label_hints).read_text()) if a.label_hints else None
    cfg = PrepareConfig(xmidi_root=str(resolve_xmidi_root(a.xmidi_root)), out_dir=a.out, max_files=a.max_files,
                        workers=a.workers, split_salt=a.split_salt, min_root_confidence=a.min_root_confidence,
                        root_overrides=a.root_overrides, label_hints=hints)
    print(f"Preparing XMIDI from {cfg.xmidi_root} -> {cfg.out_dir} (max_files={cfg.max_files}, workers={cfg.workers})")
    summary = prepare_dataset(cfg, progress=_progress)
    print(json.dumps(summary, indent=2))


def cmd_train_classifiers(a: argparse.Namespace) -> None:
    from .classifiers import train_classifiers
    res = train_classifiers(a.data, a.out, epochs=a.epochs, device=a.device)
    print(json.dumps(res, indent=2))


def cmd_train(a: argparse.Namespace) -> None:
    from .model.train import train_from_config
    overrides = {}
    for kv in a.set or []:
        k, v = kv.split("=", 1)
        try:
            overrides[k] = json.loads(v)
        except json.JSONDecodeError:
            overrides[k] = v
    res = train_from_config(a.config, overrides)
    print(json.dumps(res, indent=2))


def cmd_analyze(a: argparse.Namespace) -> None:
    from .analyzer import analyze_score
    from .midi_io import load_midi
    res = analyze_score(load_midi(a.file), root_override=a.root, input_role=a.input_role)
    if not a.full:
        res.pop("root_names", None)
        res["chords"] = [c["name"] for c in res["chords"]]
    print(json.dumps(res, indent=2))


def cmd_generate(a: argparse.Namespace) -> None:
    from .generation.pipeline import GenerationRequest, run_generation
    req = GenerationRequest(
        mode="midi" if a.input else "metadata", genre=a.genre, emotion=a.emotion, key=a.key, scale_mode=a.mode,
        tempo=a.tempo, meter=a.meter, bars=a.bars, density=a.density, instrumentation=a.instrumentation,
        form=a.form, input_role=a.input_role,
        generated_roles=[r.strip() for r in a.roles.split(",")] if a.roles else ["full_composition"],
        target_key=a.target_key, root_override=a.root, n_variations=a.variations, seed=a.seed, engine=a.engine)
    input_bytes = Path(a.input).read_bytes() if a.input else None
    state = run_generation(req, input_bytes=input_bytes, input_name=Path(a.input).name if a.input else None)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(state.midi_bytes(0))
    man = out.with_suffix(".manifest.json")
    man.write_text(json.dumps(state.manifest, indent=2, default=str))
    print(f"Wrote {out} (score {state.candidates[0].scores['total']:.3f}) and {man}")
    for w in state.warnings:
        print("warning:", w)


def cmd_serve(a: argparse.Namespace) -> None:
    import uvicorn
    uvicorn.run("midicomposer.api.app:app", host=a.host, port=a.port, reload=False)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="midicomposer")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare", help="build manifest/splits/report from an XMIDI folder")
    s.add_argument("--xmidi-root", default=None, help="XMIDI folder (or env XMIDI_ROOT)")
    s.add_argument("--out", default="data/xmidi")
    s.add_argument("--max-files", type=int, default=None, help="dev mode: deterministic subset size")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--split-salt", default="xmidi-v1")
    s.add_argument("--min-root-confidence", type=float, default=0.4)
    s.add_argument("--root-overrides", default=None, help="JSON or CSV (file,key) of manual roots")
    s.add_argument("--label-hints", default=None, help="JSON {emotions:[...], genres:[...]} for multi-word labels")
    s.set_defaults(fn=cmd_prepare)

    s = sub.add_parser("train-classifiers", help="train genre/emotion feature classifiers")
    s.add_argument("--data", default="data/xmidi")
    s.add_argument("--out", default="checkpoints/classifiers.pt")
    s.add_argument("--epochs", type=int, default=60)
    s.add_argument("--device", default="auto")
    s.set_defaults(fn=cmd_train_classifiers)

    s = sub.add_parser("train", help="train the hierarchical track generator")
    s.add_argument("--config", default="configs/dev_train.yaml")
    s.add_argument("--set", nargs="*", help="override config values, e.g. max_steps=200")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("analyze", help="analyze a MIDI file")
    s.add_argument("file")
    s.add_argument("--root", default=None, help="manual root override, e.g. 'D minor'")
    s.add_argument("--input-role", default=None)
    s.add_argument("--full", action="store_true")
    s.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("generate", help="generate MIDI (metadata-only or around an input MIDI)")
    s.add_argument("--input", default=None, help="anchor MIDI (mode 1); omit for metadata-only (mode 2)")
    s.add_argument("--input-role", default="lead")
    s.add_argument("--roles", default=None, help="comma-separated generated roles (default full_composition)")
    s.add_argument("--genre", default="pop")
    s.add_argument("--emotion", default="happy")
    s.add_argument("--key", default="C")
    s.add_argument("--mode", default=None)
    s.add_argument("--target-key", default=None)
    s.add_argument("--root", default=None, help="manual root override for the input MIDI")
    s.add_argument("--tempo", type=float, default=None)
    s.add_argument("--meter", default="4/4")
    s.add_argument("--bars", type=int, default=None)
    s.add_argument("--density", type=float, default=0.5)
    s.add_argument("--instrumentation", default=None)
    s.add_argument("--form", default=None)
    s.add_argument("--variations", type=int, default=3)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--engine", default="rule", choices=["rule", "neural", "hybrid"])
    s.add_argument("--out", default="examples/generated.mid", help="the best candidate; its manifest goes next to it")
    s.set_defaults(fn=cmd_generate)

    s = sub.add_parser("serve", help="start the API + web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
