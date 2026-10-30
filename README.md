# MIDI Composer

Conditional symbolic-music composition trained on [XMIDI](https://github.com/xmusic-project/XMIDI_Dataset), with a
reversible **root-relative** MIDI representation.

Two generation modes:

1. **From MIDI** — upload a MIDI file, declare its role (lead, bass, highs, lows, chordal, full mix), choose what to
   generate (full composition, chord bed, arpeggio, bass line, walking bass, octave doubles, counterline, harmony,
   drums, pad, high/low texture). The input is analysed (root, key/mode, tempo map, meter, roles, chords,
   sections) and the generated parts are built around it, keeping its key and rhythmic identity. The input track
   is locked and never modified (except a uniform transposition when a target key is chosen).
2. **From metadata** — genre, emotion, key/root, mode, tempo, meter, length, instrumentation, density and form
   produce a complete multi-track composition (melody, harmony, bass, drums, textures).

Every request produces several candidates (different seeds), which are validated, repaired, scored and ranked.
Output: MIDI files plus a JSON **generation manifest** for each job.

Nothing about a request is stored. Uploads and results live in server memory and are discarded when the page closes
or refreshes (see [Transient storage](#transient-storage)), so download what you want to keep.

---

## What has actually been run (status)

| Item | Status in this repository |
|---|---|
| XMIDI preprocessing | Run on **all 108,023 files** (240 s, 20 workers). 96,395 usable; 9,301 duplicates detected. Report: [`reports/dataset_report.md`](reports/dataset_report.md) |
| Genre/emotion classifiers | Trained on the full train split (77,086 files). Held-out **test accuracy: genre 52.8% (majority baseline 23.6%), emotion 48.1% (baseline 20.6%)**. [`reports/classifier_report.json`](reports/classifier_report.json) |
| Neural track generator | **Development-scale only**: 6.8M parameters, trained on **8,000 of 77,086** usable training files for 4,000 steps (7.6 min, one RTX 5070 laptop GPU). Validation loss 5.26 → 1.234 and still falling, so it is under-trained. [`reports/composer_dev_training.json`](reports/composer_dev_training.json). It has **not** been trained on all of XMIDI. |
| Rule-based generator | Complete, deterministic, the default engine. |
| Tests | 153 Python tests + 13 Node UI-state tests, all passing. |

Engine comparison (best of 2 candidates, mean over prompts; anchor prompts are 16-bar melodies from XMIDI
**test-split** files): [`reports/engine_comparison.md`](reports/engine_comparison.md)

| | rule | neural (dev) | hybrid |
|---|---:|---:|---:|
| Metadata prompts: harmonic compatibility | 0.903 | 0.647 | 0.896 |
| Metadata prompts: genre score | 0.857 | **0.962** | 0.853 |
| Anchor prompts: harmonic compatibility | **0.871** | 0.654 | **0.871** |
| Anchor prompts: total (ranking score) | 0.794 | 0.838 | **0.869** |
| Valid output rate | 100% | 100% | 100% |
| Seconds per request | 0.1 | 6–7 | 3–4 |

In short: the rules are harmonically stronger; the dev neural model sounds more "XMIDI-like" to the genre
classifier; the hybrid engine (mixing both candidate types and letting ranking choose) scores best overall. The
neural numbers should improve substantially with the full training configuration (`configs/full_train.yaml`),
which has not been run.

---

## Setup

Python 3.10+ (developed on 3.14, Windows 11). Node 18+ is only needed for the UI unit tests.

```powershell
cd project
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training install the CUDA build of PyTorch from <https://pytorch.org> first. (In this workspace the venv was
created with `--system-site-packages` to reuse an existing CUDA PyTorch install.)

The XMIDI folder is configurable: pass `--xmidi-root PATH` or set `XMIDI_ROOT`
(`$env:XMIDI_ROOT="C:\data\XMIDI_Dataset"` in PowerShell, `export XMIDI_ROOT=...` in bash).

## Exact commands

All commands run from `project/` with the venv active.

**1. Preprocess XMIDI** (manifest, hashes, duplicates, splits, features, report):
```powershell
python -m midicomposer.cli prepare --xmidi-root XMIDI_Dataset --out data/xmidi --workers 20
# fast development subset (deterministic 2,000 files; labels are still counted over all files):
python -m midicomposer.cli prepare --xmidi-root XMIDI_Dataset --out data/xmidi_dev --max-files 2000
```

**2. Train the genre/emotion classifiers** used in ranking (about 10 s on GPU):
```powershell
python -m midicomposer.cli train-classifiers --data data/xmidi --out checkpoints/classifiers.pt
```

**3. Run a development training job** for the neural track generator:
```powershell
python -m midicomposer.cli train --config configs/dev_train.yaml
# quicker smoke run:
python -m midicomposer.cli train --config configs/dev_train.yaml --set max_train_files=500 max_steps=300 out=checkpoints/composer_quick.pt
```

**4. Start the API** (also serves the web UI at <http://127.0.0.1:8000/>):
```powershell
python -m midicomposer.cli serve --port 8000
```

**5. Start the frontend separately** (optional; any static server works):
```powershell
python -m http.server 5173 --directory web
# open http://127.0.0.1:5173/?api=http://127.0.0.1:8000
```

**6. Generate a test MIDI file:**
```powershell
# Mode 2: metadata only
python -m midicomposer.cli generate --genre pop --emotion happy --key D --bars 16 --out examples/test_pop.mid
# Mode 1: around an input melody, transposed to F, with the hybrid engine
python -m midicomposer.cli generate --input examples/example_input_melody.mid --input-role lead `
    --roles chord_bed,bass_line,counterline,drums --target-key F --engine hybrid --out examples/test_accompaniment.mid
# Analysis only
python -m midicomposer.cli analyze examples/example_input_melody.mid
```
Through the API (with the server running):
```bash
curl -F "file=@examples/example_input_melody.mid" -F "input_role=lead" http://127.0.0.1:8000/analyze-midi
curl -H "Content-Type: application/json" -d '{"mode":"midi","upload_id":"<id>","input_role":"lead","generated_roles":["chord_bed","bass_line","drums"]}' http://127.0.0.1:8000/generate
curl http://127.0.0.1:8000/generation/<job_id>
curl -o out.mid "http://127.0.0.1:8000/generation/<job_id>/download?candidate=0"
```
Without a session (as here), uploads and jobs are kept for 10 minutes after their last use.

**7. Tests:**
```powershell
python -m pytest                              # 153 tests, ~6 s, uses tests/fixtures only (no XMIDI needed)
node --test web/tests/ui-state.test.mjs       # UI button-state and error-handling logic
```

**8. Evaluation and examples:**
```powershell
python scripts/evaluate_engines.py --n-anchors 8   # writes reports/engine_comparison.{md,json}
python scripts/make_examples.py                    # regenerates examples/
```

---

## Architecture

```
MIDI file ──► midi_io (beat-based Score) ──► analyzer ─┬─ root.py: tonic, mode, confidence
                                                      ├─ roles.py: per-track roles
                                                      ├─ harmony.py: chords (Viterbi)
                                                      └─ sections, rhythm profile, energy curve
                         representation.py: normalize (pitch - (60 + root_pc)) ◄──┘
                                      │
metadata ──► planner.py + styles.py ──┤  form, sections, progressions, energy, registers, instruments
                                      ▼
                   generation/ engines (per role, root-relative)
                     ├─ rule_based.py   deterministic fallback for every role
                     └─ neural.py       HierarchicalComposer checkpoint, grammar-constrained decoding
                                      ▼
            denormalize (rel + 60 + target_pc) ─► validation.py (check + repair) ─► ranking.py ─► MIDI + manifest
```

| Module | Responsibility |
|---|---|
| `midicomposer/midi_io.py` | Parse/write MIDI via `mido`; beats not seconds; tempo map, time/key signatures, tick resolution preserved; corrupted/empty input rejected with clear errors |
| `midicomposer/root.py` | Root/mode detection with evidence, confidence, ambiguity flags, overrides |
| `midicomposer/representation.py` | Reversible root-relative normalization, denormalization, transposition, instrument-range fitting |
| `midicomposer/roles.py` | Role vocabulary, heuristic role detection, `full_composition` expansion |
| `midicomposer/harmony.py` | Chord inference for chordal parts, bass lines or bare melodies |
| `midicomposer/analyzer.py` | Full analysis: tempo/meter, roles, ranges, density, rhythm, chords, sections, local keys, energy |
| `midicomposer/styles.py`, `planner.py` | Genre/emotion profiles (any label maps by keyword), genre-aware forms, progressions, registers, instrumentation |
| `midicomposer/generation/rule_based.py` | Lead melody (motif-based), chord bed (voice-led comping), pad, arpeggio, bass line, walking bass, counterline (consonance + contrary motion), harmony, octave doubles, textures, drums |
| `midicomposer/model/` | Vocabulary, tokenizer, hierarchical transformer, masked-task data sampler, training loop |
| `midicomposer/generation/neural.py` | Checkpoint loading and constrained sampling |
| `midicomposer/validation.py` | Pre-export checks and light automatic repair |
| `midicomposer/classifiers.py`, `features.py`, `ranking.py` | Feature classifiers and candidate scoring |
| `midicomposer/generation/pipeline.py` | End-to-end generation in memory: candidates, regeneration of tracks/sections, MIDI bytes, manifests |
| `midicomposer/dataset/` | XMIDI discovery, filename parsing, manifest, dedup, splits, report |
| `midicomposer/api/` | FastAPI app, in-process async jobs, in-memory session store |
| `web/` | Dependency-free UI (`ui-state.js` holds all button/validation logic) |

### Root-relative representation

See **[`docs/representation.md`](docs/representation.md)** for the full explanation. Summary:
`root_reference_midi = 60 + root_pitch_class`; `relative_pitch = midi_pitch - root_reference_midi`
(C4→0, D4→2, B3→-1 in C; D4→0, F#4→4, C4→-2 in D). The root is a sequence-level condition (`<ROOT=D> <MODE=major>`),
not repeated per note. Output: `midi_pitch = relative_pitch + 60 + target_pc` (+ an optional octave offset to keep
transpositions within a tritone). Out-of-range notes are moved by octaves, never clipped, and reported.
Percussion is never normalized. Ambiguous roots are flagged with candidates; undeterminable roots require an override.

### Dataset pipeline

* Recursive discovery of `.mid/.midi` (any case); labels parsed **dynamically** from
  `XMIDI_<Emotion>_<Genre>_<ID>.midi`. Underscore separates fields, `-`/space join words inside a label; names with
  more than two label fields (e.g. `very_happy_hip_hop`) are split using a vocabulary learned from unambiguous names;
  unresolvable splits are flagged `ambiguous`. Raw spellings are kept next to normalized (`lower_snake_case`) labels.
  XMIDI itself: 11 emotions × 6 genres, all 108,023 names parsed.
* Per file: SHA-256, a note-content hash and a transposition-invariant hash; corrupted/empty files are recorded, not
  dropped silently.
* Duplicates (any of the three hashes, transitively) form one group; the group is assigned to one split by a salted
  hash of its canonical file, so duplicates and transposed copies never leak across splits. Later copies are
  excluded from training. XMIDI has no song/artist metadata, so this is the strongest leakage control available.
* 80/10/10 deterministic split; `--max-files` selects a deterministic, label-spread subset.
* `manifest.jsonl` (full records), `manifest.csv`, `features.npy`, `label_vocab.json`, `dataset_report.{md,json}`.

Manifest fields: path, filename, file_id, raw/normalized emotion and genre, label status, size, hashes, status/error,
duration (s), total beats, bars, tempo, tempo changes, meter, meter changes, ticks per beat, root name/pc/reference,
mode, confidence, method, ambiguity flags, top candidates, override, track list (name, program, instrument, channel,
drum flag, note count, detected role, pitch range), duplicate group/kind, split, usable flag.

Observations worth knowing: XMIDI has **no key-signature events**, every file is **4/4**, and 88% of files carry the
default tempo (~119 BPM), so tempo and meter are weak conditioning signals in this dataset. Instrumentation is a small
canonical set (acoustic bass, steel guitar, piano, strings, flute, trumpet, tenor sax, drums, ...).

### Neural model

* **Tokens.** Condition prefix `GENRE= EMOTION= ROOT= MODE= TEMPO= METER= DENSITY= FORM= SECTION= INROLE= OUTROLE=
  INSTR=`, then context tracks, then the target track. Notes: `POS` (12 steps per quarter), `PITCH` (root-relative,
  -48..47) or `DRUM` (GM number), `DUR`, `VEL`. Genre/emotion tokens come from the discovered labels and are saved in
  the checkpoint.
* **Training tasks** (sampled per example from complete XMIDI pieces): masked-track reconstruction with random
  context subsets (including no context, i.e. metadata-only), masked-section infilling (`<FILL>` placeholders), and
  continuation. This covers melody-conditioned accompaniment, bass-conditioned harmony and partial-track generation.
* **HierarchicalComposer.** A bidirectional bar-level encoder summarises what the context tracks play in every bar;
  a causal token decoder (KV-cached) receives those bar vectors plus bar/segment/position embeddings. Bar summaries use
  only context tokens, so no target information leaks (tested).
* **Decoding** is grammar-constrained (exact bar count, increasing positions, role register, drum vs. pitch tokens,
  one onset per note for monophonic roles). In metadata mode the planner's chord progression is supplied as a hidden
  scaffold track so independently generated parts share one harmony.
* **Checkpoint interface.** `load_neural_generator(path)` returns `None` when no checkpoint exists, and the pipeline
  falls back to the rule engine with a warning. Each checkpoint stores its config, vocabulary, training history,
  file counts and an `honesty_note` describing exactly what it was trained on.

### Ranking and validation

Candidates are scored on **quality** (validity, key consistency, harmonic compatibility with the anchor or between
parts, rhythmic compatibility with the anchor, structural coherence, instrument range, density), **genre** and
**emotion** (each = 50% classifier probability, 50% style heuristics; heuristics only when no classifier is
available), and **novelty** (diversity versus the other candidates, and not copying the anchor). Invalid candidates are
penalised. The classifiers are trained on the train split and evaluated on held-out data; on generated music their
scores are out-of-distribution estimates, which is why they are never the sole criterion.

Before export every candidate is checked for: durations (≥ 1 tick), note-on/off pairing in the written bytes,
monophonic overlaps, instrument ranges, key/chord consistency, meter and grid, programs/channels (drums on channel 10),
locked/anchor tracks unchanged, requested roles present, and a successful re-parse. Repairs (unlocked tracks only):
onset quantization, minimum durations, velocity clamping, duplicate removal, overlap trimming, octave range fitting,
channel assignment.

---

## API

| Method & path | Purpose |
|---|---|
| `GET /health` | Liveness, plus how many sessions, uploads and jobs are in memory (counts only) |
| `GET /options` | Genres/emotions (from `label_vocab.json`), keys, modes, roles, forms, presets, engine availability, model/classifier status |
| `POST /session` | Open a session → `session_id`, `heartbeat_seconds`, `ttl_seconds`. Send the id as an `X-Session-Id` header on later requests |
| `POST /session/{id}/heartbeat` | Keep a session alive; 410 once it has ended |
| `POST /session/{id}/close` | Dispose of the session's uploads and jobs now |
| `POST /analyze-midi` | Multipart `file` (+ optional `root_override`, `input_role`) → `upload_id` + analysis. 400 empty, 411 no Content-Length, 413 too large, 422 corrupted/not MIDI |
| `POST /generate` | JSON request (see `GenerateBody` in `api/app.py`) → 202 + `job_id` (async) |
| `GET /generation/{job_id}` | Status, progress, and the full result when done |
| `GET /generation/{job_id}/download?candidate=N` | MIDI file (409 while running, 404 once its session has ended) |
| `GET /generation/{job_id}/manifest` | Generation manifest |
| `POST /regenerate-track` | `{job_id, candidate, track_id, seed?}` → new job; other tracks unchanged |
| `POST /regenerate-section` | `{job_id, candidate, section_index \| start_bar+end_bar, track_ids?, seed?}` → new job; locked tracks unchanged |
| `POST /generation/{job_id}/lock` | `{candidate, track_id, locked}`; input tracks are always locked |
| `POST /prepare-dataset` | `{xmidi_root?, out_dir, max_files}` → background preprocessing job (`GET /jobs/{id}`) |

A request whose `X-Session-Id` has ended gets **410**. Interactive docs: <http://127.0.0.1:8000/docs>.

## Web UI

Two tabs (From MIDI / From metadata). Uploading a file triggers analysis and shows the detected root, root confidence,
key/mode, method, tempo, meter, bars, every track with its detected role (editable) and ambiguity warnings with
candidate roots. Generate, Analyze, Download, Play, Regenerate-track and Regenerate-section are enabled only when their
prerequisites hold; when disabled, the reasons are listed under the button and in its tooltip. Results show ranked
candidates with score breakdowns, a piano roll with section boundaries, per-track lock toggles, a simple in-browser
preview synth, warnings, validation issues and applied repairs. `?mode=metadata` opens the metadata tab.

## Transient storage

Uploads and generation results are held only in the server process's memory. They are never written to disk, including
as temp files: the upload parser is configured to keep every accepted file in memory, and larger uploads are refused
before parsing. `tests/test_sessions.py` enforces this with an audit hook that sees every file the process opens.

Each page load opens its own session, whose id exists only in that page, so a refresh is a new session and each tab is
independent.

| What happens | When the data goes |
|---|---|
| Page closed or refreshed | Immediately (the page tells the server as it unloads) |
| Browser crash, lost network, laptop asleep | After `session_ttl_seconds` without a heartbeat (default 3 min, checked every 15 s) |
| Server stopped or restarted | Immediately, since memory is gone. Open pages say so and reconnect |
| `curl`/scripts with no session | `anonymous_ttl_seconds` after last use (default 10 min) |
| Long session | Only the newest `max_jobs_per_session` (20) jobs and `max_uploads_per_session` (8) uploads are kept |

A job whose session ends stops before its next step and discards its result. If a page lost its session while asleep,
it says so on waking, disables Download and starts a new session. Settings are in `configs/default.yaml` (or
`MIDICOMPOSER_<SETTING>` environment variables).

The server must run as a single process, since workers don't share memory. Files appear on disk only when you ask for
them: browser downloads, `generate --out` (the best candidate and its manifest), `prepare`, `train`, and the scripts.

## Generation manifest

`GET /generation/{job_id}/manifest` (while the job's session lives), next to the file written by `generate --out`, and
next to every example in `examples/`. It records: job id/parent, timestamps,
**input conditions** (the full request), input file name + SHA-256, **root** (input root, confidence, method,
ambiguity, output root, output reference MIDI, transposition), **model version** (package, engine requested, rule
version, neural checkpoint description with its honesty note, classifier test metrics), the full plan (form, sections,
chords, programs, registers, energy curve), and per candidate: rank, **random seed**, scores, engine used per role,
**output roles** (track, role, program, instrument, channel, source, locked, note count) and the validation report.

## Examples

`examples/` (regenerate with `python scripts/make_examples.py`; each `.mid` has a `.manifest.json`):

| File | What it shows |
|---|---|
| `metadata_pop_happy_D_major.mid` | Mode 2, 32 bars, intro/verse/chorus form |
| `metadata_jazz_lazy_Bb_walking_bass.mid` | Walking bass, swing, ii-V-I progressions |
| `metadata_classical_sad_A_minor_waltz.mid` | 3/4 meter, ternary form, no drums |
| `metadata_rock_angry_E_minor_neural.mid` | Neural engine (dev checkpoint) |
| `metadata_country_warm_G_hybrid.mid` | Hybrid engine, 4 candidates ranked |
| `example_input_melody.mid` | A 16-bar melody from an XMIDI test-split file, to try Mode 1 |
| `accompaniment_for_example_melody.mid` | Mode 1: chord bed, bass, counterline, drums around it |
| `full_composition_for_example_melody_transposed_to_F.mid` | Mode 1 + target key (A major → F major) |

## Limitations

* The neural model is development-scale (see status). Its harmonic compatibility is clearly below the rule engine's;
  use `rule` or `hybrid` for best results until a full training run is done.
* Genre/emotion classifiers are modest (≈50% accuracy over 6 genres / 11 emotions) and learn XMIDI's statistics; XMIDI
  genres share almost the same instrumentation, so genre cues are mostly rhythmic/harmonic.
* Mode labels beyond major/minor (dorian, mixolydian, lydian, phrygian) come from a characteristic-degree heuristic.
* One global root per piece; per-section local keys are detected and reported but generation follows the global key.
* Chord inference is one chord per half bar at most.
* Results exist only until the page closes or refreshes, so one stray refresh loses anything not downloaded. Saved
  links to a job and reopening a job after a server restart are not possible.
* Jobs run in an in-process thread pool on a single server process. There is no authentication; job and upload ids are
  the only access control, so run the server locally.
* The UI preview synth is a rough sine/triangle sketch, not a rendering of the GM instruments.
