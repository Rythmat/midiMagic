# Evaluation

Implements the evaluation plan from the proposal (slide 7): test features in isolation with fixed melodic lines,
then with fixed prompts and chords, then the whole pipeline with the GUI. (Lyrics were out of scope.)

| File | What |
|---|---|
| `test_sets.py` | The fixed test sets: 6 melodies, 12 prompts with hand-labelled settings, 4 chord progressions |
| `run_evaluation.py` | Runs tests 1–3 and writes `results/results.md` + `results/results.json`; also exports the test inputs as MIDI to `data/` |
| `user_study.md` | The user-study protocol and questionnaire (the human half of test 3) |
| `user_study_results.csv` | Empty sheet to fill in, one row per participant |
| `analyze_user_study.py` | Turns the sheet into `results/user_study.md` (SUS score, task completion, question means) |
| `heldout_prompts.json` | *(to be written by a teammate)* held-out prompts for a fair planner score |

```bash
python evaluation/run_evaluation.py                  # ~2 min on a laptop CPU (rule + hybrid engines)
python evaluation/analyze_user_study.py              # after the user study
```

Results are deterministic (fixed seeds), except timings.

## Held-out prompts

The 12 development prompts were written alongside the keyword planner, so its 100% score on them is optimistic.
For a fair number, someone who has **not** read `midicomposer/agents.py` writes 10–15 prompts and the settings a
musician would expect from each, using only these fields and values:

```json
[
  {"prompt": "A gloomy piano waltz in F minor", "expected": {"emotion": "sad", "key": "F", "scale_mode": "minor"}},
  {"prompt": "Fast punk song, 180 bpm", "expected": {"genre": "rock", "tempo": 180}}
]
```

`genre`: classical, country, jazz, pop, rock, traditional · `emotion`: angry, exciting, fear, funny, happy, lazy,
magnificent, quiet, romantic, sad, warm · `key`: C C# D D# E F F# G G# A A# B (sharps; Bb = A#) · `scale_mode`:
major, minor · `tempo`, `bars`: numbers. Only list what the prompt actually says. Rerun the evaluation afterwards.
When the LLM planner replaces the keyword one, the same held-out set compares the two.

## Findings (run of 2026-11-16, see `results/results.md` for every number)

**Test 1, melodies.**
* Key root detected correctly for 6 of 6 melodies. The mode was wrong once: Drunken Sailor (D dorian) was read as
  D mixolydian.
* Every output passed validation and the melody was kept unchanged.
* Rule and hybrid engines score about the same (harmonic compatibility 0.87, total 0.83). The rule engine takes
  0.3 s per request and the hybrid 6 s on a CPU, so the rule engine is the better default.
* **Emotion reader on a bare melody: found and fixed.**
  * The first run showed the classifier reading every melody as "fear" or "sad", happy tunes included (0 of 6
    plausible). It was trained on full multi-track XMIDI songs, so a single melody line is out of distribution. The
    same happened for every chord-only input.
  * Single-part inputs now use a mode/energy reading (major/minor, tempo, loudness, note density). Full
    arrangements still use the classifier.
  * A reading only drives generation when it is reliable: the key is clear, or the classifier is at least 40% sure.
    Otherwise the user's settings stay and the UI says why.
  * Result: 6 of 6 plausible readings (Ode to Joy → happy, Greensleeves → sad). 5 of 6 were applied; Drunken
    Sailor's key is ambiguous, so it wasn't.
  * Caveat: the "plausible moods" (`MELODY_MOODS`) were labelled by the developer. A teammate should re-label them
    without looking at the readings before this number goes in the report.
  * Generation also improved once the right mood drove it: the total went from 0.76 to 0.83 with the rule engine.

**Test 2a, planner.**
* 100% field accuracy on the development prompts. This is optimistic (see above); the held-out run is still to do.

**Test 2b, emotion → music.**
* The requested emotion clearly shapes the music:
  * arousal correlates with tempo (r = 0.79), loudness (r = 1.00) and note density (r = 0.85);
  * negative-valence emotions get a minor key (r = −0.94 between valence and minor).
* The independent emotion classifier hears the requested emotion in its top 3 in **45%** of generations, against
  27% with no emotion requested (which equals chance).
* It is strong for sad, quiet and warm, and weak for fear, funny, lazy and magnificent. Those emotions share
  tempo/loudness/mode settings with others, so the generator needs more emotion-specific traits (rhythm, register,
  articulation) to tell them apart.

**Test 2c, chords.**
* Melody and bass over fixed progressions: harmonic compatibility 0.83 and every output valid.
* The 12-bar blues is lowest (0.75), because of its dominant-7th chords.
* The A-minor loop (i–VI–III–VII) was detected as **C major**, its relative major. This is a known ambiguity of key
  finding on chord loops; the manual root override in the UI handles it.

**Test 3, whole pipeline.**
* 6 of 6 prompt → plan → generate → critic → build on this → stems runs succeeded.
* Every step takes under a second on a laptop with the rule engine. The free Render server is slower, and its
  first request after sleeping takes about 60 s.
* The critic gives about 4 notes per result, mostly instrumentation and progression observations.
* Only 4% of its notes carry a one-click fix, because regenerating can't change an instrument. The LLM critic could
  add actions that change settings.
* The user study is still to run (`user_study.md`).
