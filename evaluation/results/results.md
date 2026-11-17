# MIDI Magic evaluation results

Generated 2026-11-16 18:30 on Windows AMD64, Python 3.12.10 by `python evaluation/run_evaluation.py`. Engines: rule, hybrid. Seed 0, 3 candidates per request; metrics are for the top-ranked candidate (0-1, higher is better).

## Test 1: melodic lines in isolation

Six fixed melodies (`evaluation/test_sets.py`). The app detects key and emotion, then generates chords, bass and drums around the locked melody.

| engine | key root acc. | mode acc. | harmonic | rhythmic | key consist. | structure | total | valid | s/request |
|---|---|---|---|---|---|---|---|---|---|
| rule | 1.000 | 0.833 | 0.872 | 0.885 | 1.000 | 0.842 | 0.764 | 1.000 | 0.223 |
| hybrid | 1.000 | 0.833 | 0.778 | 0.939 | 0.998 | 0.775 | 0.783 | 1.000 | 6.797 |


| melody | engine | detected key | root ok | mode ok | read emotion | harmonic | total |
|---|---|---|---|---|---|---|---|
| ode_to_joy | rule | C major | True | True | fear | 0.800 | 0.746 |
| ode_to_joy | hybrid | C major | True | True | fear | 0.800 | 0.764 |
| twinkle_twinkle | rule | C major | True | True | sad | 0.796 | 0.759 |
| twinkle_twinkle | hybrid | C major | True | True | sad | 0.790 | 0.766 |
| frere_jacques | rule | F major | True | True | sad | 0.861 | 0.819 |
| frere_jacques | hybrid | F major | True | True | sad | 0.869 | 0.827 |
| amazing_grace | rule | G major | True | True | fear | 0.943 | 0.732 |
| amazing_grace | hybrid | G major | True | True | fear | 0.679 | 0.737 |
| drunken_sailor | rule | D mixolydian | True | False | sad | 0.941 | 0.791 |
| drunken_sailor | hybrid | D mixolydian | True | False | sad | 0.936 | 0.847 |
| greensleeves | rule | A minor | True | True | fear | 0.891 | 0.738 |
| greensleeves | hybrid | A minor | True | True | fear | 0.591 | 0.760 |

## Test 2a: text planner accuracy

**Development set** (12 prompts, `PROMPTS` in `test_sets.py`): field accuracy 100%, prompts fully correct 100% (planner: `keywords`). Caution: these prompts were written by the same person who wrote the planner's keyword lists, so this number is optimistic. The held-out set below is the fair measure.

| field | correct | total | accuracy |
|---|---|---|---|
| genre | 7 | 7 | 1.000 |
| emotion | 12 | 12 | 1.000 |
| key | 6 | 6 | 1.000 |
| scale_mode | 3 | 3 | 1.000 |
| tempo | 2 | 2 | 1.000 |
| bars | 2 | 2 | 1.000 |
| density | 1 | 1 | 1.000 |


| prompt | expected | got | all correct |
|---|---|---|---|
| A sad, slow piano and cello piece in D minor, 16 bars | {"emotion": "sad", "key": "D", "scale_mode": "minor", "bars": 16} | {"emotion": "sad", "key": "D", "scale_mode": "minor", "bars": 16} | True |
| Upbeat summer pop song with guitar at 128 bpm | {"genre": "pop", "emotion": "exciting", "tempo": 128} | {"genre": "pop", "emotion": "exciting", "tempo": 128} | True |
| Epic heroic orchestral theme for a movie trailer | {"genre": "classical", "emotion": "magnificent"} | {"genre": "classical", "emotion": "magnificent"} | True |
| A lazy jazz trio tune in Bb for a Sunday morning | {"genre": "jazz", "emotion": "lazy", "key": "A#"} | {"genre": "jazz", "emotion": "lazy", "key": "A#"} | True |
| Angry heavy rock riff in E minor, fast | {"genre": "rock", "emotion": "angry", "key": "E", "scale_mode": "minor"} | {"genre": "rock", "emotion": "angry", "key": "E", "scale_mode": "minor"} | True |
| Warm cozy country ballad in G major | {"genre": "country", "emotion": "warm", "key": "G", "scale_mode": "major"} | {"genre": "country", "emotion": "warm", "key": "G", "scale_mode": "major"} | True |
| Creepy, tense music for a horror scene | {"emotion": "fear"} | {"emotion": "fear"} | True |
| Romantic love song with strings, 90 bpm | {"emotion": "romantic", "tempo": 90} | {"emotion": "romantic", "tempo": 90} | True |
| Calm and peaceful meditation music, minimal | {"emotion": "quiet", "density": 0.2} | {"emotion": "quiet", "density": 0.2} | True |
| A playful, silly cartoon tune in C | {"emotion": "funny", "key": "C"} | {"emotion": "funny", "key": "C"} | True |
| Happy traditional folk dance in A, 32 bars | {"genre": "traditional", "emotion": "happy", "key": "A", "bars": 32} | {"genre": "traditional", "emotion": "happy", "key": "A", "bars": 32} | True |
| Exciting energetic rock anthem with drums | {"genre": "rock", "emotion": "exciting"} | {"genre": "rock", "emotion": "exciting"} | True |

**Held-out set: not run yet.** Ask a teammate who has not read `agents.py` to write 10-15 prompts with their intended settings into `evaluation/heldout_prompts.json` (format in `evaluation/README.md`), then rerun this script.


## Test 2b: does the requested emotion reach the music?

Emotion sweep: every emotion × 2 genres (pop, rock) × 3 seeds, key C, everything else automatic, plus the same requests with no emotion as a baseline (72 generations). (1) The trained emotion classifier (48% test accuracy over 11 emotions) listens to each result; a hit means the requested emotion is in its top 3. (2) Correlation between each emotion's arousal/valence and measurable musical traits.

| condition | top-3 hit rate |
|---|---|
| emotion requested | 0.455 |
| no emotion (baseline) | 0.273 |
| chance | 0.273 |


| relationship | Pearson r |
|---|---|
| arousal~tempo | 0.792 |
| arousal~velocity | 0.999 |
| arousal~notes_per_bar | 0.847 |
| valence~minor | -0.939 |


| emotion | top-3 hit | baseline hit | tempo | velocity | notes/bar | minor share |
|---|---|---|---|---|---|---|
| angry | 0.167 | 0.000 | 133.000 | 98.400 | 38.200 | 1.000 |
| exciting | 0.500 | 0.500 | 133.000 | 98.500 | 37.850 | 0.000 |
| fear | 0.000 | 0.000 | 124.000 | 88.760 | 38.200 | 1.000 |
| funny | 0.000 | 0.000 | 127.500 | 93.070 | 37.850 | 0.000 |
| happy | 0.667 | 0.500 | 126.000 | 90.960 | 37.850 | 0.000 |
| lazy | 0.000 | 0.000 | 104.500 | 64.450 | 24.390 | 0.000 |
| magnificent | 0.000 | 0.000 | 129.500 | 95.170 | 37.850 | 0.000 |
| quiet | 1.000 | 0.000 | 103.000 | 62.350 | 24.390 | 0.000 |
| romantic | 0.667 | 0.500 | 111.500 | 73.600 | 35.570 | 0.000 |
| sad | 1.000 | 0.500 | 108.000 | 68.580 | 24.560 | 1.000 |
| warm | 1.000 | 1.000 | 113.500 | 75.730 | 35.570 | 0.000 |

## Test 2c: chord progressions in isolation

Four fixed progressions as the locked input; the app generates a lead melody and bass line over them.

| engine | key root acc. | harmonic | rhythmic | key consist. | total | valid | s/request |
|---|---|---|---|---|---|---|---|
| rule | 0.750 | 0.828 | 0.816 | 0.962 | 0.757 | 1.000 | 0.168 |
| hybrid | 0.750 | 0.828 | 0.816 | 0.962 | 0.757 | 1.000 | 2.295 |


| progression | engine | detected key | root ok | harmonic | total |
|---|---|---|---|---|---|
| pop_I_V_vi_IV_in_C | rule | C major | True | 0.871 | 0.772 |
| pop_I_V_vi_IV_in_C | hybrid | C major | True | 0.871 | 0.772 |
| jazz_ii_V_I_in_F | rule | F major | True | 0.843 | 0.741 |
| jazz_ii_V_I_in_F | hybrid | F major | True | 0.843 | 0.741 |
| minor_i_VI_III_VII_in_A | rule | C major | False | 0.848 | 0.763 |
| minor_i_VI_III_VII_in_A | hybrid | C major | False | 0.848 | 0.763 |
| blues_12_bar_in_G | rule | G mixolydian | True | 0.752 | 0.753 |
| blues_12_bar_in_G | hybrid | G mixolydian | True | 0.752 | 0.753 |

## Test 3: whole pipeline through the API

Prompt → plan → generate → critic → build on this (+ pad) → stems, for 6 prompts. **Success rate 100%.**

| step | mean seconds |
|---|---|
| plan | 0.007 |
| generate | 0.730 |
| critique | 0.019 |
| build_on | 0.108 |
| generate_more | 0.630 |
| stems | 0.149 |

Critic: 4.0 notes per result; by kind {'instrumentation': 11, 'progression': 7, 'emotion': 6}; by severity {'suggestion': 11, 'info': 12, 'warning': 1}; 4% carry a one-click fix.

The user-study half of test 3 (people using the GUI) is in `evaluation/user_study.md`.
