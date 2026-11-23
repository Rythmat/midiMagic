# MIDI Magic evaluation results

Generated 2026-11-21 15:02 on Windows AMD64, Python 3.13.5 by `python evaluation/run_evaluation.py`. Engines: rule, hybrid. Seed 0, 3 candidates per request; metrics are for the top-ranked candidate (0-1, higher is better).

## Test 1: melodic lines in isolation

Six fixed melodies (`evaluation/test_sets.py`). The app detects key and emotion, then generates chords, bass and drums around the locked melody.

| engine | key root acc. | mode acc. | harmonic | rhythmic | key consist. | structure | total | valid | s/request |
|---|---|---|---|---|---|---|---|---|---|
| rule | 1.000 | 0.833 | 0.871 | 0.940 | 0.985 | 0.847 | 0.826 | 1.000 | 0.105 |
| hybrid | 1.000 | 0.833 | 0.871 | 0.941 | 0.986 | 0.846 | 0.830 | 1.000 | 2.997 |


| melody | engine | detected key | root ok | mode ok | read emotion | harmonic | total |
|---|---|---|---|---|---|---|---|
| ode_to_joy | rule | C major | True | True | happy | 0.791 | 0.780 |
| ode_to_joy | hybrid | C major | True | True | happy | 0.791 | 0.789 |
| twinkle_twinkle | rule | C major | True | True | warm | 0.785 | 0.825 |
| twinkle_twinkle | hybrid | C major | True | True | warm | 0.787 | 0.828 |
| frere_jacques | rule | F major | True | True | warm | 0.884 | 0.822 |
| frere_jacques | hybrid | F major | True | True | warm | 0.884 | 0.829 |
| amazing_grace | rule | G major | True | True | romantic | 0.942 | 0.825 |
| amazing_grace | hybrid | G major | True | True | romantic | 0.942 | 0.825 |
| drunken_sailor | rule | D mixolydian | True | False | happy | 0.931 | 0.830 |
| drunken_sailor | hybrid | D mixolydian | True | False | happy | 0.932 | 0.835 |
| greensleeves | rule | A minor | True | True | sad | 0.891 | 0.871 |
| greensleeves | hybrid | A minor | True | True | sad | 0.891 | 0.871 |

**Emotion reader on single melodies.** A reading is plausible if its top emotion is one a listener could hear in the tune (`MELODY_MOODS` in `test_sets.py`). The classifier was trained on full songs; the emotion reader now uses a mode/energy reading for single parts and is applied only when reliable. `MELODY_MOODS` has 2-3 moods per melody; with 6 melodies the score mostly reflects how many moods each set allows.

| reader | plausible top emotion |
|---|---|
| classifier only (before the fix) | 0.000 |
| emotion reader (now) | 0.333 |

Readings reliable enough to drive generation: 83%.

| melody | classifier only | emotion reader | applied |
|---|---|---|---|
| ode_to_joy | fear | happy | True |
| twinkle_twinkle | sad | warm | True |
| frere_jacques | sad | warm | True |
| amazing_grace | fear | romantic | True |
| drunken_sailor | sad | magnificent | False |
| greensleeves | fear | sad | True |

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

**Held-out set** (`evaluation/heldout_prompts.json`, 12 prompts written blind to `agents.py`): field accuracy 79%, prompts fully correct 58%.

| prompt | expected | got | all correct |
|---|---|---|---|
| piano piece for the day my dog died, B flat minor | {"emotion": "sad", "key": "A#", "scale_mode": "minor"} | {"emotion": null, "key": null, "scale_mode": "minor"} | False |
| Smooth jazz for a candlelit dinner date with my girlfriend. Eb major, 32 bars please | {"genre": "jazz", "emotion": "romantic", "key": "D#", "scale_mode": "major", "bars": 32} | {"genre": "jazz", "emotion": "romantic", "key": null, "scale_mode": "major", "bars": 32} | False |
| cheerful bouncy bluegrass hoedown 140bpm | {"genre": "country", "emotion": "happy", "tempo": 140} | {"genre": "country", "emotion": "happy", "tempo": 140} | True |
| horror movie chase scene, something creepy and terifying | {"emotion": "fear"} | {"emotion": "fear"} | True |
| very soft hushed lullaby, barely audible | {"emotion": "quiet"} | {"emotion": "quiet"} | True |
| Grand orchestral fanfare for a royal coronation, in D major | {"genre": "classical", "emotion": "magnificent", "key": "D", "scale_mode": "major"} | {"genre": "classical", "emotion": "magnificent", "key": "D", "scale_mode": "major"} | True |
| heavy metal riff, pure rage, 16 bars | {"genre": "rock", "emotion": "angry", "bars": 16} | {"genre": "rock", "emotion": "angry", "bars": 16} | True |
| silly cartoon music, like a clown tripping over his own shoes | {"emotion": "funny"} | {"emotion": "funny"} | True |
| sluggish sleepy hammock afternoon, cant be bothered to move | {"emotion": "lazy"} | {"emotion": "quiet"} | False |
| 8 bar irish folk jig in G | {"genre": "traditional", "key": "G", "bars": 8} | {"genre": "traditional", "key": "G", "bars": null} | False |
| high energy pop banger for the club, adrenaline rush, one hundred twenty eight beats per minute | {"genre": "pop", "emotion": "exciting", "tempo": 128} | {"genre": "pop", "emotion": "exciting", "tempo": null} | False |
| cozy evening by the fireplace with family and hot cocoa | {"emotion": "warm"} | {"emotion": "warm"} | True |

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
| rule | 0.750 | 0.828 | 0.816 | 0.962 | 0.757 | 1.000 | 0.068 |
| hybrid | 0.750 | 0.828 | 0.816 | 0.962 | 0.757 | 1.000 | 1.390 |


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
| plan | 0.003 |
| generate | 0.354 |
| critique | 0.005 |
| build_on | 0.052 |
| generate_more | 0.267 |
| stems | 0.066 |

Critic: 4.0 notes per result; by kind {'instrumentation': 11, 'progression': 7, 'emotion': 6}; by severity {'suggestion': 11, 'info': 12, 'warning': 1}; 4% carry a one-click fix.

The user-study half of test 3 (people using the GUI) is in `evaluation/user_study.md`.
