"""The evaluation's fixed test sets and the user-study scoring stay well formed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.analyze_user_study import sus  # noqa: E402
from evaluation.test_sets import MELODIES, PROGRESSIONS, PROMPTS, line_to_score, progression_to_score  # noqa: E402
from midicomposer.agents import plan_from_text  # noqa: E402
from midicomposer.root import detect_root  # noqa: E402


def test_melodies_fill_whole_bars_and_keep_their_tonic():
    for name, notes, meter, tempo, pc, _ in MELODIES:
        beats = sum(d for _, d in notes)
        assert beats % (meter[0] * 4 / meter[1]) == 0, name
        assert detect_root(line_to_score(notes, meter, tempo)).root_pitch_class == pc, name


def test_progressions_build():
    for name, chords, pc, mode, tempo, *_ in PROGRESSIONS:
        s = progression_to_score(chords, pc, mode, tempo)
        assert len({n.start for n in s.tracks[0].notes}) == len(chords), name


def test_prompt_labels_use_valid_fields():
    allowed = {"genre", "emotion", "key", "scale_mode", "tempo", "bars", "density"}
    for prompt, expected in PROMPTS:
        assert expected and set(expected) <= allowed, prompt
        assert "settings" in plan_from_text(prompt, ["pop"], ["happy"])


def test_sus_scoring():
    assert sus({f"SUS{i}": "3" for i in range(1, 11)}) == 50.0
    assert sus({f"SUS{i}": ("5" if i % 2 else "1") for i in range(1, 11)}) == 100.0
    assert sus({"SUS1": "4"}) is None
