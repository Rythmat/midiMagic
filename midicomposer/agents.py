"""The 'agentic' layer around the generator: text planner, emotion reader and critic.

Each agent has a deterministic baseline here (keywords, the trained classifier, music rules), so the app works with
no LLM and no API key. The LLM versions (Llama planner/critic, see the proposal) should plug in behind the same
functions and return the same JSON shapes, so the API and the UI don't change:

    plan_from_text(text, genres, emotions)  -> PLAN       (POST /plan-from-text)
    read_emotion(score, root, classifier)   -> EMOTION    (included in POST /analyze-midi as analysis["style"])
    critique(result, candidate_index)       -> CRITIQUE   (GET /generation/{job_id}/critique)

PLAN     {"source", "prompt", "settings": {genre, emotion, key, scale_mode, tempo, bars, form, instrumentation,
          density, generated_roles}, "emotion_scores": [{label, score}], "explanation": [str], "unmatched": [str]}
          Every setting may be null (= leave the form as it is / automatic).
EMOTION  {"source", "emotion": [{label, score}] (top 3), "genre": [{label, score}] (top 3)}
CRITIQUE {"source", "summary", "notes": [{kind, severity, message, action}]}
          kind: instrumentation | arrangement | transition | progression | emotion | validation
          severity: info | suggestion | warning
          action: null | {"type": "regenerate_track", "track_id", "label"}
                       | {"type": "regenerate_section", "section_index", "label"}
"""
from __future__ import annotations

import re

import numpy as np

from .features import score_features
from .midi_io import Score
from .root import RootAnalysis
from .styles import EMOTION_KEYWORDS, EMOTIONS, GENRE_KEYWORDS, INSTRUMENT_WORDS, PRESETS
from .theory import FLAT_NAMES, PITCH_NAMES

# ----------------------------------------------------------------------------- text planner

_EXTRA_EMOTION_WORDS = {
    "happy": ["happy", "joy", "cheer", "sunny", "bright", "uplift", "glad", "summer"],
    "sad": ["sad", "melanch", "gloom", "heartbreak", "lonely", "cry", "tears", "grief", "rain", "blue mood"],
    "exciting": ["excit", "upbeat", "energetic", "hype", "party", "intense", "driving", "adrenaline"],
    "angry": ["angry", "aggress", "rage", "fury", "furious", "heavy"],
    "fear": ["fear", "scary", "dark", "tense", "anxious", "horror", "creepy", "eerie", "suspense"],
    "quiet": ["quiet", "calm", "peace", "still", "soft", "meditat", "sleep"],
    "lazy": ["lazy", "relax", "chill", "laid back", "sunday", "slow afternoon", "lofi", "lo-fi"],
    "warm": ["warm", "cozy", "gentle", "nostalg", "home"],
    "romantic": ["romantic", "love", "tender", "wedding", "date", "intimate"],
    "funny": ["funny", "playful", "humor", "silly", "quirky", "cartoon"],
    "magnificent": ["magnificent", "epic", "grand", "heroic", "majestic", "triumph", "cinematic", "victory"],
}
_TEMPO_WORDS = [("very slow", 64), ("slow", 76), ("relaxed", 88), ("moderate", 104), ("medium", 104),
                ("upbeat", 124), ("fast", 140), ("very fast", 168)]
_DENSITY_WORDS = [("minimal", 0.2), ("sparse", 0.25), ("simple", 0.35), ("busy", 0.75), ("dense", 0.8),
                  ("complex", 0.75), ("rich", 0.65)]
_FORM_WORDS = [("12 bar blues", "12-bar-blues"), ("12-bar", "12-bar-blues"), ("aaba", "AABA"), ("aabb", "AABB"),
               ("aba", "ABA"), ("verse", "verse-chorus"), ("chorus", "verse-chorus"), ("rondo", "rondo"),
               ("drop", "build-drop"), ("head", "head-solo-head"), ("strophic", "strophic"),
               ("through-composed", "through-composed")]
_KEY_RE = re.compile(r"\b(?:in|key of)\s+([A-Ga-g])([#b♯♭]?)(?:\s*-?\s*(major|minor|maj|min|m|dorian|mixolydian|lydian|"
                     r"phrygian|locrian|harmonic minor))?\b")
_MODE_NAMES = {"maj": "major", "major": "major", "min": "minor", "minor": "minor", "m": "minor",
               "harmonic minor": "harmonic_minor"}


def _has(text: str, word: str) -> bool:
    return re.search(r"(?<![a-z])" + re.escape(word), text) is not None


def _emotion_scores(text: str, emotions: list[str]) -> list[dict]:
    hits: dict[str, float] = {}
    for label in emotions:
        words = set(_EXTRA_EMOTION_WORDS.get(label, [label]))
        words |= {k for k, v in EMOTION_KEYWORDS.items() if v == label}
        n = sum(1 for w in words if _has(text, w))
        if n:
            hits[label] = float(n)
    total = sum(hits.values())
    return [{"label": k, "score": round(v / total, 3)} for k, v in sorted(hits.items(), key=lambda kv: -kv[1])]


def _genre(text: str, genres: list[str]) -> tuple[str | None, str | None]:
    for g in genres:
        if _has(text, g.replace("_", " ")) or _has(text, g):
            return g, g
    for kw, target in GENRE_KEYWORDS.items():
        if _has(text, kw.replace("_", " ")) and target in genres:
            return target, kw
    return None, None


def plan_from_text(text: str, genres: list[str], emotions: list[str]) -> dict:
    """Keyword baseline for the text planner: maps a free-text prompt onto the generator's settings."""
    t = " " + text.lower().strip() + " "
    explain: list[str] = []
    s: dict = {k: None for k in ("genre", "emotion", "key", "scale_mode", "tempo", "bars", "form",
                                 "instrumentation", "density")}
    s["generated_roles"] = ["full_composition"]

    s["genre"], why = _genre(t, genres)
    if s["genre"]:
        explain.append(f"genre '{s['genre']}' from \"{why}\"")
    emo = [e for e in _emotion_scores(t, emotions)]
    if emo:
        s["emotion"] = emo[0]["label"]
        explain.append(f"emotion '{s['emotion']}' from the mood words")

    m = _KEY_RE.search(text)
    if m:
        name = m.group(1).upper() + m.group(2).replace("♯", "#").replace("♭", "b")
        pc = (PITCH_NAMES.index(name) if name in PITCH_NAMES else FLAT_NAMES.index(name)
              if name in FLAT_NAMES else None)
        if pc is not None:
            s["key"] = PITCH_NAMES[pc]
            if m.group(3):
                mode = m.group(3).lower()
                s["scale_mode"] = _MODE_NAMES.get(mode, mode)
            explain.append(f"key {s['key']}" + (f" {s['scale_mode']}" if s["scale_mode"] else ""))
    if s["scale_mode"] is None:                 # otherwise auto: the planner derives the mode from the emotion
        for mode in ("minor", "major"):
            if _has(t, mode):
                s["scale_mode"] = mode
                explain.append(f"{mode} mode")
                break

    bpm = re.search(r"(\d{2,3})\s*(?:bpm|beats per minute)", t)
    if bpm and 20 <= int(bpm.group(1)) <= 300:
        s["tempo"] = int(bpm.group(1))
        explain.append(f"tempo {s['tempo']} BPM")
    else:
        for word, value in sorted(_TEMPO_WORDS, key=lambda kv: -len(kv[0])):
            if _has(t, word):
                s["tempo"] = value
                explain.append(f"tempo ~{value} BPM from \"{word}\"")
                break
    bars = re.search(r"(\d{1,3})\s*(?:bars|measures)", t)
    if bars and 1 <= int(bars.group(1)) <= 256:
        s["bars"] = int(bars.group(1))
        explain.append(f"{s['bars']} bars")
    for word, form in _FORM_WORDS:
        if _has(t, word):
            s["form"] = form
            explain.append(f"form {form}")
            break
    for word, value in _DENSITY_WORDS:
        if _has(t, word):
            s["density"] = value
            explain.append(f"density {value} from \"{word}\"")
            break

    preset = next((p for p in sorted(PRESETS, key=len, reverse=True) if _has(t, p)), None)
    if preset:
        s["instrumentation"] = preset
    else:
        found = []
        for word in sorted(INSTRUMENT_WORDS, key=len, reverse=True):
            if _has(t, word) and not any(word in f for f in found):
                found.append(word)
        if found:
            s["instrumentation"] = ", ".join(reversed(found))
    if s["instrumentation"]:
        explain.append(f"instruments: {s['instrumentation']}")
    if re.search(r"\b(no|without)\s+drums?\b", t):
        s["instrumentation"] = ", ".join(filter(None, [s["instrumentation"], "no drums"]))
        explain.append("no drums")

    unmatched = [k for k in ("genre", "emotion") if s[k] is None]
    return {"source": "keywords", "prompt": text, "settings": s, "emotion_scores": emo[:3],
            "explanation": explain, "unmatched": unmatched}


# ----------------------------------------------------------------------------- emotion reader

def _nearest_emotions(valence: float, arousal: float, emotions: list[str]) -> list[dict]:
    d = {e: np.hypot(EMOTIONS[e].valence - valence, EMOTIONS[e].arousal - arousal) for e in emotions if e in EMOTIONS}
    w = {e: float(np.exp(-3 * x)) for e, x in d.items()}
    total = sum(w.values()) or 1.0
    return [{"label": e, "score": round(v / total, 3)} for e, v in sorted(w.items(), key=lambda kv: -kv[1])[:3]]


def read_emotion(score: Score, root: RootAnalysis, classifier, emotions: list[str]) -> dict:
    """Emotion (and genre) of a MIDI file: the trained classifier when available, else a mode/energy heuristic."""
    if classifier is not None and root.root_pitch_class is not None:
        probs = classifier.predict(score_features(score, root))
        top = lambda d: [{"label": k, "score": round(v, 3)} for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:3]]
        return {"source": "classifier", "emotion": top(probs["emotion"]), "genre": top(probs["genre"])}
    notes = [n for t in score.tracks for n in t.notes]
    if not notes:
        return {"source": "heuristic", "emotion": [], "genre": []}
    valence = 0.5 if root.mode_family == "major" else -0.5
    vel = float(np.mean([n.velocity for n in notes]))
    beats = max(1.0, score.end_beat)
    onsets_per_beat = len({round(n.start, 2) for n in notes}) / beats
    arousal = float(np.clip((score.initial_bpm - 100) / 60 + (vel - 75) / 60 + (onsets_per_beat - 1.5) / 3, -1, 1))
    return {"source": "heuristic", "emotion": _nearest_emotions(valence, arousal, emotions), "genre": []}


# ----------------------------------------------------------------------------- critic

def _bar_beats(meter: str) -> float:
    num, den = (int(x) for x in meter.split("/"))
    return num * 4.0 / den


def _notes_in(track: dict, start: float, end: float) -> list:
    return [n for n in track["notes"] if start <= n[0] < end]


def critique(result: dict, candidate_index: int = 0) -> dict:
    """Rule-based critic over the API's result view: instrumentation, arrangement, transitions, progression."""
    cand = result["candidates"][candidate_index]
    plan = result["plan"]
    tracks = [t for t in cand["tracks"] if t["n_notes"]]
    sections = plan.get("sections") or []
    bb = _bar_beats(result.get("meter") or plan.get("meter") or "4/4")
    regen_ok = {t["track_id"] for t in tracks if t["source"] == "generated" and not t["locked"]}
    notes: list[dict] = []

    def track_action(t):
        if t["track_id"] not in regen_ok:
            return None
        return {"type": "regenerate_track", "track_id": t["track_id"], "label": f"Regenerate {t['name']}"}

    def section_action(i):
        if not regen_ok or i >= len(sections):
            return None
        return {"type": "regenerate_section", "section_index": i,
                "label": f"Regenerate {sections[i]['name']} (bars {sections[i]['start_bar'] + 1}-{sections[i]['end_bar']})"}

    def add(kind, severity, message, action=None):
        notes.append({"kind": kind, "severity": severity, "message": message, "action": action})

    # Instrumentation: duplicate timbres and crowded registers among pitched tracks.
    pitched = [t for t in tracks if not t["is_drum"]]
    by_program: dict[int, list] = {}
    for t in pitched:
        by_program.setdefault(t["program"], []).append(t)
    for prog, ts in by_program.items():
        if len(ts) > 1:
            # No action button: regenerating keeps the plan's instrument, so the fix is a new Instrumentation setting.
            names = ", ".join(t["name"] for t in ts)
            add("instrumentation", "suggestion",
                f"{names} {'both' if len(ts) == 2 else 'all'} use {ts[0]['instrument']}; giving one of them a "
                f"different instrument would separate the parts more clearly (set it in Instrumentation).")
    medians = {t["track_id"]: float(np.median([n[2] for n in t["notes"]])) for t in pitched}
    for i, a in enumerate(pitched):
        for b in pitched[i + 1:]:
            if abs(medians[a["track_id"]] - medians[b["track_id"]]) <= 2 and a["program"] != b["program"]:
                add("instrumentation", "suggestion",
                    f"{a['name']} and {b['name']} sit in the same register (around MIDI note "
                    f"{int(medians[a['track_id']])}), so they may mask each other. Lower the density or drop one "
                    "of them (lock the others and regenerate), or pick instruments with different ranges.")
    roles = {t["role"] for t in tracks}
    if pitched and not roles & {"bass", "bass_line", "walking_bass", "low_texture", "lows"}:
        add("instrumentation", "suggestion", "There is no bass part, so the low end may feel thin. Try adding a "
            "bass line with Build on this.")
    has_drums = any(t["is_drum"] for t in tracks)
    if not has_drums and plan.get("drum_style") not in (None, "none", "orchestral"):
        add("instrumentation", "suggestion", f"{plan.get('genre', 'This style')} usually has drums, but this "
            "arrangement has none. Add a drum part with Build on this, unless you want it unplugged.")
    if pitched and not roles & {"lead", "lead_melody", "highs", "counterline"}:
        add("arrangement", "info", "There is no lead melody; this works as a backing track. Add a lead melody with "
            "Build on this if you want a tune on top.")

    # Arrangement and transitions between sections.
    for i, sec in enumerate(sections):
        start, end = sec["start_bar"] * bb, sec["end_bar"] * bb
        silent = [t for t in tracks if t["track_id"] in regen_ok and len(_notes_in(t, 0, 1e9)) >= 8
                  and not _notes_in(t, start, end) and t["role"] in (sec.get("active_roles") or [t["role"]])]
        for t in silent:
            add("arrangement", "suggestion", f"{t['name']} is silent in the {sec['name']} (bars {sec['start_bar'] + 1}"
                f"-{sec['end_bar']}) although it plays elsewhere.", section_action(i))
    for i in range(1, len(sections)):
        prev, cur = sections[i - 1], sections[i]
        de = (cur.get("energy") or 0) - (prev.get("energy") or 0)
        if prev.get("name") != cur.get("name") and abs(de) < 0.05 and prev.get("progression") == cur.get("progression"):
            add("transition", "suggestion", f"The move from {prev['name']} to {cur['name']} (bar {cur['start_bar'] + 1}) "
                "has the same chords and energy, so the new section may not feel new.", section_action(i))
        elif de > 0.35:
            add("transition", "info", f"Big energy jump into the {cur['name']} at bar {cur['start_bar'] + 1} "
                f"({prev.get('energy', 0):.2f} → {cur.get('energy', 0):.2f}). A fill or build at the end of the "
                f"{prev['name']} would smooth it.", section_action(i - 1))
    names = [(s.get("name") or "").lower() for s in sections]
    letters = [s.get("letter") for s in sections]
    contrast = "B" in letters or any(w in n for n in names for w in ("bridge", "solo", "break", "drop"))
    if len(sections) >= 4 and not contrast and ("chorus" in names or len(set(letters)) <= 2):
        add("progression", "info", "The form has no bridge or contrasting section. A bridge before the last chorus "
            "would add contrast (choose a form such as AABA, or one with a solo or bridge).")
    if len(sections) >= 2:
        peak = max(range(len(sections)), key=lambda i: sections[i].get("energy") or 0)
        shape = " → ".join(f"{s['name']} {s.get('energy', 0):.2f}" for s in sections)
        add("progression", "info", f"Energy shape: {shape}. The peak is the {sections[peak]['name']} at bar "
            f"{sections[peak]['start_bar'] + 1}.")

    # Emotion / genre fit and validation.
    sc = cand["scores"]
    if sc.get("emotion", 1) < 0.5:
        heard = (sc.get("classifier_top") or {}).get("emotion") or {}
        heard_txt = f" The emotion model hears it as {', '.join(heard)}." if heard else ""
        lead = next((t for t in tracks if t["role"] in ("lead_melody", "lead") and t["track_id"] in regen_ok), None)
        add("emotion", "warning", f"This candidate only weakly matches '{plan.get('emotion')}' "
            f"(emotion fit {sc['emotion']:.2f}).{heard_txt} Try another candidate or regenerate the melody.",
            track_action(lead) if lead else None)
    elif sc.get("emotion") is not None:
        how = "well" if sc["emotion"] >= 0.7 else "partly"
        add("emotion", "info", f"Matches the requested '{plan.get('emotion')}' mood {how} "
            f"(emotion fit {sc['emotion']:.2f}).")
    issues = [i for i in (cand.get("validation") or {}).get("issues", []) if i.get("severity") in ("error", "warning")]
    if issues:
        add("validation", "warning", f"{len(issues)} validation issue(s), e.g. {issues[0].get('message')}")

    order = {"warning": 0, "suggestion": 1, "info": 2}
    notes.sort(key=lambda n: order[n["severity"]])
    n_sugg = sum(n["severity"] != "info" for n in notes)
    summary = (f"{n_sugg} suggestion(s) for candidate #{candidate_index + 1}." if n_sugg
               else f"No problems found in candidate #{candidate_index + 1}.")
    return {"source": "rules", "summary": summary, "notes": notes}
