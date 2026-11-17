"""Split a generated candidate into one MIDI file per track (editable stems), zipped in memory.

Nothing is written to disk: the zip is built in a BytesIO buffer, like every other result.
"""
from __future__ import annotations

import io
import re
import zipfile

from .midi_io import Score, score_to_bytes


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "track"


def stem_scores(score: Score, role_of_track: dict[int, str] | None = None) -> list[tuple[str, Score]]:
    """One single-track Score per track, keeping the tempo map, meter and key signatures."""
    out: list[tuple[str, Score]] = []
    for i, track in enumerate(score.tracks, start=1):
        if not track.notes:
            continue
        stem = score.copy()
        stem.tracks = [t for t in stem.tracks if t.track_id == track.track_id]
        role = (role_of_track or {}).get(track.track_id) or track.role or "track"
        name = f"{i:02d}_{_slug(role)}_{_slug(track.instrument_name)}.mid"
        out.append((name, stem))
    return out


def stems_zip(score: Score, full_midi: bytes, prefix: str, role_of_track: dict[int, str] | None = None) -> bytes:
    """Zip with the full arrangement plus one .mid per non-empty track."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{prefix}_full.mid", full_midi)
        for name, stem in stem_scores(score, role_of_track):
            z.writestr(f"stems/{name}", score_to_bytes(stem))
    return buf.getvalue()
