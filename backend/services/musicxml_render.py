"""Render a piano performance directly to MusicXML for local score outputs."""
from __future__ import annotations

import tempfile
from pathlib import Path

from backend.contracts import HumanizedPerformance


class MusicXmlRenderError(RuntimeError):
    """Raised when a performance cannot be rendered to MusicXML."""


def _metadata_title(perf: HumanizedPerformance) -> str:
    return perf.score.metadata.title or "Untitled"


def _metadata_composer(perf: HumanizedPerformance) -> str:
    return perf.score.metadata.composer or "Unknown"


def _key_name(key_label: str) -> tuple[str, str]:
    root, _, mode = key_label.partition(":")
    return root or "C", mode or "major"


def render_musicxml_bytes(perf: HumanizedPerformance) -> bytes:
    """Render a two-staff piano MusicXML document from a performance."""
    try:
        from music21 import chord, clef, instrument, key, metadata, meter, note, stream, tempo
    except ImportError as exc:
        raise MusicXmlRenderError("music21 is not installed; cannot render MusicXML") from exc

    score = stream.Score(id="ohsheet-score")
    score.insert(0, metadata.Metadata(title=_metadata_title(perf), composer=_metadata_composer(perf)))

    tempo_map = perf.score.metadata.tempo_map
    bpm = tempo_map[0].bpm if tempo_map else 120.0
    ts_num, ts_den = perf.score.metadata.time_signature
    key_root, key_mode = _key_name(perf.score.metadata.key)

    def make_part(part_id: str, part_name: str, part_clef):
        part = stream.Part(id=part_id)
        part.partName = part_name
        part.insert(0, instrument.Piano())
        part.insert(0, meter.TimeSignature(f"{ts_num}/{ts_den}"))
        part.insert(0, tempo.MetronomeMark(number=bpm))
        try:
            part.insert(0, key.Key(key_root, key_mode))
        except Exception:
            part.insert(0, key.Key("C", "major"))
        part.insert(0, part_clef)
        return part

    def fill_part(part, hand: str) -> None:
        groups: dict[tuple[float, float], list[tuple[int, int]]] = {}
        for en in perf.expressive_notes:
            if en.hand != hand:
                continue
            onset = max(0.0, en.onset_beat)
            duration = max(0.25, en.duration_beat)
            groups.setdefault((onset, duration), []).append((en.pitch, en.velocity))

        for (onset, duration), pitches in sorted(groups.items()):
            if len(pitches) == 1:
                obj = note.Note(pitches[0][0])
                obj.volume.velocity = pitches[0][1]
            else:
                obj = chord.Chord([pitch for pitch, _velocity in pitches])
                obj.volume.velocity = max(velocity for _pitch, velocity in pitches)
            obj.quarterLength = duration
            part.insert(onset, obj)

    right = make_part("P1", "Piano RH", clef.TrebleClef())
    left = make_part("P2", "Piano LH", clef.BassClef())
    fill_part(right, "rh")
    fill_part(left, "lh")
    if not right.recurse().notes and not left.recurse().notes:
        raise MusicXmlRenderError("performance contained no notes to render")

    score.insert(0, right)
    score.insert(0, left)

    with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        score.write("musicxml", fp=str(tmp_path))
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)
