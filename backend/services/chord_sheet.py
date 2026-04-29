"""Chord-sheet parser that emits a simple piano-oriented transcription.

The parser intentionally accepts the common chord-chart shapes users paste
from lead sheets:

* bracketed chords in lyric lines, e.g. ``[C]Amazing [G]grace``
* bare chord rows, e.g. ``C Am F G7 | Dm7 G7 C``
* section headings ending in ``:``, e.g. ``Verse:``

Each chord becomes one 4/4 measure. The right hand plays an eighth-note scale
pattern matching the chord quality, while the left hand sustains root/fifth
support. Downstream arrange/engrave stages turn this into sheet artifacts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from backend.contracts import (
    SCHEMA_VERSION,
    HarmonicAnalysis,
    InputBundle,
    InstrumentRole,
    MidiTrack,
    Note,
    QualitySignal,
    RealtimeChordEvent,
    Section,
    SectionLabel,
    TempoMapEntry,
    TranscriptionResult,
)

_CHORD_RE = re.compile(
    r"^(?P<root>[A-Ga-g])(?P<accidental>[#b]?)(?P<quality>maj7|maj9|maj|min7|min9|min|m7|m9|m|dim7|dim|aug|sus2|sus4|add9|7|9|11|13|6)?(?P<bass>/[A-Ga-g][#b]?)?$",
)
_BRACKETED_RE = re.compile(r"\[([^\]]+)\]")

_PC_BY_NAME = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}
_CANONICAL_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

_MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11, 12]
_MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10, 12]
_DIM_SCALE = [0, 2, 3, 5, 6, 8, 9, 12]
_AUG_SCALE = [0, 2, 4, 6, 8, 10, 12, 14]


@dataclass(frozen=True)
class ParsedChord:
    raw: str
    label: str
    root_pc: int
    bass_pc: int
    quality: str


def _normalise_pitch_name(root: str, accidental: str = "") -> str:
    return root.upper() + accidental


def _parse_chord_token(token: str) -> ParsedChord | None:
    token = token.strip().strip("|,;")
    if not token or token.upper() in {"N.C.", "NC", "N/C"}:
        return None

    match = _CHORD_RE.match(token)
    if match is None:
        return None

    root_name = _normalise_pitch_name(match.group("root"), match.group("accidental") or "")
    root_pc = _PC_BY_NAME.get(root_name)
    if root_pc is None:
        return None

    bass_raw = match.group("bass")
    bass_pc = root_pc
    if bass_raw:
        bass_name = bass_raw[1].upper() + bass_raw[2:]
        bass_pc = _PC_BY_NAME.get(bass_name, root_pc)

    raw_quality = match.group("quality") or ""
    quality = _quality_family(raw_quality)
    label_quality = {
        "major": "maj",
        "minor": "min",
        "dominant": "7",
        "diminished": "dim",
        "augmented": "aug",
        "suspended": "sus",
    }[quality]
    return ParsedChord(
        raw=token,
        label=f"{_CANONICAL_NAMES[root_pc]}:{label_quality}",
        root_pc=root_pc,
        bass_pc=bass_pc,
        quality=quality,
    )


def _quality_family(raw_quality: str) -> str:
    q = raw_quality.lower()
    if q in {"m", "m7", "m9", "min", "min7", "min9"}:
        return "minor"
    if q.startswith("dim"):
        return "diminished"
    if q == "aug":
        return "augmented"
    if q.startswith("sus"):
        return "suspended"
    if q in {"7", "9", "11", "13"}:
        return "dominant"
    return "major"


def _extract_chords(line: str) -> list[ParsedChord]:
    if line.endswith(":"):
        return []

    bracketed = [_parse_chord_token(m.group(1)) for m in _BRACKETED_RE.finditer(line)]
    chords = [ch for ch in bracketed if ch is not None]
    if chords:
        return chords

    # For bare chord rows, only keep tokens that fully parse as chord symbols.
    return [
        ch
        for token in line.replace("|", " | ").split()
        if (ch := _parse_chord_token(token)) is not None
    ]


def _scale_for_quality(chord: ParsedChord) -> list[int]:
    if chord.quality == "minor":
        intervals = _MINOR_SCALE
    elif chord.quality == "diminished":
        intervals = _DIM_SCALE
    elif chord.quality == "augmented":
        intervals = _AUG_SCALE
    else:
        intervals = _MAJOR_SCALE
    base = 60 + chord.root_pc
    return [base + interval for interval in intervals]


def _bass_pitch(pc: int) -> int:
    return 36 + pc


def _section_label(text: str) -> SectionLabel:
    normalized = text.strip().strip(":").lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "prechorus": "pre_chorus",
        "pre_chorus": "pre_chorus",
        "verse": "verse",
        "chorus": "chorus",
        "bridge": "bridge",
        "intro": "intro",
        "interlude": "interlude",
        "outro": "outro",
        "solo": "solo",
    }
    value = aliases.get(normalized, "other")
    return SectionLabel(value)


def chord_sheet_to_transcription(bundle: InputBundle, *, bpm: float = 96.0) -> TranscriptionResult:
    """Convert pasted chord-sheet text into a scale-based piano transcription."""
    text = (bundle.chord_sheet_text or "").strip()
    if not text:
        raise ValueError("chord sheet text is empty")

    tempo_map = [TempoMapEntry(time_sec=0.0, beat=0.0, bpm=bpm)]
    sec_per_beat = 60.0 / bpm
    measure_beats = 4.0
    measure_sec = measure_beats * sec_per_beat
    eighth_sec = 0.5 * sec_per_beat

    melody_notes: list[Note] = []
    bass_notes: list[Note] = []
    chord_events: list[RealtimeChordEvent] = []
    sections: list[Section] = []
    warnings: list[str] = []

    beat_cursor = 0.0
    current_section: tuple[float, SectionLabel] | None = None
    parsed_any = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        chords = _extract_chords(line)
        if not chords:
            if line.endswith(":"):
                if current_section is not None:
                    start, label = current_section
                    sections.append(
                        Section(
                            start_sec=start * sec_per_beat,
                            end_sec=beat_cursor * sec_per_beat,
                            label=label,
                        ),
                    )
                current_section = (beat_cursor, _section_label(line))
            continue

        parsed_any = True
        for chord in chords:
            measure_start_sec = beat_cursor * sec_per_beat
            scale = _scale_for_quality(chord)
            for idx, pitch in enumerate(scale):
                onset = measure_start_sec + idx * eighth_sec
                melody_notes.append(
                    Note(
                        pitch=max(0, min(127, pitch)),
                        onset_sec=onset,
                        offset_sec=onset + eighth_sec * 0.9,
                        velocity=82,
                    ),
                )

            root_pitch = _bass_pitch(chord.bass_pc)
            fifth_pitch = _bass_pitch((chord.root_pc + 7) % 12)
            bass_notes.extend(
                [
                    Note(
                        pitch=root_pitch,
                        onset_sec=measure_start_sec,
                        offset_sec=measure_start_sec + measure_sec,
                        velocity=72,
                    ),
                    Note(
                        pitch=fifth_pitch,
                        onset_sec=measure_start_sec + 2 * sec_per_beat,
                        offset_sec=measure_start_sec + measure_sec,
                        velocity=66,
                    ),
                ],
            )
            chord_events.append(
                RealtimeChordEvent(
                    time_sec=measure_start_sec,
                    duration_sec=measure_sec,
                    label=chord.label,
                    root=chord.root_pc,
                    confidence=1.0,
                ),
            )
            beat_cursor += measure_beats

    if current_section is not None:
        start, label = current_section
        sections.append(
            Section(
                start_sec=start * sec_per_beat,
                end_sec=max(start * sec_per_beat, beat_cursor * sec_per_beat),
                label=label,
            ),
        )

    if not parsed_any:
        raise ValueError("no chord symbols found in chord sheet text")
    if not sections:
        sections = [
            Section(
                start_sec=0.0,
                end_sec=beat_cursor * sec_per_beat,
                label=SectionLabel.OTHER,
            ),
        ]

    first = chord_events[0]
    first_quality = chord_events[0].label.split(":", 1)[1]
    key_quality = "minor" if first_quality.startswith("min") else "major"
    key = f"{_CANONICAL_NAMES[first.root]}:{key_quality}"

    return TranscriptionResult(
        schema_version=SCHEMA_VERSION,
        midi_tracks=[
            MidiTrack(
                notes=melody_notes,
                instrument=InstrumentRole.MELODY,
                program=0,
                confidence=1.0,
            ),
            MidiTrack(
                notes=bass_notes,
                instrument=InstrumentRole.BASS,
                program=0,
                confidence=1.0,
            ),
        ],
        analysis=HarmonicAnalysis(
            key=key,
            time_signature=(4, 4),
            tempo_map=tempo_map,
            chords=chord_events,
            sections=sections,
        ),
        quality=QualitySignal(
            overall_confidence=0.95,
            warnings=warnings,
        ),
    )
