import time

import pytest

from backend.api.deps import get_job_manager
from backend.contracts import InputBundle, InputMetadata
from backend.main import app
from backend.services.chord_sheet import chord_sheet_to_transcription
from backend.services.musicxml_render import render_musicxml_bytes


def test_chord_sheet_parser_turns_chords_into_piano_tracks():
    bundle = InputBundle(
        chord_sheet_text="""
Verse:
C Am F G7
Chorus:
[Dm7]Lift [G7]up [C]again
""",
        metadata=InputMetadata(title="Chart", artist="QA", source="chord_sheet"),
    )

    txr = chord_sheet_to_transcription(bundle)

    assert len(txr.midi_tracks) == 2
    assert txr.midi_tracks[0].instrument == "melody"
    assert txr.midi_tracks[1].instrument == "bass"
    assert len(txr.analysis.chords) == 7
    assert txr.analysis.chords[0].label == "C:maj"
    assert txr.analysis.chords[1].label == "A:min"
    assert txr.analysis.chords[3].label == "G:7"
    assert txr.analysis.sections[0].label == "verse"
    assert txr.analysis.sections[1].label == "chorus"


def test_chord_sheet_parser_rejects_text_without_chords():
    bundle = InputBundle(
        chord_sheet_text="Verse:\nThese are only lyrics",
        metadata=InputMetadata(title="Lyrics", source="chord_sheet"),
    )

    with pytest.raises(ValueError, match="no chord symbols"):
        chord_sheet_to_transcription(bundle)


def test_chord_sheet_musicxml_renderer_outputs_notation_xml():
    bundle = InputBundle(
        chord_sheet_text="C Am F G7",
        metadata=InputMetadata(title="Chart", artist="QA", source="chord_sheet"),
    )
    txr = chord_sheet_to_transcription(bundle)

    from backend.services.arrange import ArrangeService

    import asyncio

    score = asyncio.run(ArrangeService().run(txr))
    from backend.contracts import ExpressionMap, ExpressiveNote, HumanizedPerformance, QualitySignal

    perf = HumanizedPerformance(
        expressive_notes=[
            ExpressiveNote(
                score_note_id=n.id,
                pitch=n.pitch,
                onset_beat=n.onset_beat,
                duration_beat=n.duration_beat,
                velocity=n.velocity,
                hand=hand,
                voice=n.voice,
                timing_offset_ms=0.0,
                velocity_offset=0,
            )
            for hand, notes in (("rh", score.right_hand), ("lh", score.left_hand))
            for n in notes
        ],
        expression=ExpressionMap(),
        score=score,
        quality=QualitySignal(overall_confidence=0.95),
    )

    musicxml = render_musicxml_bytes(perf)

    assert b"<score-partwise" in musicxml
    assert b"<part-name" in musicxml


def test_create_job_from_chord_sheet_runs_to_completion(client, monkeypatch):
    from backend.config import settings

    monkeypatch.setattr(settings, "score_pipeline", "arrange")
    response = client.post(
        "/v1/jobs",
        json={
            "title": "Practice Chart",
            "artist": "QA",
            "chord_sheet_text": "Verse:\nC Am F G7\nChorus:\nDm7 G7 C",
        },
    )
    assert response.status_code == 202, response.text
    job = response.json()
    assert job["variant"] == "chord_sheet"
    job_id = job["job_id"]

    deadline = time.time() + 5
    status = None
    while time.time() < deadline:
        status = client.get(f"/v1/jobs/{job_id}").json()
        if status["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.05)

    assert status is not None
    assert status["status"] == "succeeded", status
    assert status["result"]["musicxml_uri"]
    assert status["result"]["humanized_midi_uri"]

    musicxml = client.get(f"/v1/artifacts/{job_id}/musicxml")
    assert musicxml.status_code == 200
    assert b"<score-partwise" in musicxml.content

    midi = client.get(f"/v1/artifacts/{job_id}/midi")
    assert midi.status_code == 200
    assert midi.content.startswith(b"MThd")


def test_chord_sheet_text_lands_on_bundle(client):
    response = client.post(
        "/v1/jobs",
        json={"title": "Stored Chart", "chord_sheet_text": "C F G C"},
    )
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]

    manager = app.dependency_overrides.get(get_job_manager, get_job_manager)()
    record = manager.get(job_id)
    assert record is not None
    assert record.bundle.metadata.source == "chord_sheet"
    assert record.bundle.chord_sheet_text == "C F G C"


def test_create_job_rejects_chord_sheet_with_audio(client):
    audio = client.post(
        "/v1/uploads/audio",
        files={"file": ("song.wav", b"RIFFfake wav data", "audio/wav")},
    ).json()
    response = client.post(
        "/v1/jobs",
        json={"audio": audio, "chord_sheet_text": "C F G C"},
    )
    assert response.status_code == 400
    assert "chord_sheet_text" in response.json()["detail"]
