"""The JSON handoff between stages that run on different machines."""

import job_state


def _shot(shot_id, frame_count=2):
    return {
        "id": shot_id,
        "start": float(shot_id),
        "end": float(shot_id + 1),
        "frames": [
            {
                "index": i,
                "time": shot_id + i * 0.5,
                "key": f"frames/shot_{shot_id:04d}_{i:02d}.jpg",
            }
            for i in range(frame_count)
        ],
    }


def test_shots_roundtrip_with_probe_metadata(job_blobs):
    shots = [_shot(0), _shot(1)]
    meta = {"fps": 25.0, "frame_count": 100.0, "duration_sec": 4.0}

    job_state.save_shots(job_blobs, shots, meta)
    loaded_shots, loaded_meta = job_state.load_shots(job_blobs)

    assert loaded_shots == shots
    assert loaded_meta == meta


def test_vision_results_are_stored_per_shot(job_blobs):
    """One blob per shot is what keeps concurrent vision tasks from colliding."""
    shot_a, shot_b = _shot(0), _shot(1)
    shot_a["frames"][0]["visual"] = {"description": "a red car"}
    shot_b["frames"][0]["visual"] = {"description": "a blue door"}

    job_state.save_shot_vision(job_blobs, shot_a)
    job_state.save_shot_vision(job_blobs, shot_b)

    assert job_blobs.path("state/vision/shot_0000.json").exists()
    assert job_blobs.path("state/vision/shot_0001.json").exists()

    merged = job_state.load_vision_into(job_blobs, [_shot(0), _shot(1)])
    assert merged[0]["frames"][0]["visual"]["description"] == "a red car"
    assert merged[1]["frames"][0]["visual"]["description"] == "a blue door"


def test_missing_vision_leaves_that_shot_unanalyzed_rather_than_failing(job_blobs):
    """Losing one shot's descriptions must not lose the whole job.

    `visual` is Optional everywhere downstream, so an unanalyzed shot still
    produces a timeline; raising here would discard every other stage's work.
    """
    shot_a = _shot(0)
    shot_a["frames"][0]["visual"] = {"description": "a red car"}
    job_state.save_shot_vision(job_blobs, shot_a)  # shot 1 is never written

    merged = job_state.load_vision_into(job_blobs, [_shot(0), _shot(1)])

    assert merged[0]["frames"][0]["visual"]["description"] == "a red car"
    assert "visual" not in merged[1]["frames"][0]


def test_audio_roundtrip_restores_regions_as_tuples(job_blobs):
    """JSON has no tuples, but every consumer unpacks regions positionally."""
    regions = [(0.0, 1.5), (3.0, 4.25)]
    transcript = [{"start": 0.0, "end": 1.5, "text": "hello"}]

    job_state.save_audio(job_blobs, True, regions, transcript)
    has_audio, loaded_regions, loaded_transcript = job_state.load_audio(job_blobs)

    assert has_audio is True
    assert loaded_regions == regions
    assert all(isinstance(r, tuple) for r in loaded_regions)
    assert loaded_transcript == transcript


def test_audio_roundtrip_for_a_video_with_no_soundtrack(job_blobs):
    job_state.save_audio(job_blobs, False, [], [])

    has_audio, regions, transcript = job_state.load_audio(job_blobs)

    assert has_audio is False
    assert regions == []
    assert transcript == []
