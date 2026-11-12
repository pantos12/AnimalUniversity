import pytest

from animaluniversity.pipeline import RtspSource, VideoFileSource
from animaluniversity.ingest.milestone import MilestoneRecordingManifest
from tests.test_milestone_ingest import BASE


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_sources_reject_invalid_sample_rates(tmp_path, fps):
    clip = tmp_path / "clip.mp4"
    clip.touch()
    with pytest.raises(ValueError, match="fps"):
        VideoFileSource(clip, target_fps=fps)
    with pytest.raises(ValueError, match="fps"):
        RtspSource("rtsp://test", target_fps=fps)


@pytest.mark.parametrize("name", ["", ".", "..", "..\\rec.mp4", "C:rec.mp4", "rec\x00.mp4"])
def test_manifest_rejects_non_portable_or_empty_filenames(name):
    with pytest.raises(ValueError, match="file name"):
        MilestoneRecordingManifest(**{**BASE, "video_filename": name})


@pytest.mark.parametrize("field", ["camera_id", "camera_name", "enclosure_id"])
def test_manifest_rejects_blank_identifiers(field):
    with pytest.raises(ValueError, match="non-empty"):
        MilestoneRecordingManifest(**{**BASE, field: "  "})
