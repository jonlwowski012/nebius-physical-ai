from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import pytest

from npa.workflows.encord_groot_loop import EncordGrootError, materialize


class FakeStorage:
    def __init__(self, source: Path, generated: Path) -> None:
        self.source, self.generated = source, generated
        self.uploaded: Path | None = None

    def download_directory(self, uri: str, destination: str) -> str:
        shutil.copytree(self.source if uri == "s3://test/source/" else self.generated, destination)
        return destination

    def upload_directory(self, local: str, uri: str) -> str:
        self.uploaded = self.generated.parent / "uploaded"
        shutil.copytree(local, self.uploaded)
        return uri

    def upload_file(self, local: str, uri: str) -> str:
        return uri


def _dataset(root: Path) -> None:
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "observation.images.front" / "chunk-000").mkdir(parents=True)
    pq.write_table(pa.table({"episode_index": pa.array([0, 0], type=pa.int64()), "index": pa.array([0, 1], type=pa.int64()), "frame_index": pa.array([0, 1], type=pa.int64()), "task_index": pa.array([0, 0], type=pa.int64()), "timestamp": pa.array([0.0, 0.05], type=pa.float32()), "observation.state": pa.array([[0.5], [0.6]]), "action": pa.array([[1.0], [2.0]])}), root / "data/chunk-000/file-000.parquet")
    pq.write_table(pa.table({"episode_index": pa.array([0], type=pa.int64()), "data/chunk_index": pa.array([0], type=pa.int64()), "data/file_index": pa.array([0], type=pa.int64()), "dataset_from_index": pa.array([0], type=pa.int64()), "dataset_to_index": pa.array([2], type=pa.int64()), "videos/observation.images.front/chunk_index": pa.array([0], type=pa.int64()), "videos/observation.images.front/file_index": pa.array([0], type=pa.int64()), "videos/observation.images.front/from_timestamp": pa.array([0.0]), "videos/observation.images.front/to_timestamp": pa.array([0.1])}), root / "meta/episodes/chunk-000/file-000.parquet")
    (root / "meta/info.json").write_text(json.dumps({"fps": 20, "total_episodes": 1, "total_frames": 2, "features": {"observation.images.front": {"dtype": "video", "shape": [64, 64, 3]}, "observation.state": {"dtype": "float32", "shape": [1]}, "action": {"dtype": "float32", "shape": [1]}, "episode_index": {"dtype": "int64", "shape": [1]}, "frame_index": {"dtype": "int64", "shape": [1]}, "task_index": {"dtype": "int64", "shape": [1]}, "timestamp": {"dtype": "float32", "shape": [1]}, "index": {"dtype": "int64", "shape": [1]}}}))
    (root / "videos/observation.images.front/chunk-000/file-000.mp4").write_bytes(b"original")


def test_materialize_preserves_original_and_adds_one_synthetic_episode(tmp_path: Path) -> None:
    source, generated = tmp_path / "source", tmp_path / "generated"
    _dataset(source)
    (generated / "variant-1").mkdir(parents=True)
    (generated / "variant-1" / "vision.mp4").write_bytes(b"synthetic")
    storage = FakeStorage(source, generated)
    summary = materialize("s3://test/source/", "s3://test/generated/", "s3://test/output/", "observation.images.front", "0", "s3://test/output/materialization.json", storage_client=storage)
    assert summary["original_episodes"] == 1 and summary["synthetic_episodes"] == 1
    assert storage.uploaded is not None
    data = pa.concat_tables(
        [
            pq.read_table(storage.uploaded / "data/chunk-000/episode_000000.parquet"),
            pq.read_table(storage.uploaded / "data/chunk-000/episode_000001.parquet"),
        ]
    )
    assert data["episode_index"].to_pylist() == [0, 0, 1, 1]
    assert (
        storage.uploaded / "videos/chunk-000/observation.images.front/episode_000001.mp4"
    ).read_bytes() == b"synthetic"
    assert (storage.uploaded / "meta/modality.json").is_file()
    modality_config = (storage.uploaded / "meta/npa_groot_modality_config.py").read_text()
    assert "ActionRepresentation.RELATIVE" not in modality_config
    assert "ActionRepresentation.ABSOLUTE" in modality_config


def _two_episode_dataset(root: Path, *, with_stale_jsonl: bool = False) -> None:
    """A v3 tree with two originals (lengths 2 and 3) and one video file per episode."""

    camera = "observation.images.front"
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "videos" / camera / "chunk-000").mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([0, 0, 1, 1, 1], type=pa.int64()),
                "index": pa.array([0, 1, 2, 3, 4], type=pa.int64()),
                "frame_index": pa.array([0, 1, 0, 1, 2], type=pa.int64()),
                "task_index": pa.array([0] * 5, type=pa.int64()),
                "timestamp": pa.array([0.0, 0.05, 0.0, 0.05, 0.1], type=pa.float32()),
                "observation.state": pa.array([[0.5], [0.6], [0.7], [0.8], [0.9]]),
                "action": pa.array([[1.0], [2.0], [3.0], [4.0], [5.0]]),
            }
        ),
        root / "data/chunk-000/file-000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([0, 1], type=pa.int64()),
                "data/chunk_index": pa.array([0, 0], type=pa.int64()),
                "data/file_index": pa.array([0, 0], type=pa.int64()),
                "dataset_from_index": pa.array([0, 2], type=pa.int64()),
                "dataset_to_index": pa.array([2, 5], type=pa.int64()),
                # Real v3 seeds carry ``length`` and the GR00T adapter trusts it
                # over a row count, so the fixture must carry it too (live run 5).
                "length": pa.array([2, 3], type=pa.int64()),
                f"videos/{camera}/chunk_index": pa.array([0, 0], type=pa.int64()),
                f"videos/{camera}/file_index": pa.array([0, 1], type=pa.int64()),
                f"videos/{camera}/from_timestamp": pa.array([0.0, 0.0]),
                f"videos/{camera}/to_timestamp": pa.array([0.1, 0.15]),
            }
        ),
        root / "meta/episodes/chunk-000/file-000.parquet",
    )
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "fps": 20,
                "total_episodes": 2,
                "total_frames": 5,
                "features": {
                    camera: {"dtype": "video", "shape": [64, 64, 3]},
                    "observation.state": {"dtype": "float32", "shape": [1]},
                    "action": {"dtype": "float32", "shape": [1]},
                    "episode_index": {"dtype": "int64", "shape": [1]},
                    "frame_index": {"dtype": "int64", "shape": [1]},
                    "task_index": {"dtype": "int64", "shape": [1]},
                    "timestamp": {"dtype": "float32", "shape": [1]},
                    "index": {"dtype": "int64", "shape": [1]},
                },
            }
        )
    )
    (root / f"videos/{camera}/chunk-000/file-000.mp4").write_bytes(b"original-0")
    (root / f"videos/{camera}/chunk-000/file-001.mp4").write_bytes(b"original-1")
    if with_stale_jsonl:
        (root / "meta/episodes.jsonl").write_text('{"episode_index": 0, "tasks": [], "length": 2}\n')


def _generated(tmp_path: Path, count: int = 2) -> Path:
    generated = tmp_path / "generated"
    for index in range(1, count + 1):
        (generated / f"variant-{index}").mkdir(parents=True)
        (generated / f"variant-{index}" / "vision.mp4").write_bytes(f"synthetic-{index}".encode())
    return generated


def _materialize(source: Path, generated: Path, heldout: str = "") -> tuple[dict, FakeStorage]:
    storage = FakeStorage(source, generated)
    summary = materialize(
        "s3://test/source/", "s3://test/generated/", "s3://test/output/",
        "observation.images.front", "0", "s3://test/output/materialization.json", heldout,
        storage_client=storage,
    )
    return summary, storage


def test_materialize_records_held_out_and_action_lineage(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _two_episode_dataset(source)
    summary, storage = _materialize(source, _generated(tmp_path), heldout="1")
    assert summary["original_episodes"] == 2 and summary["synthetic_episodes"] == 2
    assert summary["total_episodes"] == 4
    assert summary["augmented_episode_index"] == 0
    assert summary["heldout_episode_index"] == 1
    assert summary["synthetic_episode_indices"] == [2, 3]
    assert summary["action_lineage"] == {"2": 0, "3": 0}
    assert storage.uploaded is not None
    # The GR00T tree carries the metadata prepare-split requires from the adapter.
    assert (storage.uploaded / "meta/npa_groot_adapter.json").is_file()
    assert (storage.uploaded / "meta/npa_groot_modality_config.py").is_file()
    episodes = [json.loads(line) for line in (storage.uploaded / "meta/episodes.jsonl").read_text().splitlines()]
    assert [row["episode_index"] for row in episodes] == [0, 1, 2, 3]
    assert [row["length"] for row in episodes] == [2, 3, 2, 2]  # synthetic copies of episode 0
    # The held-out original's video and the synthetic variants all land per episode.
    videos = storage.uploaded / "videos/chunk-000/observation.images.front"
    assert (videos / "episode_000001.mp4").read_bytes() == b"original-1"
    assert (videos / "episode_000002.mp4").read_bytes() == b"synthetic-1"
    assert (videos / "episode_000003.mp4").read_bytes() == b"synthetic-2"


def test_materialize_without_heldout_keeps_the_legacy_seven_argument_contract(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _two_episode_dataset(source)
    summary, _storage = _materialize(source, _generated(tmp_path, count=1))
    assert summary["heldout_episode_index"] is None
    assert summary["synthetic_episode_indices"] == [2]


def test_materialize_refuses_to_hold_out_the_augmented_episode(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _two_episode_dataset(source)
    with pytest.raises(EncordGrootError, match="must differ from the augmented episode"):
        _materialize(source, _generated(tmp_path), heldout="0")


def test_materialize_refuses_a_missing_held_out_episode(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _two_episode_dataset(source)
    with pytest.raises(EncordGrootError, match="held-out LeRobot episode 7"):
        _materialize(source, _generated(tmp_path), heldout="7")


def test_materialize_refuses_a_single_episode_source_when_holding_out(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _dataset(source)
    with pytest.raises(EncordGrootError, match="at least two original episodes"):
        _materialize(source, _generated(tmp_path), heldout="1")


def test_materialize_refuses_a_stale_episodes_jsonl(tmp_path: Path) -> None:
    # The GR00T adapter prefers meta/episodes.jsonl over the v3 parquet, which
    # would silently drop the synthetic episodes appended to the parquet.
    source = tmp_path / "source"
    _two_episode_dataset(source, with_stale_jsonl=True)
    with pytest.raises(EncordGrootError, match="episodes.jsonl"):
        _materialize(source, _generated(tmp_path), heldout="1")


def _box(kind: bytes, payload: bytes) -> bytes:
    return (8 + len(payload)).to_bytes(4, "big") + kind + payload


def _fake_mp4(frames: int, *, handler: bytes = b"vide") -> bytes:
    """A minimal ISO-BMFF tree: ftyp + moov/trak/mdia/{hdlr, minf/stbl/stsz}."""

    stsz = _box(b"stsz", (0).to_bytes(4, "big") + (0).to_bytes(4, "big") + frames.to_bytes(4, "big"))
    hdlr = _box(b"hdlr", (0).to_bytes(4, "big") + b"\0\0\0\0" + handler + b"\0" * 12)
    mdia = _box(b"mdia", hdlr + _box(b"minf", _box(b"stbl", stsz)))
    return _box(b"ftyp", b"isom\0\0\0\0isom") + _box(b"moov", _box(b"trak", mdia)) + _box(b"mdat", b"\xff" * 16)


def test_mp4_frame_count_walks_the_first_video_track(tmp_path: Path) -> None:
    from npa.workflows.encord_groot_loop import mp4_video_frame_count

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(_fake_mp4(61))
    assert mp4_video_frame_count(clip) == 61
    audio_only = tmp_path / "audio.mp4"
    audio_only.write_bytes(_fake_mp4(61, handler=b"soun"))
    assert mp4_video_frame_count(audio_only) is None
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"synthetic-bytes")
    assert mp4_video_frame_count(junk) is None
    assert mp4_video_frame_count(tmp_path / "missing.mp4") is None


def test_materialize_trims_synthetic_rows_to_the_clip_frame_count(tmp_path: Path) -> None:
    # Episode 0 has 2 action rows; the Cosmos clip for variant 1 has only 1 frame.
    # GR00T indexes frames by row, so the synthetic episode keeps 1 row, and the
    # variant whose container is not parseable keeps the legacy full copy.
    source = tmp_path / "source"
    _two_episode_dataset(source)
    generated = tmp_path / "generated"
    (generated / "variant-1").mkdir(parents=True)
    (generated / "variant-1" / "vision.mp4").write_bytes(_fake_mp4(1))
    (generated / "variant-2").mkdir(parents=True)
    (generated / "variant-2" / "vision.mp4").write_bytes(b"opaque-bytes")
    summary, storage = _materialize(source, generated, heldout="1")
    assert summary["synthetic_frame_counts"] == {"2": 1, "3": None}
    assert summary["synthetic_rows_truncated"] == {"2": 1}
    episodes = [json.loads(line) for line in (storage.uploaded / "meta/episodes.jsonl").read_text().splitlines()]
    assert [(row["episode_index"], row["length"]) for row in episodes] == [(0, 2), (1, 3), (2, 1), (3, 2)]
    table = pq.read_table(storage.uploaded / "data/chunk-000/episode_000002.parquet")
    assert table.num_rows == 1 and table["action"].to_pylist() == [[1.0]]  # leading row kept
    # The v3 episodes metadata the adapter read must agree with the trimmed rows.
    meta = pq.read_table(storage.uploaded / "meta/episodes/chunk-000/file-000.parquet") if (storage.uploaded / "meta/episodes/chunk-000/file-000.parquet").exists() else None
    if meta is not None:
        assert dict(zip(meta["episode_index"].to_pylist(), meta["length"].to_pylist())) == {0: 2, 1: 3, 2: 1, 3: 2}


def _real_mp4(path: Path, *, width: int, height: int, fps: int, frames: int) -> None:
    """Write a genuine encoded video, so ffprobe reports real geometry."""
    import shutil as _shutil
    import subprocess as _subprocess

    ffmpeg = _shutil.which("ffmpeg")
    if ffmpeg is None:  # pragma: no cover - environment without ffmpeg
        pytest.skip("ffmpeg is required to synthesise a real variant")
    path.parent.mkdir(parents=True, exist_ok=True)
    _subprocess.run(
        [
            ffmpeg, "-y", "-f", "lavfi",
            "-i", f"testsrc=size={width}x{height}:rate={fps}:duration={frames / fps}",
            "-pix_fmt", "yuv420p", str(path),
        ],
        check=True, capture_output=True,
    )


def _probe(path: Path) -> tuple[int, int, float]:
    import json as _json
    import subprocess as _subprocess

    out = _subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    stream = (_json.loads(out)["streams"])[0]
    num, _, den = str(stream["r_frame_rate"]).partition("/")
    return int(stream["width"]), int(stream["height"]), float(num) / float(den or 1)


def test_materialize_conforms_a_variant_that_disagrees_with_metadata(
    tmp_path: Path,
) -> None:
    """Cosmos output must be stored at the geometry `info.json` declares.

    Live run cosmos-check-20260909T173508Z copied Cosmos output verbatim into a
    96x96/10fps pusht dataset, so episodes 3 and 4 were stored at 1280x720 and
    24fps. Frame counts matched the declared lengths, so every metadata-only
    check passed while the resolution disagreed by 13x and the action timebase
    was wrong by 2.4x.
    """
    source = tmp_path / "source"
    _two_episode_dataset(source)
    info = json.loads((source / "meta" / "info.json").read_text())
    camera = next(k for k, v in (info.get("features") or {}).items()
                  if (v or {}).get("dtype") == "video")
    shape = info["features"][camera]["shape"]
    declared_h, declared_w = int(shape[0]), int(shape[1])
    declared_fps = float(info.get("fps") or 0)
    assert declared_w and declared_h and declared_fps

    generated = tmp_path / "generated"
    # Deliberately wrong on both axes, exactly as Cosmos emitted.
    _real_mp4(
        generated / "variant-1" / "vision.mp4",
        width=declared_w * 4, height=declared_h * 4,
        fps=int(declared_fps * 2), frames=2,
    )

    _summary, storage = _materialize(source, generated, heldout="1")

    stored = sorted((storage.uploaded / "videos").rglob("*.mp4"))
    assert stored, "no episode videos were written"
    synthetic = stored[-1]
    width, height, fps = _probe(synthetic)
    assert (width, height) == (declared_w, declared_h), (
        f"stored {width}x{height} but metadata declares "
        f"{declared_w}x{declared_h}; a dataset declares one shape per camera"
    )
    assert abs(fps - declared_fps) <= 0.01, (
        f"stored {fps:g}fps against a declared {declared_fps:g}fps; the action "
        "timebase for this episode would be wrong"
    )


def test_materialize_copies_a_variant_that_already_conforms(tmp_path: Path) -> None:
    """The common case must stay a byte copy, with no re-encode."""
    source = tmp_path / "source"
    _two_episode_dataset(source)
    info = json.loads((source / "meta" / "info.json").read_text())
    camera = next(k for k, v in (info.get("features") or {}).items()
                  if (v or {}).get("dtype") == "video")
    shape = info["features"][camera]["shape"]
    declared_h, declared_w = int(shape[0]), int(shape[1])
    declared_fps = float(info.get("fps") or 0)

    generated = tmp_path / "generated"
    variant = generated / "variant-1" / "vision.mp4"
    _real_mp4(
        variant, width=declared_w, height=declared_h,
        fps=int(declared_fps), frames=2,
    )
    original = variant.read_bytes()

    _summary, storage = _materialize(source, generated, heldout="1")

    stored = sorted((storage.uploaded / "videos").rglob("*.mp4"))[-1]
    assert stored.read_bytes() == original, (
        "a conforming variant was re-encoded; the fast path should copy bytes"
    )
