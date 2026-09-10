"""Tests for windowing, generation fan-out, and gated merge of Cosmos augmentations."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from npa.workflows.groot_augment import (
    GrootAugmentError,
    METADATA_NAME,
    VIDEO_NAME,
    _probe_clip,
    _stable_seed,
    generate_augmented_variants,
    materialize_augmented,
    plan_windows,
    stitch_windows,
)


# ---------------------------------------------------------------------------
# plan_windows
# ---------------------------------------------------------------------------


def test_plan_windows_covers_a_short_episode_in_one_window() -> None:
    assert plan_windows(50, 61, 8) == [(0, 50)]


def test_plan_windows_slides_with_overlap_and_clips_the_last_window() -> None:
    windows = plan_windows(100, 61, 8)
    assert windows == [(0, 61), (53, 100)]
    # Consecutive windows overlap by at least the requested amount.
    assert windows[0][1] - windows[1][0] == 8


def test_plan_windows_covers_every_frame_with_no_gap() -> None:
    windows = plan_windows(382, 61, 8)
    assert windows[0][0] == 0
    assert windows[-1][1] == 382
    for (_, prev_end), (next_start, _) in zip(windows, windows[1:]):
        assert next_start <= prev_end  # no gap between consecutive windows


@pytest.mark.parametrize(
    ("total", "window", "overlap"),
    [(0, 61, 8), (-5, 61, 8), (100, 0, 8), (100, 61, 61), (100, 61, 70)],
)
def test_plan_windows_rejects_bad_inputs(total: int, window: int, overlap: int) -> None:
    with pytest.raises(GrootAugmentError):
        plan_windows(total, window, overlap)


def test_stable_seed_is_deterministic_and_varies_by_variant() -> None:
    a = _stable_seed(0, "cam", 0)
    b = _stable_seed(0, "cam", 0)
    c = _stable_seed(0, "cam", 1)
    assert a == b
    assert a != c


# ---------------------------------------------------------------------------
# stitch_windows (real ffmpeg)
# ---------------------------------------------------------------------------


def _solid_clip(path: Path, *, color: str, width: int, height: int, fps: int, frames: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:  # pragma: no cover - environment without ffmpeg
        pytest.skip("ffmpeg is required to synthesise a clip")
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg, "-y", "-f", "lavfi",
            "-i", f"color=c={color}:size={width}x{height}:rate={fps}:duration={frames / fps}",
            "-pix_fmt", "yuv420p", "-frames:v", str(frames), str(path),
        ],
        check=True, capture_output=True,
    )


def test_stitch_windows_copies_a_single_clip_through(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    _solid_clip(clip, color="red", width=32, height=32, fps=10, frames=5)
    output = tmp_path / "out" / "stitched.mp4"
    stitch_windows([clip], overlap_frames=2, output=output)
    assert output.is_file()
    probe = _probe_clip(output)
    assert probe is not None
    assert probe[0] == 5


def test_stitch_windows_joins_two_overlapping_clips(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:  # pragma: no cover
        pytest.skip("ffmpeg is required")
    fps = 10
    # window 0 covers source frames [0, 10), window 1 covers [8, 18) -- a
    # 2-frame overlap, matching how plan_windows(18, 10, 2) would split it.
    clip0 = tmp_path / "w0.mp4"
    clip1 = tmp_path / "w1.mp4"
    _solid_clip(clip0, color="red", width=32, height=32, fps=fps, frames=10)
    _solid_clip(clip1, color="blue", width=32, height=32, fps=fps, frames=10)
    output = tmp_path / "stitched.mp4"
    stitch_windows([clip0, clip1], overlap_frames=2, output=output)
    probe = _probe_clip(output)
    assert probe is not None
    frames, out_fps = probe
    assert out_fps == pytest.approx(fps, rel=0.05)
    # 10 + 10 - 2 overlap = 18 source frames' worth of duration, allowing for
    # xfade's own re-encoding rounding.
    assert 15 <= frames <= 20


def test_stitch_windows_needs_at_least_one_clip(tmp_path: Path) -> None:
    with pytest.raises(GrootAugmentError, match="at least one clip"):
        stitch_windows([], overlap_frames=2, output=tmp_path / "out.mp4")


# ---------------------------------------------------------------------------
# generate_augmented_variants
# ---------------------------------------------------------------------------


class _FakeStorage:
    def __init__(self, dataset: Path, *, keep_dir: Path) -> None:
        self.dataset = dataset
        # Copy "uploads" outside the production function's own temp
        # directory, which it tears down when it returns -- a real S3 upload
        # would not vanish when the caller's local scratch space is cleaned up.
        self.keep_dir = keep_dir
        self.uploaded_dirs: dict[str, Path] = {}
        self.uploaded_files: dict[str, Path] = {}

    def download_directory(self, uri: str, destination: str) -> str:
        assert uri == "s3://test/dataset/"
        shutil.copytree(self.dataset, destination)
        return destination

    def upload_directory(self, local: str, uri: str) -> str:
        dest = self.keep_dir / f"uploaded-{len(self.uploaded_dirs)}"
        shutil.copytree(local, dest)
        self.uploaded_dirs[uri] = dest
        return uri

    def upload_file(self, local: str, uri: str) -> str:
        dest = self.keep_dir / f"uploaded-file-{len(self.uploaded_files)}"
        shutil.copy2(local, dest)
        self.uploaded_files[uri] = dest
        return uri


def _two_camera_dataset(root: Path, *, episode_frames: dict[int, int], fps: int = 10) -> None:
    cameras = ["cam_a", "cam_b"]
    (root / "videos").mkdir(parents=True)
    for camera in cameras:
        chunk_dir = root / "videos" / camera / "chunk-000"
        chunk_dir.mkdir(parents=True)
        for episode, frames in episode_frames.items():
            _solid_clip(
                chunk_dir / f"file-{episode:03d}.mp4",
                color="green", width=32, height=32, fps=fps, frames=frames,
            )


def _fake_generate_fn(**kwargs: object) -> dict[str, object]:
    """Stand in for Cosmos3: copies the window source through as the 'generated' clip.

    Exercises the real slice -> generate -> stitch pipeline without a model or
    GPU, the same way this repo's other GPU-tool tests substitute a fake
    callable rather than mocking ffmpeg itself away.
    """

    input_path = Path(str(kwargs["input_path"]))
    output_path = Path(str(kwargs["output_path"]))
    output_path.mkdir(parents=True, exist_ok=True)
    target = output_path / "vision.mp4"
    shutil.copy2(input_path, target)
    return {"video_path": str(target)}


def test_generate_augmented_variants_fans_out_over_episodes_cameras_and_windows(
    tmp_path: Path,
) -> None:
    if shutil.which("ffmpeg") is None:  # pragma: no cover
        pytest.skip("ffmpeg is required")
    dataset = tmp_path / "dataset"
    _two_camera_dataset(dataset, episode_frames={0: 18, 1: 12})
    keep_dir = tmp_path / "uploads"
    keep_dir.mkdir()
    storage = _FakeStorage(dataset, keep_dir=keep_dir)
    summary = generate_augmented_variants(
        "s3://test/dataset/", "s3://test/variants/", "s3://test/manifest.json",
        cameras=["cam_a", "cam_b"], episode_indices=[0, 1], augmentation_count=2,
        window_frames=10, overlap_frames=2, prompt="a test prompt",
        storage_client=storage, generate_fn=_fake_generate_fn,
    )
    # 2 episodes x 2 cameras x 2 variants = 8 clips.
    assert len(summary["clips"]) == 8
    clip_ids = {clip["clip_id"] for clip in summary["clips"]}
    assert "episode-000-camera-cam_a-variant-00" in clip_ids
    assert "episode-001-camera-cam_b-variant-01" in clip_ids
    # Episode 0 is 18 frames -> 2 windows at (10, 2); episode 1 is 12 -> 2 windows.
    by_id = {clip["clip_id"]: clip for clip in summary["clips"]}
    assert by_id["episode-000-camera-cam_a-variant-00"]["window_count"] == 2
    assert by_id["episode-001-camera-cam_a-variant-00"]["window_count"] == 2
    # Every clip directory was uploaded with the Evaluator-compatible layout.
    assert len(storage.uploaded_dirs) == 8
    for uploaded in storage.uploaded_dirs.values():
        assert (uploaded / VIDEO_NAME).is_file()
        metadata = json.loads((uploaded / METADATA_NAME).read_text())
        assert metadata["prompt"] == "a test prompt"
        assert "inference_seed" in metadata
    # The same seed is reused across every window of one variant (checked
    # indirectly: two variants of the same episode/camera get different seeds).
    seed_00 = by_id["episode-000-camera-cam_a-variant-00"]["seed"]
    seed_01 = by_id["episode-000-camera-cam_a-variant-01"]["seed"]
    assert seed_00 != seed_01


def test_generate_augmented_variants_resolves_empty_cameras_and_episodes(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:  # pragma: no cover
        pytest.skip("ffmpeg is required")
    dataset = tmp_path / "dataset"
    _two_camera_dataset(dataset, episode_frames={0: 12, 1: 12})
    # generate_augmented_variants reads meta/info.json for declared video
    # cameras and meta/episodes/**/*.parquet for the episode list when either
    # is left empty -- the "augment every train episode/camera" default the
    # spec documents.
    (dataset / "meta").mkdir()
    (dataset / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 10,
                "features": {
                    "cam_a": {"dtype": "video", "shape": [32, 32, 3]},
                    "cam_b": {"dtype": "video", "shape": [32, 32, 3]},
                    "action": {"dtype": "float32", "shape": [1]},
                },
            }
        )
    )
    (dataset / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    pq.write_table(
        pa.table({"episode_index": pa.array([0, 1], type=pa.int64())}),
        dataset / "meta/episodes/chunk-000/file-000.parquet",
    )
    keep_dir = tmp_path / "uploads"
    keep_dir.mkdir()
    storage = _FakeStorage(dataset, keep_dir=keep_dir)
    summary = generate_augmented_variants(
        "s3://test/dataset/", "s3://test/variants/", "s3://test/manifest.json",
        cameras=[], episode_indices=[], augmentation_count=1,
        window_frames=10, overlap_frames=2, prompt="p",
        storage_client=storage, generate_fn=_fake_generate_fn,
    )
    assert summary["cameras"] == ["cam_a", "cam_b"]
    assert summary["episode_indices"] == [0, 1]
    assert len(summary["clips"]) == 4  # 2 episodes x 2 cameras x 1 variant


def test_generate_augmented_variants_requires_positive_count(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    keep_dir = tmp_path / "uploads"
    keep_dir.mkdir()
    with pytest.raises(GrootAugmentError, match="augmentation_count must be positive"):
        generate_augmented_variants(
            "s3://test/dataset/", "s3://test/variants/", "s3://test/manifest.json",
            cameras=["cam_a"], episode_indices=[0], augmentation_count=0,
            window_frames=10, overlap_frames=2, prompt="p",
            storage_client=_FakeStorage(dataset, keep_dir=keep_dir), generate_fn=_fake_generate_fn,
        )


# ---------------------------------------------------------------------------
# materialize_augmented
# ---------------------------------------------------------------------------


class _MergeFakeStorage:
    def __init__(self, dataset: Path, variants: Path, report: Path, generation: Path) -> None:
        self.dataset, self.variants = dataset, variants
        self.report, self.generation = report, generation
        self.uploaded: Path | None = None

    def download_directory(self, uri: str, destination: str) -> str:
        source = {"s3://test/dataset/": self.dataset, "s3://test/variants/": self.variants}[uri]
        shutil.copytree(source, destination)
        return destination

    def download_file(self, uri: str, destination: str) -> str:
        source = {
            "s3://test/report.json": self.report,
            "s3://test/generation.json": self.generation,
        }[uri]
        shutil.copy2(source, destination)
        return destination

    def upload_directory(self, local: str, uri: str) -> str:
        self.uploaded = self.variants.parent / "merged-output"
        shutil.copytree(local, self.uploaded)
        return uri

    def upload_file(self, local: str, uri: str) -> str:
        return uri


def _train_dataset(root: Path) -> None:
    camera = "cam_a"
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "videos" / camera / "chunk-000").mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([0, 0, 1, 1], type=pa.int64()),
                "index": pa.array([0, 1, 2, 3], type=pa.int64()),
                "frame_index": pa.array([0, 1, 0, 1], type=pa.int64()),
                "task_index": pa.array([0, 0, 0, 0], type=pa.int64()),
                "timestamp": pa.array([0.0, 0.1, 0.0, 0.1], type=pa.float32()),
                "observation.state": pa.array([[0.1], [0.2], [0.3], [0.4]]),
                "action": pa.array([[1.0], [2.0], [3.0], [4.0]]),
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
                "dataset_to_index": pa.array([2, 4], type=pa.int64()),
                "length": pa.array([2, 2], type=pa.int64()),
                f"videos/{camera}/chunk_index": pa.array([0, 0], type=pa.int64()),
                f"videos/{camera}/file_index": pa.array([0, 1], type=pa.int64()),
                f"videos/{camera}/from_timestamp": pa.array([0.0, 0.0]),
                f"videos/{camera}/to_timestamp": pa.array([0.2, 0.2]),
            }
        ),
        root / "meta/episodes/chunk-000/file-000.parquet",
    )
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "fps": 10,
                "total_episodes": 2,
                "total_frames": 4,
                "features": {
                    camera: {"dtype": "video", "shape": [32, 32, 3]},
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
    for episode in (0, 1):
        _solid_clip(
            root / f"videos/{camera}/chunk-000/file-{episode:03d}.mp4",
            color="green", width=32, height=32, fps=10, frames=2,
        )
    (root / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "Pick up the block."}) + "\n"
    )


def _variant_clip_dir(root: Path, clip_id: str, *, frames: int, width: int = 32, height: int = 32) -> None:
    clip_dir = root / clip_id
    clip_dir.mkdir(parents=True)
    _solid_clip(clip_dir / VIDEO_NAME, color="blue", width=width, height=height, fps=10, frames=frames)
    (clip_dir / METADATA_NAME).write_text(json.dumps({"prompt": "p", "inference_seed": "1"}))


def test_materialize_augmented_merges_only_gate_passing_variants(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:  # pragma: no cover
        pytest.skip("ffmpeg is required")
    dataset = tmp_path / "dataset"
    _train_dataset(dataset)
    variants = tmp_path / "variants"
    _variant_clip_dir(variants, "episode-000-camera-cam_a-variant-00", frames=2)
    _variant_clip_dir(variants, "episode-001-camera-cam_a-variant-00", frames=2)
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "clips": [
                    {"clip_id": "episode-000-camera-cam_a-variant-00", "passed": True},
                    {"clip_id": "episode-001-camera-cam_a-variant-00", "passed": False},
                ]
            }
        )
    )
    generation = tmp_path / "generation.json"
    generation.write_text(
        json.dumps(
            {
                "clips": [
                    {"clip_id": "episode-000-camera-cam_a-variant-00", "episode_index": 0, "camera": "cam_a"},
                    {"clip_id": "episode-001-camera-cam_a-variant-00", "episode_index": 1, "camera": "cam_a"},
                ]
            }
        )
    )
    storage = _MergeFakeStorage(dataset, variants, report, generation)
    summary = materialize_augmented(
        "s3://test/dataset/", "s3://test/variants/", "s3://test/report.json",
        "s3://test/generation.json", "s3://test/output/", "s3://test/manifest.json",
        storage_client=storage,
    )
    assert summary["original_episodes"] == 2
    assert summary["synthetic_episodes"] == 1
    assert summary["included_clips"] == ["episode-000-camera-cam_a-variant-00"]
    assert summary["rejected_clips"] == ["episode-001-camera-cam_a-variant-00"]
    assert summary["action_lineage"] == {"2": 0}
    assert storage.uploaded is not None
    data = pq.read_table(storage.uploaded / "data/chunk-000/file-000.parquet")
    assert pc.max(data["episode_index"]).as_py() == 2
    # The synthetic episode copies episode 0's action rows verbatim.
    synthetic_rows = data.filter(pc.equal(data["episode_index"], 2))
    assert synthetic_rows["action"].to_pylist() == [[1.0], [2.0]]
    assert (storage.uploaded / "videos/cam_a/chunk-000/file-002.mp4").is_file()


def test_materialize_augmented_refuses_when_nothing_passed_the_gate(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:  # pragma: no cover
        pytest.skip("ffmpeg is required")
    dataset = tmp_path / "dataset"
    _train_dataset(dataset)
    variants = tmp_path / "variants"
    _variant_clip_dir(variants, "episode-000-camera-cam_a-variant-00", frames=2)
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"clips": [{"clip_id": "episode-000-camera-cam_a-variant-00", "passed": False}]})
    )
    generation = tmp_path / "generation.json"
    generation.write_text(json.dumps({"clips": []}))
    storage = _MergeFakeStorage(dataset, variants, report, generation)
    with pytest.raises(GrootAugmentError, match="no augmentation variant passed"):
        materialize_augmented(
            "s3://test/dataset/", "s3://test/variants/", "s3://test/report.json",
            "s3://test/generation.json", "s3://test/output/", "s3://test/manifest.json",
            storage_client=storage,
        )
