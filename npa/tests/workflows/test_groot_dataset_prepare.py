"""Dataset preparation for the Encord to GR00T fine-tuning path.

Three contracts are under test:

* ``audit_dataset`` answers "is my data usable?" and fails closed on the
  defects no optimizer can recover from, before curation or GPU time.
* ``curated_episode_ids`` turns a verified Encord pull into an episode
  allowlist, refusing to guess when an item cannot be attributed.
* ``prepare_split`` honours that allowlist, and records the curation in both
  the published manifest and the split identity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from npa.adapter.groot import (
    DATASET_AUDIT,
    DATASET_AUDIT_SCHEMA,
    GR00TAdapterError,
    audit_dataset,
    lerobot_to_groot,
)
from npa.workflows.groot_learning import (
    GrootVisualizationError,
    curated_episode_ids,
    deterministic_episode_split,
    deterministic_experiment_split,
    prepare_dataset,
    prepare_split,
)

PULL_SCHEMA = "npa.encord.pull_manifest.v1"
REPORT_SCHEMA = "npa.encord.roundtrip_report.v1"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


CAMERA = "observation.images.top"


def _v3_dataset(
    root: Path,
    *,
    episodes: int = 4,
    frames: int = 3,
    fps: int = 20,
    state_dim: int = 3,
    constant_action_dim: bool = False,
    timestamps: list[float] | None = None,
    cameras: bool = False,
) -> Path:
    """Write a standard LeRobot v3.0 dataset with one packed data file.

    The action's last dimension is deliberately varied unless
    ``constant_action_dim`` asks for the stuck-gripper case. ``cameras`` adds
    one video file per episode, which is the already-unpacked shape a
    per-episode recorder produces and which needs no ffmpeg to convert.
    """

    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    total = episodes * frames
    states: list[list[float]] = []
    actions: list[list[float]] = []
    stamps: list[float] = []
    for episode in range(episodes):
        for frame in range(frames):
            base = float(episode * frames + frame)
            states.append([base + offset for offset in range(state_dim)])
            last = 0.0 if constant_action_dim else float(frame % 2)
            actions.append([base * 0.1, base * 0.2, last])
            stamps.append(timestamps[frame] if timestamps is not None else frame / fps)
    data = pa.table(
        {
            "observation.state": pa.array(
                states, type=pa.list_(pa.float32(), state_dim)
            ),
            "action": pa.array(actions, type=pa.list_(pa.float32(), 3)),
            "episode_index": pa.array(
                [episode for episode in range(episodes) for _ in range(frames)],
                type=pa.int64(),
            ),
            "frame_index": pa.array(
                [frame for _ in range(episodes) for frame in range(frames)],
                type=pa.int64(),
            ),
            "timestamp": pa.array(stamps, type=pa.float32()),
            "index": pa.array(list(range(total)), type=pa.int64()),
            "task_index": pa.array([0] * total, type=pa.int64()),
        }
    )
    pq.write_table(data, root / "data" / "chunk-000" / "file-000.parquet")

    episode_columns: dict[str, Any] = {
        "episode_index": pa.array(list(range(episodes)), type=pa.int64()),
        "data/chunk_index": pa.array([0] * episodes, type=pa.int64()),
        "data/file_index": pa.array([0] * episodes, type=pa.int64()),
        "dataset_from_index": pa.array(
            [episode * frames for episode in range(episodes)], type=pa.int64()
        ),
        "dataset_to_index": pa.array(
            [(episode + 1) * frames for episode in range(episodes)],
            type=pa.int64(),
        ),
        "length": pa.array([frames] * episodes, type=pa.int64()),
        "tasks": pa.array([["pick"]] * episodes),
    }
    if cameras:
        # One video file per episode and no from/to timestamps: nothing is
        # packed, so conversion copies rather than cutting.
        episode_columns[f"videos/{CAMERA}/chunk_index"] = pa.array(
            [0] * episodes, type=pa.int64()
        )
        episode_columns[f"videos/{CAMERA}/file_index"] = pa.array(
            list(range(episodes)), type=pa.int64()
        )
        for episode in range(episodes):
            video = root / "videos" / CAMERA / "chunk-000" / f"file-{episode:03d}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(f"episode-{episode}-video".encode())
    pq.write_table(
        pa.table(episode_columns),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array([0], type=pa.int64()),
                "task": pa.array(["pick"], type=pa.string()),
            }
        ),
        root / "meta" / "tasks.parquet",
    )
    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": [state_dim], "names": None},
        "action": {
            "dtype": "float32",
            "shape": [3],
            "names": ["x", "y", "gripper"],
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    info: dict[str, Any] = {
        "codebase_version": "v3.0",
        "robot_type": "testbot",
        "total_episodes": episodes,
        "total_frames": total,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "features": features,
    }
    if cameras:
        features[CAMERA] = {
            "dtype": "video",
            "shape": [48, 64, 3],
            "names": ["height", "width", "channel"],
            "info": {"video.fps": float(fps)},
        }
        info["video_path"] = (
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        )
    _write_json(root / "meta" / "info.json", info)
    _write_json(root / "meta" / "stats.json", {})
    return root


def _converted(tmp_path: Path, **kwargs: Any) -> Path:
    source = _v3_dataset(tmp_path / "source", **kwargs)
    return lerobot_to_groot(
        source, tmp_path / "groot", robot_embodiment="NEW_EMBODIMENT"
    )


# --------------------------------------------------------------------------
# audit_dataset
# --------------------------------------------------------------------------


def test_conversion_publishes_an_audit_of_the_dataset_it_wrote(tmp_path: Path) -> None:
    out = _converted(tmp_path, episodes=4, frames=3, fps=20)

    audit = json.loads((out / "meta" / DATASET_AUDIT).read_text())
    assert audit["schema"] == DATASET_AUDIT_SCHEMA
    assert audit["dataset"]["episodes"] == 4
    assert audit["dataset"]["frames"] == 12
    assert audit["dataset"]["fps"] == 20.0
    assert audit["dataset"]["codebase_version"] == "v2.1"
    assert audit["dataset"]["duration_seconds"] == pytest.approx(0.6)
    assert audit["tasks"] == ["pick"]
    assert [episode["episode_index"] for episode in audit["episodes_detail"]] == [
        0,
        1,
        2,
        3,
    ]
    assert all(episode["frames"] == 3 for episode in audit["episodes_detail"])
    assert all(check["status"] != "failed" for check in audit["checks"])


def test_audit_reports_per_dimension_ranges_a_company_can_read(tmp_path: Path) -> None:
    out = _converted(tmp_path, episodes=2, frames=2, state_dim=3)

    audit = audit_dataset(out)
    state = audit["tensors"]["observation.state"]
    assert state["dimensions"] == 3
    assert state["samples"] == 4
    # states are base + offset for base in 0..3, so dim 0 spans 0..3.
    assert state["min"][0] == pytest.approx(0.0)
    assert state["max"][0] == pytest.approx(3.0)
    assert state["mean"][0] == pytest.approx(1.5)
    assert state["std"][0] > 0
    assert audit["tensors"]["action"]["dimensions"] == 3


def test_audit_flags_a_dimension_that_never_moves(tmp_path: Path) -> None:
    out = _converted(tmp_path, constant_action_dim=True)

    audit = audit_dataset(out)
    assert audit["tensors"]["action"]["constant_dimensions"] == [2]
    constant = [
        advisory
        for advisory in audit["advisories"]
        if advisory["finding"] == "constant_dimension"
    ]
    assert len(constant) == 1
    # The advisory names the joint, not just its index.
    assert "gripper" in constant[0]["detail"]


def test_audit_flags_an_irregular_timebase_without_blocking(tmp_path: Path) -> None:
    out = _converted(tmp_path, frames=3, fps=20, timestamps=[0.0, 0.05, 0.9])

    audit = audit_dataset(out)
    assert [advisory["finding"] for advisory in audit["advisories"]] == [
        "irregular_timebase"
    ]
    assert audit["dataset"]["episodes"] == 4


def test_audit_records_camera_coverage(tmp_path: Path) -> None:
    source = _v3_dataset(tmp_path / "source", episodes=2, frames=2)
    info_path = source / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["video_path"] = (
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    info["features"]["observation.images.top"] = {
        "dtype": "video",
        "shape": [48, 64, 3],
        "names": ["height", "width", "channel"],
    }
    _write_json(info_path, info)
    episodes_path = source / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes = pq.read_table(episodes_path)
    for column, values in (
        ("videos/observation.images.top/chunk_index", [0, 0]),
        ("videos/observation.images.top/file_index", [0, 0]),
    ):
        episodes = episodes.append_column(column, pa.array(values, type=pa.int64()))
    episodes = episodes.append_column(
        "videos/observation.images.top/from_timestamp",
        pa.array([0.0, 0.1], type=pa.float64()),
    )
    episodes = episodes.append_column(
        "videos/observation.images.top/to_timestamp",
        pa.array([0.1, 0.2], type=pa.float64()),
    )
    pq.write_table(episodes, episodes_path)
    packed = source / "videos" / "observation.images.top" / "chunk-000" / "file-000.mp4"
    packed.parent.mkdir(parents=True)
    packed.write_bytes(b"packed-two-episode-video")

    def fake_run(command: list[str], **_kwargs: object) -> None:
        Path(command[-1]).write_bytes(b"episode-video")

    import npa.adapter.groot as adapter

    original_which = adapter.shutil.which
    original_run = adapter.subprocess.run
    adapter.shutil.which = lambda name: f"/usr/bin/{name}"  # type: ignore[assignment]
    adapter.subprocess.run = fake_run  # type: ignore[assignment]
    try:
        out = lerobot_to_groot(
            source, tmp_path / "groot", robot_embodiment="NEW_EMBODIMENT"
        )
    finally:
        adapter.shutil.which = original_which  # type: ignore[assignment]
        adapter.subprocess.run = original_run  # type: ignore[assignment]

    audit = json.loads((out / "meta" / DATASET_AUDIT).read_text())
    cameras = {camera["original_key"]: camera for camera in audit["cameras"]}
    top = cameras["observation.images.top"]
    assert top["episodes_with_video"] == 2
    assert top["resolution"] == "64x48"
    assert top["bytes"] == 2 * len(b"episode-video")
    assert all(
        episode["cameras"] == ["observation.images.top"]
        for episode in audit["episodes_detail"]
    )


@pytest.mark.parametrize(
    "mutate, expected",
    [
        pytest.param(
            lambda root: _corrupt_action(root, float("nan")),
            "non-finite",
            id="non-finite-action",
        ),
        pytest.param(
            lambda root: _corrupt_declared_length(root),
            "declares 99 frames",
            id="length-mismatch",
        ),
        pytest.param(
            lambda root: _corrupt_timestamps(root),
            "do not increase",
            id="non-monotonic-timestamps",
        ),
        pytest.param(
            lambda root: _corrupt_fps(root),
            "no positive fps",
            id="zero-fps",
        ),
        pytest.param(
            lambda root: _corrupt_declared_dim(root),
            "metadata declares 7",
            id="declared-dimension-mismatch",
        ),
        pytest.param(
            lambda root: _remove_episode_data(root),
            "has no data file",
            id="missing-episode-data",
        ),
    ],
)
def test_audit_fails_closed_on_data_no_optimizer_can_recover_from(
    tmp_path: Path, mutate: Any, expected: str
) -> None:
    out = _converted(tmp_path, episodes=2, frames=3)
    mutate(out)

    with pytest.raises(GR00TAdapterError) as excinfo:
        audit_dataset(out)
    assert expected in str(excinfo.value)


def _episode_parquet(root: Path, episode: int = 0) -> Path:
    return root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"


def _corrupt_action(root: Path, value: float) -> None:
    path = _episode_parquet(root)
    table = pq.read_table(path)
    actions = table["action"].to_pylist()
    actions[1] = [value, 0.0, 0.0]
    position = table.schema.get_field_index("action")
    table = table.set_column(
        position, "action", pa.array(actions, type=pa.list_(pa.float32(), 3))
    )
    pq.write_table(table, path)


def _corrupt_declared_length(root: Path) -> None:
    path = root / "meta" / "episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows[0]["length"] = 99
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _corrupt_timestamps(root: Path) -> None:
    path = _episode_parquet(root)
    table = pq.read_table(path)
    position = table.schema.get_field_index("timestamp")
    table = table.set_column(
        position,
        "timestamp",
        pa.array([0.0, 0.2, 0.1], type=pa.float32()),
    )
    pq.write_table(table, path)


def _corrupt_fps(root: Path) -> None:
    path = root / "meta" / "info.json"
    info = json.loads(path.read_text())
    info["fps"] = 0
    _write_json(path, info)


def _corrupt_declared_dim(root: Path) -> None:
    path = root / "meta" / "info.json"
    info = json.loads(path.read_text())
    info["features"]["action"]["shape"] = [7]
    _write_json(path, info)


def _remove_episode_data(root: Path) -> None:
    _episode_parquet(root, 1).unlink()


def test_audit_requires_a_state_and_action_feature(tmp_path: Path) -> None:
    out = _converted(tmp_path, episodes=2, frames=2)
    info_path = out / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    del info["features"]["action"]
    _write_json(info_path, info)

    with pytest.raises(GR00TAdapterError, match="declares no 'action' feature"):
        audit_dataset(out)


def test_audit_fails_closed_when_a_declared_camera_has_no_bytes(tmp_path: Path) -> None:
    out = _converted(tmp_path, episodes=2, frames=2)
    info_path = out / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.images.top"] = {
        "dtype": "video",
        "shape": [48, 64, 3],
        "names": ["height", "width", "channel"],
    }
    _write_json(info_path, info)

    with pytest.raises(GR00TAdapterError, match="no video bytes"):
        audit_dataset(out)


# --------------------------------------------------------------------------
# curated_episode_ids
# --------------------------------------------------------------------------


def _item(
    episode: int, camera: str = "observation.images.top", **overrides: Any
) -> dict[str, Any]:
    name = f"episode_{episode:06d}.mp4"
    item = {
        "item_uuid": f"uuid-{camera}-{episode}",
        "name": name,
        "source_uri": f"s3://bucket/prepared/videos/chunk-000/{camera}/{name}",
    }
    item.update(overrides)
    return item


def test_curation_resolves_episodes_across_every_pushed_camera() -> None:
    manifest = {
        "schema": PULL_SCHEMA,
        "items": [
            _item(3),
            _item(7),
            _item(3, camera="observation.images.wrist"),
        ],
    }

    resolved = curated_episode_ids(manifest)
    assert resolved["episode_ids"] == [3, 7]
    assert resolved["items_total"] == 3
    assert {item["attributed_by"] for item in resolved["items"]} == {"source_uri"}


def test_curation_accepts_a_passing_roundtrip_report() -> None:
    manifest = {"schema": PULL_SCHEMA, "items": [_item(1)]}
    report = {"schema": REPORT_SCHEMA, "status": "passed"}

    assert curated_episode_ids(manifest, report=report)["episode_ids"] == [1]


def test_curation_falls_back_to_the_item_name_when_identity_is_absent() -> None:
    manifest = {
        "schema": PULL_SCHEMA,
        "items": [_item(5, source_uri="")],
    }

    resolved = curated_episode_ids(manifest)
    assert resolved["episode_ids"] == [5]
    assert resolved["items"][0]["attributed_by"] == "name"


@pytest.mark.parametrize(
    "manifest, report, expected",
    [
        pytest.param(
            {"schema": "npa.encord.push_receipt.v1", "items": [_item(0)]},
            None,
            "not a npa.encord.pull_manifest.v1",
            id="wrong-artifact",
        ),
        pytest.param(
            {"schema": PULL_SCHEMA, "items": []},
            None,
            "selected no items",
            id="zero-selection",
        ),
        pytest.param(
            {
                "schema": PULL_SCHEMA,
                "items": [{"name": "clip.mp4", "source_uri": "s3://b/clip.mp4"}],
            },
            None,
            "cannot be attributed",
            id="unattributable-item",
        ),
        pytest.param(
            {"schema": PULL_SCHEMA, "items": [_item(0, error="download failed")]},
            None,
            "failed: download failed",
            id="errored-item",
        ),
        pytest.param(
            {"schema": PULL_SCHEMA, "items": [_item(0)]},
            {"schema": REPORT_SCHEMA, "status": "failed"},
            "did not pass",
            id="failed-roundtrip",
        ),
        pytest.param(
            {"schema": PULL_SCHEMA, "items": [_item(0)]},
            {"schema": "npa.encord.pull_manifest.v1", "status": "passed"},
            "not a npa.encord.roundtrip_report.v1",
            id="wrong-report-artifact",
        ),
    ],
)
def test_curation_fails_closed_rather_than_guessing(
    manifest: dict[str, Any], report: dict[str, Any] | None, expected: str
) -> None:
    with pytest.raises(GrootVisualizationError) as excinfo:
        curated_episode_ids(manifest, report=report)
    assert expected in str(excinfo.value)


# --------------------------------------------------------------------------
# split candidates
# --------------------------------------------------------------------------


def test_split_without_an_allowlist_is_unchanged() -> None:
    explicit = deterministic_episode_split(
        10, train_episodes=4, heldout_episodes=2, seed="s", candidates=None
    )
    default = deterministic_episode_split(
        10, train_episodes=4, heldout_episodes=2, seed="s"
    )
    assert explicit == default
    assert len(default["train"]) == 4
    assert not set(default["train"]) & set(default["heldout"])


def test_split_draws_only_from_the_allowlist() -> None:
    eligible = [1, 3, 5, 7, 9]
    split = deterministic_episode_split(
        10, train_episodes=3, heldout_episodes=2, seed="s", candidates=eligible
    )
    assert set(split["train"]) | set(split["heldout"]) <= set(eligible)
    assert not set(split["train"]) & set(split["heldout"])

    experiment = deterministic_experiment_split(
        10,
        train_episodes=2,
        validation_episodes=1,
        final_episodes=1,
        seed="s",
        candidates=[0, 2, 4, 6],
    )
    selected = experiment["train"] + experiment["validation"] + experiment["final"]
    assert set(selected) <= {0, 2, 4, 6}
    assert len(selected) == len(set(selected))


@pytest.mark.parametrize(
    "candidates, expected",
    [
        pytest.param([1, 1, 2], "duplicates", id="duplicates"),
        pytest.param([2, 99], "outside the dataset", id="out-of-range"),
        pytest.param([-1, 2], "outside the dataset", id="negative"),
        pytest.param([], "no episodes are eligible", id="empty"),
        pytest.param([1, 2], "too few episodes", id="too-few"),
    ],
)
def test_split_rejects_an_unusable_allowlist(
    candidates: list[int], expected: str
) -> None:
    with pytest.raises(GrootVisualizationError) as excinfo:
        deterministic_episode_split(
            10, train_episodes=3, heldout_episodes=2, seed="s", candidates=candidates
        )
    assert expected in str(excinfo.value)


# --------------------------------------------------------------------------
# prepare_split honouring curation
# --------------------------------------------------------------------------


class _Body:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body


class FakeS3:
    """Enough of the boto3 surface for the split's read/write helpers."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def seed_directory(self, root: Path, bucket: str, prefix: str) -> None:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                key = f"{prefix.strip('/')}/{path.relative_to(root).as_posix()}"
                self.objects[(bucket, key)] = path.read_bytes()

    def seed_json(self, bucket: str, key: str, payload: dict[str, Any]) -> None:
        self.objects[(bucket, key)] = json.dumps(payload).encode()

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        if (Bucket, Key) not in self.objects:
            raise KeyError(f"missing s3://{Bucket}/{Key}")
        return {"Body": _Body(self.objects[(Bucket, Key)])}

    def put_object(
        self, *, Bucket: str, Key: str, Body: bytes, ContentType: str = ""
    ) -> dict[str, Any]:
        self.objects[(Bucket, Key)] = Body
        return {}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        body = self.objects[(Bucket, Key)]
        return {
            "ContentLength": len(body),
            "ETag": f'"{hashlib.md5(body).hexdigest()}"',
        }

    def list_objects_v2(
        self, *, Bucket: str, Prefix: str = "", **_kwargs: Any
    ) -> dict[str, Any]:
        contents = [
            {
                "Key": key,
                "Size": len(body),
                "ETag": f'"{hashlib.md5(body).hexdigest()}"',
            }
            for (bucket, key), body in sorted(self.objects.items())
            if bucket == Bucket and key.startswith(Prefix)
        ]
        return {"Contents": contents, "IsTruncated": False}

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        target = Path(filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.objects[(bucket, key)])


@pytest.fixture()
def curated_source(tmp_path: Path) -> tuple[FakeS3, str]:
    """A converted 6-episode dataset in fake S3, plus a curation of four of them."""

    out = _converted(tmp_path, episodes=6, frames=4, cameras=True)
    client = FakeS3()
    client.seed_directory(out, "bucket", "prepared")
    client.seed_json(
        "bucket",
        "curate/manifest.json",
        {
            "schema": PULL_SCHEMA,
            "items": [_item(episode) for episode in (0, 2, 3, 5)],
        },
    )
    client.seed_json(
        "bucket",
        "curate/roundtrip_report.json",
        {"schema": REPORT_SCHEMA, "status": "passed"},
    )
    return client, "s3://bucket/prepared"


def _split(client: FakeS3, source: str, **kwargs: Any) -> dict[str, Any]:
    return prepare_split(
        source,
        "s3://bucket/data/train/",
        "s3://bucket/data/validation/",
        "s3://bucket/reports/split.json",
        "run-1",
        train_episodes=2,
        heldout_episodes=1,
        final_uri="s3://bucket/data/final/",
        final_episodes=1,
        seed="fixed-seed",
        global_batch_size=2,
        max_steps=2,
        minimum_epochs=0.001,
        s3_client=client,
        **kwargs,
    )


def test_prepare_split_trains_only_on_curated_episodes(
    curated_source: tuple[FakeS3, str],
) -> None:
    client, source = curated_source

    result = _split(
        client,
        source,
        curation_manifest_uri="s3://bucket/curate/manifest.json",
        curation_report_uri="s3://bucket/curate/roundtrip_report.json",
    )

    selection = result["selection"]
    assert selection["source"] == "encord-curation"
    assert selection["eligible_source_episode_ids"] == [0, 2, 3, 5]
    assert selection["excluded_by_curation"] == [1, 4]
    assert selection["roundtrip_verified"] is True
    assert len(selection["encord_items"]) == 4

    used = (
        result["train"]["source_episode_ids"]
        + result["heldout"]["source_episode_ids"]
        + result["final"]["source_episode_ids"]
    )
    assert set(used) <= {0, 2, 3, 5}
    assert 1 not in used and 4 not in used
    assert result["integrity"]["leakage_free"] is True


def test_prepare_split_without_curation_uses_the_whole_dataset(
    curated_source: tuple[FakeS3, str],
) -> None:
    client, source = curated_source

    result = _split(client, source)

    assert result["selection"]["source"] == "whole-dataset"
    assert result["selection"]["eligible_source_episode_ids"] == [0, 1, 2, 3, 4, 5]
    assert result["selection"]["excluded_by_curation"] == []
    assert result["selection"]["encord_items"] == []


def test_curation_changes_the_split_identity(
    curated_source: tuple[FakeS3, str],
) -> None:
    """Same seed plus different curation is a different experiment."""

    client, source = curated_source

    uncurated = _split(client, source)["split_hash"]
    curated = _split(
        client,
        source,
        curation_manifest_uri="s3://bucket/curate/manifest.json",
    )["split_hash"]

    assert uncurated != curated


def test_prepare_split_rejects_a_report_without_its_manifest(
    curated_source: tuple[FakeS3, str],
) -> None:
    client, source = curated_source

    with pytest.raises(GrootVisualizationError, match="needs the curation manifest"):
        _split(
            client,
            source,
            curation_report_uri="s3://bucket/curate/roundtrip_report.json",
        )


def test_prepare_split_fails_closed_when_curation_leaves_too_few_episodes(
    curated_source: tuple[FakeS3, str],
) -> None:
    client, source = curated_source
    client.seed_json(
        "bucket",
        "curate/thin.json",
        {"schema": PULL_SCHEMA, "items": [_item(2)]},
    )

    with pytest.raises(GrootVisualizationError, match="too few episodes"):
        _split(client, source, curation_manifest_uri="s3://bucket/curate/thin.json")


# --------------------------------------------------------------------------
# prepare_dataset (the workflow stage that runs before Encord push)
# --------------------------------------------------------------------------


def test_prepare_dataset_publishes_per_episode_media_and_an_audit(
    tmp_path: Path,
) -> None:
    """Encord curates media items, so each episode needs its own video."""

    source = _v3_dataset(tmp_path / "raw", episodes=3, frames=4, cameras=True)
    client = FakeS3()
    client.seed_directory(source, "bucket", "datasets/raw")

    result = prepare_dataset(
        "s3://bucket/datasets/raw",
        "s3://bucket/run/prepared",
        "s3://bucket/run/reports/dataset-audit.json",
        "run-1",
        robot_embodiment="NEW_EMBODIMENT",
        s3_client=client,
    )

    assert result["schema"] == "npa.groot.dataset_prepare.v1"
    assert result["status"] == "prepared"
    assert result["dataset"]["episodes"] == 3
    assert result["dataset"]["frames"] == 12
    assert result["tasks"] == ["pick"]
    assert result["objects_uploaded"] > 0

    keys = {key for (_bucket, key) in client.objects}
    # One video per episode, named by its source episode id: that filename is
    # what makes a curated Encord item attributable back to an episode.
    for episode in range(3):
        assert (
            f"run/prepared/videos/chunk-000/{CAMERA}/episode_{episode:06d}.mp4" in keys
        )
        assert f"run/prepared/data/chunk-000/episode_{episode:06d}.parquet" in keys
    # The embodiment contract the split and trainer both require.
    assert "run/prepared/meta/modality.json" in keys
    assert "run/prepared/meta/npa_groot_modality_config.py" in keys
    assert "run/prepared/meta/episodes.jsonl" in keys
    assert "run/prepared/meta/npa_dataset_audit.json" in keys

    published = json.loads(client.objects[("bucket", "run/reports/dataset-audit.json")])
    assert published["schema"] == DATASET_AUDIT_SCHEMA
    assert published["dataset"]["episodes"] == 3
    assert (
        hashlib.sha256(
            client.objects[("bucket", "run/reports/dataset-audit.json")]
        ).hexdigest()
        == result["audit_sha256"]
    )


def test_prepare_dataset_output_is_ready_for_the_split(tmp_path: Path) -> None:
    """prepare_dataset then prepare_split is the real stage order."""

    source = _v3_dataset(tmp_path / "raw", episodes=6, frames=4, cameras=True)
    client = FakeS3()
    client.seed_directory(source, "bucket", "datasets/raw")
    prepare_dataset(
        "s3://bucket/datasets/raw",
        "s3://bucket/run/prepared",
        "s3://bucket/run/reports/dataset-audit.json",
        "run-1",
        s3_client=client,
    )

    split = _split(client, "s3://bucket/run/prepared")

    assert split["status"] == "prepared"
    assert split["source"]["episodes"] == 6
    # The split reports GR00T modality keys, not raw LeRobot feature keys: the
    # adapter maps the first non-wrist camera to "front".
    assert split["source"]["camera_names"] == ["front"]
    assert split["source"]["cameras"] == [{"name": "front", "original_key": CAMERA}]
    assert split["integrity"]["leakage_free"] is True


def test_prepare_dataset_surfaces_the_adapter_failure_verbatim(
    tmp_path: Path,
) -> None:
    source = _v3_dataset(tmp_path / "raw", episodes=2, frames=3, cameras=True)
    # A camera declared in metadata whose bytes were never recorded.
    info_path = source / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.images.side"] = {
        "dtype": "video",
        "shape": [48, 64, 3],
        "names": ["height", "width", "channel"],
    }
    _write_json(info_path, info)
    client = FakeS3()
    client.seed_directory(source, "bucket", "datasets/raw")

    with pytest.raises(GrootVisualizationError, match="observation.images.side"):
        prepare_dataset(
            "s3://bucket/datasets/raw",
            "s3://bucket/run/prepared",
            "s3://bucket/run/reports/dataset-audit.json",
            "run-1",
            s3_client=client,
        )


def test_prepare_dataset_rejects_an_empty_source(tmp_path: Path) -> None:
    client = FakeS3()
    client.seed_json("bucket", "elsewhere/thing.json", {"unrelated": True})

    with pytest.raises(GrootVisualizationError, match="no material artifacts"):
        prepare_dataset(
            "s3://bucket/datasets/raw",
            "s3://bucket/run/prepared",
            "s3://bucket/run/reports/dataset-audit.json",
            "run-1",
            s3_client=client,
        )


def test_audit_treats_a_declared_zero_length_as_a_mismatch(tmp_path: Path) -> None:
    """A zero is metadata claiming an empty episode, not a missing value."""
    out = _converted(tmp_path, episodes=2, frames=3)
    path = out / "meta" / "episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows[0]["length"] = 0
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(GR00TAdapterError, match="declares 0 frames"):
        audit_dataset(out)
