"""Materialize action-preserving LeRobot visual augmentations for GR00T.

Cosmos produces pixels, not robot actions.  This module therefore retains every
original episode and adds one copy of a selected episode per generated video;
only the configured camera asset changes.  It fails closed for datasets without
the standard LeRobot v3 episode/data contracts.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


class EncordGrootError(RuntimeError):
    """Raised when a generated video cannot safely inherit trajectory labels."""


def _json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EncordGrootError(f"invalid LeRobot metadata: {path}") from exc


def _table(root: Path, pattern: str) -> pa.Table:
    paths = sorted(root.glob(pattern))
    if not paths:
        raise EncordGrootError(f"missing required LeRobot files: {pattern}")
    return pa.concat_tables([pq.read_table(path) for path in paths], promote_options="default")


def _generated_videos(root: Path) -> list[Path]:
    videos = sorted(path for path in root.rglob("*.mp4") if path.stat().st_size)
    if not videos:
        raise EncordGrootError("Cosmos output contains no non-empty .mp4 variants")
    return videos


def materialize(
    source_uri: str,
    augmented_uri: str,
    output_uri: str,
    camera: str,
    episode_index: str,
    manifest_uri: str,
    *,
    storage_client: Any = None,
) -> dict[str, Any]:
    """Create a LeRobot v3 dataset containing originals plus synthetic episodes."""

    from npa.clients.storage import StorageClient

    try:
        selected_episode = int(episode_index)
    except ValueError as exc:
        raise EncordGrootError("lerobot_episode_index must be an integer") from exc
    client = storage_client or StorageClient.from_environment()
    with tempfile.TemporaryDirectory(prefix="npa-encord-groot-") as tmp:
        root = Path(tmp)
        source, generated, output = root / "source", root / "generated", root / "output"
        client.download_directory(source_uri, str(source))
        client.download_directory(augmented_uri, str(generated))
        info = _json(source / "meta" / "info.json")
        feature = (info.get("features") or {}).get(camera) or {}
        if feature.get("dtype") != "video":
            raise EncordGrootError(f"{camera!r} is not a declared LeRobot video feature")
        data = _table(source, "data/**/*.parquet")
        episodes = _table(source, "meta/episodes/**/*.parquet")
        required = {"episode_index", "index"}
        if not required.issubset(data.column_names) or "episode_index" not in episodes.column_names:
            raise EncordGrootError("LeRobot data/episode metadata lacks episode_index/index")
        selected = episodes.filter(pc.equal(episodes["episode_index"], selected_episode))
        if selected.num_rows != 1:
            raise EncordGrootError("selected LeRobot episode must resolve to exactly one metadata row")
        source_rows = data.filter(pc.equal(data["episode_index"], selected_episode))
        if not source_rows.num_rows:
            raise EncordGrootError("selected LeRobot episode has no action/state rows")
        variants = _generated_videos(generated)
        shutil.copytree(source, output)
        next_episode = int(pc.max(data["episode_index"]).as_py()) + 1
        next_index = int(pc.max(data["index"]).as_py()) + 1
        appended: list[pa.Table] = [data]
        episode_rows = episodes.to_pylist()
        video_root = output / "videos" / camera / "chunk-000"
        video_root.mkdir(parents=True, exist_ok=True)
        for offset, variant in enumerate(variants):
            eid = next_episode + offset
            rows = source_rows
            rows = rows.set_column(rows.schema.get_field_index("episode_index"), "episode_index", pa.array([eid] * rows.num_rows, type=data["episode_index"].type))
            rows = rows.set_column(rows.schema.get_field_index("index"), "index", pa.array(range(next_index, next_index + rows.num_rows), type=data["index"].type))
            next_index += rows.num_rows
            appended.append(rows)
            row = dict(selected.to_pylist()[0])
            row["episode_index"] = eid
            row["data/chunk_index"] = 0
            row["data/file_index"] = 0
            row["dataset_from_index"] = next_index - rows.num_rows
            row["dataset_to_index"] = next_index
            row[f"videos/{camera}/chunk_index"] = 0
            row[f"videos/{camera}/file_index"] = eid
            row[f"videos/{camera}/from_timestamp"] = 0.0
            row[f"videos/{camera}/to_timestamp"] = float(rows.num_rows) / float(info.get("fps") or 1)
            episode_rows.append(row)
            shutil.copy2(variant, video_root / f"file-{eid:03d}.mp4")
        combined = pa.concat_tables(appended, promote_options="default")
        data_out = output / "data" / "chunk-000" / "file-000.parquet"
        shutil.rmtree(output / "data")
        data_out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(combined, data_out)
        episodes_out = output / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        shutil.rmtree(output / "meta" / "episodes")
        episodes_out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(episode_rows, schema=episodes.schema), episodes_out)
        info["total_episodes"] = len(episode_rows)
        info["total_frames"] = combined.num_rows
        (output / "meta" / "info.json").write_text(json.dumps(info, indent=2))
        summary = {"schema": "npa.encord_groot.materialization.v1", "source_uri": source_uri, "output_uri": output_uri, "camera": camera, "original_episodes": episodes.num_rows, "synthetic_episodes": len(variants), "total_episodes": len(episode_rows)}
        # GR00T's fine-tune loader requires the GR00T LeRobot layout plus the
        # generated modality config for NEW_EMBODIMENT; a plain LeRobot tree
        # fails config.validate() with "No modality config registered".
        from npa.adapter.groot import lerobot_to_groot

        groot_output = root / "groot-output"
        lerobot_to_groot(output, groot_output)
        # The materialized synthetic episodes copy the original episode's action
        # rows verbatim, so actions are absolute targets; GR00T's relative-action
        # statistics also cannot be computed over these short smoke episodes.
        # Same rewrite groot_learning.py applies for absolute action datasets.
        config_path = groot_output / "meta" / "npa_groot_modality_config.py"
        if config_path.is_file():
            config_path.write_text(
                config_path.read_text().replace(
                    "ActionRepresentation.RELATIVE", "ActionRepresentation.ABSOLUTE"
                )
            )
        (groot_output / "materialization.json").write_text(json.dumps(summary, indent=2))
        client.upload_directory(str(groot_output), output_uri)
        client.upload_file(str(groot_output / "materialization.json"), manifest_uri)
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) != 8 or sys.argv[1] != "materialize":
        raise SystemExit("usage: encord_groot_loop materialize SOURCE AUGMENTED OUTPUT CAMERA EPISODE MANIFEST")
    materialize(*sys.argv[2:])
