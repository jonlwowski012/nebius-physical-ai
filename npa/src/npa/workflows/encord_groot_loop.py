"""Materialize action-preserving LeRobot visual augmentations for GR00T.

Cosmos produces pixels, not robot actions.  This module therefore retains every
original episode and adds one copy of a selected episode per generated video;
only the configured camera asset changes.  It fails closed for datasets without
the standard LeRobot v3 episode/data contracts.
"""

from __future__ import annotations

import json
import shutil
import subprocess
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


def _mp4_box_children(data: bytes, start: int, end: int):
    """Yield (type, payload_start, payload_end) for the boxes in data[start:end]."""

    offset = start
    while offset + 8 <= end:
        size = int.from_bytes(data[offset : offset + 4], "big")
        box_type = data[offset + 4 : offset + 8]
        header = 8
        if size == 1:
            if offset + 16 > end:
                return
            size = int.from_bytes(data[offset + 8 : offset + 16], "big")
            header = 16
        elif size == 0:
            size = end - offset
        if size < header or offset + size > end:
            return
        yield box_type, offset + header, offset + size
        offset += size


def mp4_video_frame_count(path: Path) -> int | None:
    """Return the sample count of the first video track, or None when unparseable.

    Walks ``moov/trak/mdia/{hdlr,minf/stbl/stsz}`` with the stdlib only, so the
    materialize stage needs no ffmpeg or PyAV. GR00T loads frames by row index,
    so a synthetic episode may not carry more action rows than its clip has
    frames; ``None`` (a non-ISO file) keeps the caller's legacy behaviour.
    """

    try:
        data = path.read_bytes()
    except OSError:
        return None
    for box_type, payload_start, payload_end in _mp4_box_children(data, 0, len(data)):
        if box_type != b"moov":
            continue
        for trak_type, trak_start, trak_end in _mp4_box_children(data, payload_start, payload_end):
            if trak_type != b"trak":
                continue
            for mdia_type, mdia_start, mdia_end in _mp4_box_children(data, trak_start, trak_end):
                if mdia_type != b"mdia":
                    continue
                is_video = False
                sample_count: int | None = None
                for child_type, child_start, child_end in _mp4_box_children(data, mdia_start, mdia_end):
                    if child_type == b"hdlr" and child_end - child_start >= 12:
                        is_video = data[child_start + 8 : child_start + 12] == b"vide"
                    elif child_type == b"minf":
                        for minf_type, minf_start, minf_end in _mp4_box_children(data, child_start, child_end):
                            if minf_type != b"stbl":
                                continue
                            for stbl_type, stbl_start, stbl_end in _mp4_box_children(data, minf_start, minf_end):
                                if stbl_type == b"stsz" and stbl_end - stbl_start >= 12:
                                    sample_count = int.from_bytes(data[stbl_start + 8 : stbl_start + 12], "big")
                if is_video and sample_count is not None and sample_count > 0:
                    return sample_count
    return None


def _generated_videos(root: Path) -> list[Path]:
    videos = sorted(path for path in root.rglob("*.mp4") if path.stat().st_size)
    if not videos:
        raise EncordGrootError("Cosmos output contains no non-empty .mp4 variants")
    return videos


def _probe_variant_geometry(path: Path) -> tuple[int, int, float] | None:
    """Return (width, height, fps) of a Cosmos variant, or None if unprobeable."""

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        completed = subprocess.run(
            [
                ffprobe, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate",
                "-of", "json", str(path),
            ],
            check=True, capture_output=True, text=True,
        )
        streams = (json.loads(getattr(completed, "stdout", "") or "{}") or {}).get(
            "streams"
        ) or []
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError, ValueError):
        return None
    if not streams:
        return None
    stream = streams[0]
    rate = str(stream.get("r_frame_rate") or "0/1")
    try:
        numerator, _, denominator = rate.partition("/")
        fps = float(numerator) / float(denominator or 1)
    except (TypeError, ValueError, ZeroDivisionError):
        fps = 0.0
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        # A container ffprobe can open but that carries no real video stream is
        # not something we can conform; treat it as unprobeable.
        return None
    return width, height, fps


def _write_conformed_variant(
    variant: Path, target: Path, *, width: int, height: int, fps: float
) -> None:
    """Place a Cosmos variant into the dataset at the declared geometry.

    A LeRobot dataset declares one shape and one fps per camera, so an episode
    video that disagrees corrupts the dataset even when its frame count is
    right. Live run cosmos-check-20260909T173508Z copied Cosmos output verbatim
    into a 96x96/10fps pusht dataset, storing 1280x720/24fps episodes: the
    resolution disagreed with `info.json` by 13x and the action timebase was
    wrong by 2.4x, while every metadata-only check passed.

    Copies bytes when the variant already conforms, so the common case still
    needs no ffmpeg. Fails closed rather than writing a mismatched dataset.
    """

    geometry = _probe_variant_geometry(variant)
    if geometry is None:
        # Unprobeable: preserve the historical copy rather than fail a variant
        # we cannot reason about.
        shutil.copy2(variant, target)
        return
    actual_w, actual_h, actual_fps = geometry
    conforms = (
        (not width or actual_w == width)
        and (not height or actual_h == height)
        and (not fps or abs(actual_fps - fps) <= 0.01)
    )
    if conforms:
        shutil.copy2(variant, target)
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise EncordGrootError(
            f"Cosmos variant {variant.name} is {actual_w}x{actual_h} at "
            f"{actual_fps:g}fps but the dataset declares {width}x{height} at "
            f"{fps:g}fps, and ffmpeg is unavailable to conform it. Refusing to "
            "write a dataset whose metadata contradicts its video."
        )
    command = [
        ffmpeg, "-y", "-i", str(variant),
        "-vf", f"scale={width}:{height},fps={fps:g}",
        "-pix_fmt", "yuv420p", "-an", str(target),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise EncordGrootError(
            f"failed to conform Cosmos variant {variant.name} to "
            f"{width}x{height}@{fps:g}fps: {exc.stderr or exc}"
        ) from exc
    if not target.exists() or target.stat().st_size <= 0:
        raise EncordGrootError(
            f"conforming Cosmos variant {variant.name} produced no video bytes"
        )


def materialize(
    source_uri: str,
    augmented_uri: str,
    output_uri: str,
    camera: str,
    episode_index: str,
    manifest_uri: str,
    heldout_episode_index: str = "",
    *,
    storage_client: Any = None,
) -> dict[str, Any]:
    """Create a LeRobot v3 dataset containing originals plus synthetic episodes.

    ``heldout_episode_index`` names an original episode that a downstream split
    will hold out. Synthetic episodes copy the augmented episode's actions
    verbatim, so the held-out episode must be a different original; the check
    fails closed here, before any GPU stage, and the lineage is recorded in the
    materialization summary.
    """

    from npa.clients.storage import StorageClient

    try:
        selected_episode = int(episode_index)
    except ValueError as exc:
        raise EncordGrootError("lerobot_episode_index must be an integer") from exc
    heldout_episode: int | None = None
    if str(heldout_episode_index).strip():
        try:
            heldout_episode = int(heldout_episode_index)
        except ValueError as exc:
            raise EncordGrootError("heldout episode index must be an integer") from exc
        if heldout_episode == selected_episode:
            raise EncordGrootError(
                "the held-out episode must differ from the augmented episode: synthetic "
                "episodes copy its actions, so holding it out would leak them"
            )
    client = storage_client or StorageClient.from_environment()
    with tempfile.TemporaryDirectory(prefix="npa-encord-groot-") as tmp:
        root = Path(tmp)
        source, generated, output = root / "source", root / "generated", root / "output"
        client.download_directory(source_uri, str(source))
        client.download_directory(augmented_uri, str(generated))
        if (source / "meta" / "episodes.jsonl").is_file():
            raise EncordGrootError(
                "LeRobot source carries meta/episodes.jsonl; the GR00T adapter prefers it "
                "over the v3 episodes parquet and would drop the synthetic episodes"
            )
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
        if heldout_episode is not None:
            if episodes.num_rows < 2:
                raise EncordGrootError("holding out an episode requires at least two original episodes")
            heldout_rows = episodes.filter(pc.equal(episodes["episode_index"], heldout_episode))
            if heldout_rows.num_rows != 1:
                raise EncordGrootError(
                    f"held-out LeRobot episode {heldout_episode} must resolve to exactly one metadata row"
                )
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
        frame_counts: dict[str, int | None] = {}
        truncated: dict[str, int] = {}
        for offset, variant in enumerate(variants):
            eid = next_episode + offset
            rows = source_rows
            frames = mp4_video_frame_count(variant)
            frame_counts[str(eid)] = frames
            if frames is not None and frames < rows.num_rows:
                # Cosmos emits a fixed-length clip regardless of the source; GR00T
                # indexes frames by row, so rows past the last frame would be
                # unreadable. Keep the leading rows the clip actually covers.
                truncated[str(eid)] = rows.num_rows - frames
                rows = rows.slice(0, frames)
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
            if "length" in row:
                # The metadata row is a copy of the source episode's; GR00T's
                # loader trusts ``length`` (episodes.jsonl) when indexing frames,
                # so a trimmed clip must not inherit the source's row count
                # (live run 5 wrote 161 for a 61-frame variant before this).
                row["length"] = rows.num_rows
            row[f"videos/{camera}/chunk_index"] = 0
            row[f"videos/{camera}/file_index"] = eid
            row[f"videos/{camera}/from_timestamp"] = 0.0
            row[f"videos/{camera}/to_timestamp"] = float(rows.num_rows) / float(info.get("fps") or 1)
            episode_rows.append(row)
            declared = (info.get("features") or {}).get(camera) or {}
            shape = declared.get("shape") or []
            declared_h = int(shape[0]) if len(shape) >= 2 else 0
            declared_w = int(shape[1]) if len(shape) >= 2 else 0
            _write_conformed_variant(
                variant,
                video_root / f"file-{eid:03d}.mp4",
                width=declared_w,
                height=declared_h,
                fps=float(info.get("fps") or 0.0),
            )
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
        synthetic_indices = [next_episode + offset for offset in range(len(variants))]
        summary = {
            "schema": "npa.encord_groot.materialization.v1",
            "source_uri": source_uri,
            "output_uri": output_uri,
            "camera": camera,
            "original_episodes": episodes.num_rows,
            "synthetic_episodes": len(variants),
            "total_episodes": len(episode_rows),
            "augmented_episode_index": selected_episode,
            "heldout_episode_index": heldout_episode,
            "synthetic_episode_indices": synthetic_indices,
            # Synthetic episodes inherit the augmented episode's action/state rows
            # verbatim; only the camera pixels differ.
            "action_lineage": {str(index): selected_episode for index in synthetic_indices},
            # Frames parsed from each variant's MP4 (None when the container was not
            # parseable) and how many trailing action rows were dropped to fit them.
            "synthetic_frame_counts": frame_counts,
            "synthetic_rows_truncated": truncated,
        }
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
    if len(sys.argv) not in (8, 9) or sys.argv[1] != "materialize":
        raise SystemExit(
            "usage: encord_groot_loop materialize SOURCE AUGMENTED OUTPUT CAMERA EPISODE "
            "MANIFEST [HELDOUT_EPISODE]"
        )
    materialize(*sys.argv[2:])
