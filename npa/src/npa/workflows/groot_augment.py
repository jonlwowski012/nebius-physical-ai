"""Window, generate, and merge Cosmos-augmented episodes into a GR00T dataset.

Cosmos emits a fixed-length clip regardless of the source episode's length (61
frames for Cosmos3 video2video, 93 for Cosmos Transfer), so a full-length
augmented episode needs a sliding window of generations stitched back
together rather than one generation per episode. This module owns that
windowing, the generation fan-out across every train-cohort episode and
camera, and the fail-closed merge of only the variants that pass the Cosmos
Evaluator gate into the training set.

It reuses ``npa.workflows.encord_groot_loop``'s proven single-episode
building blocks (the stdlib MP4 frame-count reader, and the geometry-conform
write that fixed a real live-run defect) rather than re-deriving them, and
generalizes the merge from one hardcoded episode/camera to every train
episode and every declared video camera.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from npa.workflows.encord_groot_loop import (
    EncordGrootError,
    _write_conformed_variant,
    mp4_video_frame_count,
)


class GrootAugmentError(RuntimeError):
    """Raised when augmentation cannot safely generate or merge synthetic episodes."""


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


def plan_windows(
    total_frames: int, window_frames: int, overlap_frames: int
) -> list[tuple[int, int]]:
    """Return (start, end) frame ranges, end exclusive, covering ``total_frames``.

    Windows advance by ``window_frames - overlap_frames`` and the last window
    is clipped to end at ``total_frames`` rather than padded, so its overlap
    with the previous window can exceed ``overlap_frames`` when the stride
    does not divide the episode evenly. An episode no longer than one window
    is covered by a single window.
    """

    if total_frames <= 0:
        raise GrootAugmentError("an episode must have at least one frame to window")
    if window_frames <= 0:
        raise GrootAugmentError("window_frames must be positive")
    if overlap_frames < 0 or overlap_frames >= window_frames:
        raise GrootAugmentError(
            "overlap_frames must be non-negative and smaller than window_frames"
        )
    if total_frames <= window_frames:
        return [(0, total_frames)]
    stride = window_frames - overlap_frames
    windows: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + window_frames, total_frames)
        windows.append((start, end))
        if end >= total_frames:
            break
        start += stride
    return windows


def _probe_clip(path: Path) -> tuple[int, float] | None:
    """Return (frame_count, fps) for an mp4, or None if unprobeable.

    Counts packets rather than decoding, and reuses
    ``data_factory_input``'s ``Fraction``-based fps parser (more robust than a
    manual ``num/den`` split) rather than re-deriving it, catching its
    PaidfInputError and returning None instead of raising -- this module's
    probes are fail-soft by design, the caller decides what unprobeable means.
    """

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        completed = subprocess.run(
            [
                ffprobe, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate,nb_read_packets",
                "-count_packets", "-of", "json", str(path),
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
    from npa.workflows.data_factory_input import PaidfInputError, _ffprobe_frame_rate

    try:
        fps = _ffprobe_frame_rate(str(stream.get("r_frame_rate") or ""), path)
    except PaidfInputError:
        return None
    try:
        frames = int(stream.get("nb_read_packets") or 0)
    except (TypeError, ValueError):
        frames = 0
    if frames <= 0:
        return None
    return frames, fps


def stitch_windows(clips: list[Path], overlap_frames: int, output: Path) -> None:
    """Concatenate overlapping window clips into one clip via cross-fade.

    By construction of ``plan_windows``, clip[i]'s trailing ``overlap_frames``
    source frames are clip[i+1]'s leading ``overlap_frames`` source frames, so
    a cross-fade over that many frames at the join hides the seam between two
    independently generated clips. A single clip is copied through unchanged.
    """

    if not clips:
        raise GrootAugmentError("stitch_windows needs at least one clip")
    if len(clips) == 1:
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(clips[0], output)
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise GrootAugmentError(
            "ffmpeg is required to stitch overlapping augmentation windows"
        )
    probes = [_probe_clip(clip) for clip in clips]
    unprobeable = [str(clips[i]) for i, probe in enumerate(probes) if probe is None]
    if unprobeable:
        raise GrootAugmentError(
            f"could not probe augmentation window clip(s) to stitch them: {unprobeable}"
        )
    fps = probes[0][1]
    with tempfile.TemporaryDirectory(prefix="npa-groot-augment-stitch-") as tmp:
        current = clips[0]
        current_frames = probes[0][0]
        for index in range(1, len(clips)):
            next_clip = clips[index]
            next_frames = probes[index][0]
            overlap = min(overlap_frames, current_frames, next_frames)
            if overlap <= 0:
                raise GrootAugmentError(
                    f"window clip {index} has no frame overlap with its "
                    "predecessor to cross-fade against"
                )
            duration = overlap / fps
            offset = (current_frames / fps) - duration
            joined = Path(tmp) / f"joined-{index:03d}.mp4"
            command = [
                ffmpeg, "-y", "-i", str(current), "-i", str(next_clip),
                "-filter_complex",
                f"[0:v][1:v]xfade=transition=fade:duration={duration:.6f}:"
                f"offset={offset:.6f}[v]",
                "-map", "[v]", "-pix_fmt", "yuv420p", "-an", str(joined),
            ]
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as exc:
                raise GrootAugmentError(
                    f"failed to cross-fade augmentation window {index}: "
                    f"{exc.stderr or exc}"
                ) from exc
            probe = _probe_clip(joined)
            if probe is None:
                raise GrootAugmentError(
                    f"cross-faded window {index} produced an unprobeable clip"
                )
            current, current_frames = joined, probe[0]
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current, output)


def _stable_seed(*parts: Any) -> int:
    """Derive a deterministic seed from (episode, camera, variant).

    The same seed is used for every window of one (episode, camera, variant)
    so all windows generate the same scene change -- windows are stitched
    into one episode and must look like one continuous take, not a collage.
    """

    digest = hashlib.sha256("\0".join(str(part) for part in parts).encode("utf-8"))
    # Cosmos3's seed is a plain int; keep it in a comfortably signed-32-bit
    # range so it round-trips through any downstream int/str conversion.
    return int(digest.hexdigest()[:8], 16) % 2_147_483_647


# ---------------------------------------------------------------------------
# Generation: fan out over every train episode, camera, and variant
# ---------------------------------------------------------------------------


VIDEO_NAME = "augmented_video.mp4"
METADATA_NAME = "metadata.json"


def generate_augmented_variants(
    dataset_uri: str,
    output_uri: str,
    manifest_uri: str,
    *,
    cameras: list[str],
    episode_indices: list[int],
    augmentation_count: int,
    window_frames: int,
    overlap_frames: int,
    prompt: str,
    checkpoint: str = "",
    guidance: float = 0.0,
    num_steps: int = 0,
    mode: str = "video2video",
    run_id: str = "",
    dry_run: bool = False,
    storage_client: Any = None,
    generate_fn: Any = None,
) -> dict[str, Any]:
    """Generate ``augmentation_count`` full-length variants per episode/camera.

    Writes one Cosmos-Evaluator-compatible clip directory per (episode,
    camera, variant) under ``output_uri`` -- ``{clip_id}/augmented_video.mp4``
    plus ``{clip_id}/metadata.json`` -- so ``workbench.cosmos_evaluator.evaluate``
    can gate them unmodified. ``generate_fn`` defaults to
    ``npa.workbench.cosmos.generate.generate_and_publish``; tests inject a
    fake to exercise the fan-out without a GPU or model.
    """

    from npa.clients.storage import StorageClient

    if augmentation_count <= 0:
        raise GrootAugmentError("augmentation_count must be positive to generate anything")
    if generate_fn is None:
        from npa.workbench.cosmos.generate import generate_and_publish as generate_fn
    client = storage_client or StorageClient.from_environment()

    with tempfile.TemporaryDirectory(prefix="npa-groot-augment-gen-") as tmp:
        root = Path(tmp)
        source = root / "source"
        client.download_directory(dataset_uri, str(source))
        if not cameras:
            info = _json(source / "meta" / "info.json")
            cameras = [
                key for key, feature in (info.get("features") or {}).items()
                if isinstance(feature, dict) and feature.get("dtype") == "video"
            ]
            if not cameras:
                raise GrootAugmentError(
                    "no camera was given and the dataset declares no video features"
                )
        if not episode_indices:
            episodes = _table(source, "meta/episodes/**/*.parquet")
            episode_indices = sorted(set(episodes["episode_index"].to_pylist()))
            if not episode_indices:
                raise GrootAugmentError("no episode was given and the dataset has no episodes")
        clip_records: list[dict[str, Any]] = []
        for episode_index in episode_indices:
            for camera in cameras:
                source_video = (
                    source / "videos" / camera / "chunk-000" / f"file-{episode_index:03d}.mp4"
                )
                if not source_video.is_file():
                    raise GrootAugmentError(
                        f"no source video for episode {episode_index}, camera {camera!r} "
                        f"at {source_video}"
                    )
                total_frames = mp4_video_frame_count(source_video)
                if total_frames is None or total_frames <= 0:
                    raise GrootAugmentError(
                        f"could not read the frame count of {source_video}"
                    )
                windows = plan_windows(total_frames, window_frames, overlap_frames)
                for variant in range(augmentation_count):
                    seed = _stable_seed(episode_index, camera, variant)
                    clip_id = f"episode-{episode_index:03d}-camera-{camera}-variant-{variant:02d}"
                    work = root / "work" / clip_id
                    work.mkdir(parents=True, exist_ok=True)
                    window_clips: list[Path] = []
                    for window_index, (start, end) in enumerate(windows):
                        window_source = work / f"window-{window_index:03d}-source.mp4"
                        _slice_clip(source_video, window_source, start=start, end=end)
                        window_output = work / f"window-{window_index:03d}-generated"
                        result = generate_fn(
                            mode=mode,
                            prompt=prompt,
                            checkpoint=checkpoint,
                            input_path=str(window_source),
                            output_path=str(window_output),
                            seed=seed,
                            num_steps=num_steps,
                            guidance=guidance,
                            run_id=run_id,
                            dry_run=dry_run,
                        )
                        window_clips.append(
                            _resolve_generated_video(result, window_output, dry_run=dry_run)
                        )
                    stitched = work / "stitched.mp4"
                    if dry_run:
                        # Dry-run windows are plans, not real clips; skip the
                        # ffmpeg stitch and record the plan instead of a file.
                        stitched = None
                    else:
                        stitch_windows(window_clips, overlap_frames, stitched)
                    clip_dir = root / "clips" / clip_id
                    clip_dir.mkdir(parents=True, exist_ok=True)
                    metadata = {
                        "prompt": prompt,
                        "inference_seed": str(seed),
                    }
                    (clip_dir / METADATA_NAME).write_text(json.dumps(metadata))
                    if stitched is not None:
                        shutil.copy2(stitched, clip_dir / VIDEO_NAME)
                    clip_records.append(
                        {
                            "clip_id": clip_id,
                            "episode_index": episode_index,
                            "camera": camera,
                            "variant": variant,
                            "seed": seed,
                            "window_count": len(windows),
                            "source_frames": total_frames,
                            "dry_run": dry_run,
                        }
                    )
                    if not dry_run:
                        client.upload_directory(str(clip_dir), f"{output_uri.rstrip('/')}/{clip_id}/")
        summary = {
            "schema": "npa.groot_augment.generation.v1",
            "dataset_uri": dataset_uri,
            "output_uri": output_uri,
            "cameras": cameras,
            "episode_indices": episode_indices,
            "augmentation_count": augmentation_count,
            "window_frames": window_frames,
            "overlap_frames": overlap_frames,
            "clips": clip_records,
        }
        manifest_path = root / "generation-manifest.json"
        manifest_path.write_text(json.dumps(summary, indent=2))
        if not dry_run:
            client.upload_file(str(manifest_path), manifest_uri)
    print(json.dumps(summary))
    return summary


def _slice_clip(source: Path, target: Path, *, start: int, end: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise GrootAugmentError("ffmpeg is required to slice augmentation windows")
    probe = _probe_clip(source)
    if probe is None:
        raise GrootAugmentError(f"could not probe {source} to slice a window from it")
    _frames, fps = probe
    start_time = start / fps
    count = end - start
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-y", "-ss", f"{start_time:.6f}", "-i", str(source),
        "-frames:v", str(count), "-pix_fmt", "yuv420p", "-an", str(target),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise GrootAugmentError(
            f"failed to slice window [{start}, {end}) from {source.name}: "
            f"{exc.stderr or exc}"
        ) from exc


def _resolve_generated_video(result: dict[str, Any], output_dir: Path, *, dry_run: bool) -> Path:
    if dry_run:
        return output_dir  # never read; dry-run windows are skipped by the caller
    for key in ("video_path", "output_path", "artifact_path"):
        value = result.get(key)
        if value and Path(str(value)).is_file():
            return Path(str(value))
    videos = sorted(output_dir.rglob("*.mp4")) if output_dir.is_dir() else []
    if videos:
        return videos[0]
    raise GrootAugmentError(
        f"generation reported no output video under {output_dir}: {result}"
    )


# ---------------------------------------------------------------------------
# Merge: only variants the gate passed become synthetic episodes
# ---------------------------------------------------------------------------


def materialize_augmented(
    dataset_uri: str,
    variants_uri: str,
    evaluation_report_uri: str,
    generation_manifest_uri: str,
    output_uri: str,
    manifest_uri: str,
    *,
    storage_client: Any = None,
) -> dict[str, Any]:
    """Merge gate-passing synthetic episodes into a GR00T train dataset.

    Generalizes ``encord_groot_loop.materialize`` from one hardcoded episode
    and camera to every (episode, camera, variant) the evaluator's report
    marks ``passed``. Every synthetic episode still copies its source
    episode's action/state rows verbatim (Cosmos changes pixels, not
    actions), and a synthetic episode's declared camera video is the
    already-stitched, already geometry-checked clip written for it.
    """

    from npa.clients.storage import StorageClient

    client = storage_client or StorageClient.from_environment()
    with tempfile.TemporaryDirectory(prefix="npa-groot-augment-merge-") as tmp:
        root = Path(tmp)
        source, variants, output = root / "source", root / "variants", root / "output"
        client.download_directory(dataset_uri, str(source))
        report = _load_json(client, evaluation_report_uri, root / "report.json")
        clips = {clip["clip_id"]: clip for clip in (report.get("clips") or [])}
        passing = [clip_id for clip_id, clip in clips.items() if clip.get("passed")]
        if not passing:
            raise GrootAugmentError(
                "no augmentation variant passed the evaluator gate; refusing to "
                "merge zero synthetic episodes when augmentation was requested"
            )
        client.download_directory(variants_uri, str(variants))
        generation = _load_json(
            client, generation_manifest_uri, root / "generation.json", optional=True,
        )
        generation_by_clip = {
            entry["clip_id"]: entry for entry in (generation.get("clips") or [])
        }
        info = _json(source / "meta" / "info.json")
        data = _table(source, "data/**/*.parquet")
        episodes = _table(source, "meta/episodes/**/*.parquet")
        required = {"episode_index", "index"}
        if not required.issubset(data.column_names) or "episode_index" not in episodes.column_names:
            raise EncordGrootError("LeRobot data/episode metadata lacks episode_index/index")
        shutil.copytree(source, output)
        next_episode = int(pc.max(data["episode_index"]).as_py()) + 1
        next_index = int(pc.max(data["index"]).as_py()) + 1
        appended: list[pa.Table] = [data]
        episode_rows = episodes.to_pylist()
        frame_counts: dict[str, int | None] = {}
        action_lineage: dict[str, int] = {}
        included_clips: list[str] = []
        for clip_id in passing:
            info_entry = generation_by_clip.get(clip_id) or {}
            episode_index = info_entry.get("episode_index")
            camera = info_entry.get("camera")
            if episode_index is None or not camera:
                # No generation-manifest match: cannot attribute this clip to a
                # source episode/camera, so it cannot inherit action rows.
                continue
            selected = episodes.filter(pc.equal(episodes["episode_index"], int(episode_index)))
            if selected.num_rows != 1:
                raise GrootAugmentError(
                    f"clip {clip_id} names episode {episode_index}, which does not "
                    "resolve to exactly one metadata row"
                )
            source_rows = data.filter(pc.equal(data["episode_index"], int(episode_index)))
            if not source_rows.num_rows:
                continue
            variant_video = variants / clip_id / VIDEO_NAME
            if not variant_video.is_file():
                raise GrootAugmentError(f"gate-passing clip {clip_id} has no video at {variant_video}")
            eid = next_episode
            next_episode += 1
            rows = source_rows
            frames = mp4_video_frame_count(variant_video)
            frame_counts[str(eid)] = frames
            if frames is not None and frames < rows.num_rows:
                rows = rows.slice(0, frames)
            rows = rows.set_column(
                rows.schema.get_field_index("episode_index"), "episode_index",
                pa.array([eid] * rows.num_rows, type=data["episode_index"].type),
            )
            rows = rows.set_column(
                rows.schema.get_field_index("index"), "index",
                pa.array(range(next_index, next_index + rows.num_rows), type=data["index"].type),
            )
            next_index += rows.num_rows
            appended.append(rows)
            row = dict(selected.to_pylist()[0])
            row["episode_index"] = eid
            row["data/chunk_index"] = 0
            row["data/file_index"] = 0
            row["dataset_from_index"] = next_index - rows.num_rows
            row["dataset_to_index"] = next_index
            if "length" in row:
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
            video_root = output / "videos" / camera / "chunk-000"
            video_root.mkdir(parents=True, exist_ok=True)
            _write_conformed_variant(
                variant_video, video_root / f"file-{eid:03d}.mp4",
                width=declared_w, height=declared_h, fps=float(info.get("fps") or 0.0),
            )
            action_lineage[str(eid)] = int(episode_index)
            included_clips.append(clip_id)
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
        summary = {
            "schema": "npa.groot_augment.materialization.v1",
            "dataset_uri": dataset_uri,
            "output_uri": output_uri,
            "original_episodes": episodes.num_rows,
            "synthetic_episodes": len(included_clips),
            "total_episodes": len(episode_rows),
            "included_clips": included_clips,
            "rejected_clips": sorted(set(clips) - set(included_clips)),
            "action_lineage": action_lineage,
            "synthetic_frame_counts": frame_counts,
        }
        client.upload_directory(str(output), output_uri)
        summary_path = root / "materialization-summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        client.upload_file(str(summary_path), manifest_uri)
    print(json.dumps(summary))
    return summary


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


def _load_json(client: Any, uri: str, local: Path, *, optional: bool = False) -> dict[str, Any]:
    try:
        client.download_file(uri, str(local))
    except Exception as exc:  # noqa: BLE001 - storage clients raise their own types
        if optional:
            return {}
        raise GrootAugmentError(f"could not read required manifest {uri}: {exc}") from exc
    try:
        return json.loads(local.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        if optional:
            return {}
        raise GrootAugmentError(f"invalid JSON manifest at {uri}") from exc


# ---------------------------------------------------------------------------
# Module-invocation CLI: `python3 -m npa.workflows.groot_augment <command> ...`
#
# Deliberately not a Typer `npa workbench ...` command. A capability image
# (one image, many tools) exposes exactly one tool group through its light
# CLI, selected by NPA_LIGHT_WORKBENCH_TOOL -- `generate` needs the Cosmos3
# runtime image and `materialize` needs none of it, and routing either
# through `npa workbench groot ...` would run them under the wrong tool
# group's light CLI ("No such command ..."). Module invocation bypasses the
# light CLI's command dispatch entirely, the same way every other
# `workflow.groot.*` toolRef already does (see groot_learning.py).
# ---------------------------------------------------------------------------


def _cli_generate(args: Any) -> None:
    cameras = [c.strip() for c in str(args.cameras or "").split(",") if c.strip()]
    episode_indices = [
        int(e.strip()) for e in str(args.episode_indices or "").split(",") if e.strip()
    ]
    generate_augmented_variants(
        args.dataset_uri, args.output_uri, args.manifest_uri,
        cameras=cameras, episode_indices=episode_indices,
        augmentation_count=args.augmentation_count, window_frames=args.window_frames,
        overlap_frames=args.overlap_frames, prompt=args.prompt, checkpoint=args.checkpoint,
        guidance=args.guidance, num_steps=args.num_steps, mode=args.mode, run_id=args.run_id,
        dry_run=args.dry_run,
    )


def _cli_materialize(args: Any) -> None:
    materialize_augmented(
        args.dataset_uri, args.variants_uri, args.evaluation_report_uri,
        args.generation_manifest_uri, args.output_uri, args.manifest_uri,
    )


def build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--dataset-uri", dest="dataset_uri", required=True)
    generate.add_argument("--output-uri", dest="output_uri", required=True)
    generate.add_argument("--manifest-uri", dest="manifest_uri", required=True)
    generate.add_argument("--cameras", dest="cameras", default="")
    generate.add_argument("--episode-indices", dest="episode_indices", default="")
    generate.add_argument("--augmentation-count", dest="augmentation_count", type=int, required=True)
    generate.add_argument("--window-frames", dest="window_frames", type=int, required=True)
    generate.add_argument("--overlap-frames", dest="overlap_frames", type=int, required=True)
    generate.add_argument("--prompt", dest="prompt", required=True)
    generate.add_argument("--checkpoint", dest="checkpoint", default="")
    generate.add_argument("--guidance", dest="guidance", type=float, default=0.0)
    generate.add_argument("--num-steps", dest="num_steps", type=int, default=0)
    generate.add_argument("--mode", dest="mode", default="video2video")
    generate.add_argument("--run-id", dest="run_id", default="")
    generate.add_argument("--dry-run", dest="dry_run", action="store_true")
    generate.set_defaults(func=_cli_generate)

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--dataset-uri", dest="dataset_uri", required=True)
    materialize.add_argument("--variants-uri", dest="variants_uri", required=True)
    materialize.add_argument("--evaluation-report-uri", dest="evaluation_report_uri", required=True)
    materialize.add_argument("--generation-manifest-uri", dest="generation_manifest_uri", required=True)
    materialize.add_argument("--output-uri", dest="output_uri", required=True)
    materialize.add_argument("--manifest-uri", dest="manifest_uri", required=True)
    materialize.set_defaults(func=_cli_materialize)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
