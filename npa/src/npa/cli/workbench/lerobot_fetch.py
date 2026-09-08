"""``npa workbench lerobot fetch-dataset`` — stage a public LeRobot dataset.

This is the operator-side input step for any pipeline that trains on public
demonstrations: it pins an immutable Hugging Face revision, validates the
LeRobot contract before spending upload bandwidth, and leaves a provenance
receipt inside the staged dataset so a later run can prove what it trained on.

The command is registered onto the ``lerobot`` group by that module. It lives
here rather than being appended to it because staging is self-contained: it
touches Hugging Face and object storage, and none of the VM/SSH machinery.
"""

from __future__ import annotations

import json
import re
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, NoReturn

import typer

from npa.cli.path_contract import PathContractError, validate_write_path

#: A full commit SHA. A branch or tag can move, which would silently change
#: what a "reproduced" experiment trained on.
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")

PROVENANCE_SCHEMA = "npa.lerobot.dataset_provenance.v1"
PROVENANCE_FILENAME = "npa_dataset_provenance.json"


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


def fetch_dataset_cmd(
    hf_repo: str = typer.Option(
        ...,
        "--hf-repo",
        help="Hugging Face dataset repository, e.g. lerobot/svla_so100_pickplace.",
    ),
    revision: str = typer.Option(
        ...,
        "--revision",
        help="Immutable 40-character commit SHA. A branch or tag is not reproducible.",
    ),
    output_path: str = typer.Option(
        ...,
        "--output-path",
        "--output-uri",
        help="S3 URI prefix the dataset is staged to.",
    ),
    dataset_license: str = typer.Option(
        "", "--license", help="License recorded in the provenance receipt."
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", help="Output format."
    ),
) -> None:
    """Stage a pinned public LeRobot dataset from Hugging Face into S3."""

    from npa.workflows.lerobot_dataset import (
        LeRobotDatasetError,
        download_public_lerobot_dataset,
        stage_dataset_to_s3,
        summarize_lerobot_dataset,
    )

    revision = revision.strip()
    if not _COMMIT_SHA.match(revision):
        _fail(
            "--revision must be a full 40-character commit SHA so the staged "
            f"dataset is reproducible; got {revision!r}"
        )
    try:
        staged = validate_write_path(
            output_path,
            tool="LeRobot fetch-dataset",
            option="--output-path",
            required=True,
        )
    except PathContractError as exc:
        _fail(str(exc))

    source_uri = f"hf://datasets/{hf_repo}@{revision}"
    with tempfile.TemporaryDirectory(prefix="npa-lerobot-fetch-") as tmp:
        try:
            local = download_public_lerobot_dataset(
                Path(tmp), repo_id=hf_repo, revision=revision
            )
            summary = summarize_lerobot_dataset(
                local,
                source_uri=source_uri,
                repo_id=hf_repo,
                revision=revision,
                license=dataset_license,
            )
            provenance: dict[str, Any] = {
                "schema": PROVENANCE_SCHEMA,
                "source_uri": source_uri,
                "hf_repo": hf_repo,
                "revision": revision,
                "license": dataset_license,
                "staged_uri": staged,
                "total_episodes": summary.total_episodes,
                "total_frames": summary.total_frames,
                "fps": summary.fps,
                "camera_keys": list(summary.camera_keys),
            }
            # The receipt travels with the bytes, so a consumer of the staged
            # prefix can read it without this command's stdout.
            (local / "meta" / PROVENANCE_FILENAME).write_text(
                json.dumps(provenance, indent=2, sort_keys=True) + "\n"
            )
            staged_to = stage_dataset_to_s3(local, staged)
        except LeRobotDatasetError as exc:
            _fail(str(exc))

    _emit({"status": "staged", "output_uri": staged_to, **provenance}, output)


def _fail(message: str) -> NoReturn:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)


def _emit(payload: dict[str, Any], fmt: OutputFormat) -> None:
    if fmt == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        typer.echo(f"  {key}: {value}")
