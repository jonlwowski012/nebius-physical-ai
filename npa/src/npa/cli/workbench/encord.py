"""Typer CLI for `npa workbench encord`.

Encord labeling/curation SaaS integration:

- ``push``  register S3 media in place into an Encord storage folder (and
  optionally link a dataset) through a cloud integration created once in the
  Encord app. Bytes stay in the bucket.
- ``pull``  materialize a curated Collection, a Dataset, or a Project's labels
  back to an S3 prefix as media + item JSON + a lineage manifest.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

import typer

app = typer.Typer(
    name="encord",
    help="Encord curation SaaS: register-in-place push and curated pull.",
    no_args_is_help=True,
)


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


class TransferMode(str, Enum):
    register = "register"
    upload = "upload"


class MediaFilter(str, Enum):
    videos_images = "videos-images"
    mcap = "mcap"
    all = "all"


class PullSource(str, Enum):
    collection = "collection"
    dataset = "dataset"
    project = "project"


def _fail(message: str) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(1)


def _emit(payload: dict[str, Any], *, output: OutputFormat, text: str) -> None:
    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(text)


@app.command("push")
def push_cmd(
    input_path: str = typer.Option(
        ...,
        "--input-path",
        help="s3:// prefix of media to register in place (bytes stay in the bucket).",
    ),
    integration: str = typer.Option(
        "",
        "--integration",
        help="Encord cloud-integration title or uuid (created once in the Encord "
        "app). Required for --transfer register; unused for upload.",
    ),
    folder: str = typer.Option(
        ...,
        "--folder",
        help="Encord storage folder title or uuid; a title is created if absent.",
    ),
    output_path: str = typer.Option(
        ...,
        "--output-path",
        help="s3:// destination prefix (or .json URI) for the push receipt.",
    ),
    dataset: str = typer.Option(
        "",
        "--dataset",
        help="Optional Encord dataset hash or title to link registered items into; "
        "a title is created if absent.",
    ),
    transfer: TransferMode = typer.Option(
        TransferMode.register,
        "--transfer",
        help="register: bytes stay in the bucket, Encord references objectUrls. "
        "upload: bytes are copied into Encord-hosted storage.",
    ),
    media: MediaFilter = typer.Option(
        MediaFilter.videos_images,
        "--media",
        help="Which media suffixes to register. 'mcap'/'all' enable the "
        "experimental MCAP path.",
    ),
    poll_timeout_seconds: int = typer.Option(
        1800,
        "--poll-timeout-seconds",
        help="Per-batch registration poll timeout.",
    ),
    workflow_run: str = typer.Option(
        "", "--workflow-run", help="Run id recorded in the receipt."
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.json, "--output", help="Output format."
    ),
) -> None:
    """Register S3 media in Encord and optionally link a dataset."""

    from npa.cli.path_contract import PathContractError, validate_read_path, validate_write_path

    try:
        validate_read_path(input_path, tool="encord push", allow_hf=False)
        validate_write_path(output_path, tool="encord push", required=True)
    except PathContractError as exc:
        _fail(str(exc))

    from npa.sdk.workbench.encord import push as sdk_push
    from npa.workbench.encord.schemas import EncordToolError

    try:
        receipt = sdk_push(
            input_path=input_path,
            integration=integration,
            folder=folder,
            output_path=output_path,
            dataset=dataset,
            media=media.value,
            transfer=transfer.value,
            poll_timeout_seconds=poll_timeout_seconds,
            workflow_run=workflow_run,
        )
    except EncordToolError as exc:
        _fail(str(exc))
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        _fail(f"encord push failed: {exc}")

    payload = receipt.model_dump(by_alias=True)
    _emit(
        payload,
        output=output,
        text=(
            f"pushed {receipt.units_done}/{receipt.files_discovered} item(s) to "
            f"Encord folder {receipt.folder_name!r} "
            f"(linked {receipt.linked_count}); receipt: {receipt.receipt_uri}"
        ),
    )


@app.command("pull")
def pull_cmd(
    source: PullSource = typer.Option(
        ...,
        "--source",
        help="Which Encord container to pull: collection, dataset, or project.",
    ),
    source_id: str = typer.Option(
        ...,
        "--source-id",
        help="Collection uuid / dataset hash / project hash, or a unique title.",
    ),
    output_path: str = typer.Option(
        ...,
        "--output-path",
        help="s3:// output prefix for media/, items/, labels/, and manifest.json.",
    ),
    workflow_run: str = typer.Option(
        "", "--workflow-run", help="Run id recorded in the manifest."
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.json, "--output", help="Output format."
    ),
) -> None:
    """Pull curated media + labels + lineage manifest back to S3."""

    from npa.cli.path_contract import PathContractError, validate_write_path

    try:
        validate_write_path(output_path, tool="encord pull", required=True)
    except PathContractError as exc:
        _fail(str(exc))

    from npa.sdk.workbench.encord import pull as sdk_pull
    from npa.workbench.encord.schemas import EncordToolError

    try:
        manifest = sdk_pull(
            source=source.value,
            source_id=source_id,
            output_path=output_path,
            workflow_run=workflow_run,
        )
    except EncordToolError as exc:
        _fail(str(exc))
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        _fail(f"encord pull failed: {exc}")

    payload = manifest.model_dump(by_alias=True)
    _emit(
        payload,
        output=output,
        text=(
            f"pulled {manifest.items_total} item(s) "
            f"({manifest.media_copied} copied, {manifest.media_downloaded} "
            f"downloaded, {manifest.label_rows} label rows) from "
            f"{manifest.source_kind} {manifest.source_name!r}; manifest: "
            f"{manifest.manifest_uri}"
        ),
    )
