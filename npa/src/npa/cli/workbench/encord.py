"""Typer CLI for `npa workbench encord`.

Encord labeling/curation SaaS integration:

- ``push``  register S3 media in place into an Encord storage folder (and
  optionally link a dataset) through a cloud integration created once in the
  Encord app. Bytes stay in the bucket.
- ``curate``  headless curation: declare quality filters (brightness, width,
  ...) from workbench; Encord evaluates them server-side into a Collection —
  no human in the app.
- ``pull``  materialize a curated Collection, a Dataset, or a Project's labels
  back to an S3 prefix as media + item JSON + a lineage manifest.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

import typer

from npa.workbench.encord.schemas import (
    DEFAULT_CURATE_POLL_SECONDS,
    DEFAULT_POLL_TIMEOUT_SECONDS,
)

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
        DEFAULT_POLL_TIMEOUT_SECONDS,
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


@app.command("curate")
def curate_cmd(
    folder: str = typer.Option(
        ...,
        "--folder",
        help="Encord storage folder title or uuid to curate (never created).",
    ),
    filters: list[str] = typer.Option(
        [],
        "--filter",
        help="Quality filter metric:min:max (repeatable, or comma-separated in "
        "one value), e.g. brightness:0.2:0.8. Supported metrics: width, "
        "height, area, aspect-ratio, brightness, sharpness, file-size.",
    ),
    collection: str = typer.Option(
        ...,
        "--collection",
        help="Target Encord Collection title or uuid; a title is created if "
        "absent. Pull the result with --source collection.",
    ),
    output_path: str = typer.Option(
        ...,
        "--output-path",
        help="s3:// destination prefix (or .json URI) for the curate receipt.",
    ),
    poll_seconds: float = typer.Option(
        DEFAULT_CURATE_POLL_SECONDS,
        "--poll-seconds",
        help="How long to wait for Encord's async server-side selection.",
    ),
    workflow_run: str = typer.Option(
        "", "--workflow-run", help="Run id recorded in the receipt."
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.json, "--output", help="Output format."
    ),
) -> None:
    """Headlessly curate a folder into a Collection via Encord quality filters."""

    from npa.cli.path_contract import PathContractError, validate_write_path

    try:
        validate_write_path(output_path, tool="encord curate", required=True)
    except PathContractError as exc:
        _fail(str(exc))

    from npa.sdk.workbench.encord import curate as sdk_curate
    from npa.workbench.encord.schemas import EncordToolError

    try:
        receipt = sdk_curate(
            folder=folder,
            filters=filters,
            collection=collection,
            output_path=output_path,
            workflow_run=workflow_run,
            poll_seconds=poll_seconds,
        )
    except EncordToolError as exc:
        _fail(str(exc))
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        _fail(f"encord curate failed: {exc}")

    payload = receipt.model_dump(by_alias=True)
    _emit(
        payload,
        output=output,
        text=(
            f"curated {receipt.items_selected} item(s) from folder "
            f"{receipt.folder_name!r} into collection "
            f"{receipt.collection_name!r}; receipt: {receipt.receipt_uri}"
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


@app.command("verify")
def verify_cmd(
    receipt_uri: str = typer.Option(
        ...,
        "--receipt-uri",
        help="s3:// URI of the push receipt (push_receipt.json).",
    ),
    manifest_uri: str = typer.Option(
        ...,
        "--manifest-uri",
        help="s3:// URI of the pull manifest (manifest.json).",
    ),
    output_path: str = typer.Option(
        ...,
        "--output-path",
        help="s3:// destination prefix (or .json URI) for the roundtrip report.",
    ),
    workflow_run: str = typer.Option(
        "", "--workflow-run", help="Run id recorded in the report."
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.json, "--output", help="Output format."
    ),
) -> None:
    """Verify a push receipt against a pull manifest by exact identity."""

    from npa.cli.path_contract import PathContractError, validate_read_path, validate_write_path

    try:
        validate_read_path(receipt_uri, tool="encord verify", allow_hf=False)
        validate_read_path(manifest_uri, tool="encord verify", allow_hf=False)
        validate_write_path(output_path, tool="encord verify", required=True)
    except PathContractError as exc:
        _fail(str(exc))

    from npa.sdk.workbench.encord import verify as sdk_verify
    from npa.workbench.encord.schemas import EncordToolError

    try:
        report = sdk_verify(
            receipt_uri=receipt_uri,
            manifest_uri=manifest_uri,
            output_path=output_path,
            workflow_run=workflow_run,
        )
    except EncordToolError as exc:
        _fail(str(exc))
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        _fail(f"encord verify failed: {exc}")

    payload = report.model_dump(by_alias=True)
    _emit(
        payload,
        output=output,
        text=(
            f"roundtrip {report.status}: {report.matched}/{report.expected} matched, "
            f"{report.checksum_verified} checksum-verified, "
            f"{report.checksum_unavailable} unavailable; report: {report.report_uri}"
        ),
    )


@app.command("cleanup")
def cleanup_cmd(
    title_prefix: str = typer.Option(
        ...,
        "--title-prefix",
        help="Delete Encord folders/collections/presets whose title starts "
        "with this run-scoped prefix (e.g. npa-e2e- or npa-demo-src-). "
        "Minimum 4 characters.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="List what would be deleted without deleting."
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.json, "--output", help="Output format."
    ),
) -> None:
    """Tear down run-scoped Encord state created by push/curate/seed-demo."""

    from npa.sdk.workbench.encord import cleanup as sdk_cleanup
    from npa.workbench.encord.schemas import EncordToolError

    try:
        summary = sdk_cleanup(title_prefix=title_prefix, dry_run=dry_run)
    except EncordToolError as exc:
        _fail(str(exc))
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        _fail(f"encord cleanup failed: {exc}")

    verb = "would delete" if dry_run else "deleted"
    _emit(
        summary,
        output=output,
        text=(
            f"{verb} {len(summary['folders_deleted'])} folder(s) "
            f"({summary['items_deleted']} item(s)), "
            f"{len(summary['collections_deleted'])} collection(s), "
            f"{len(summary['presets_deleted'])} preset(s); "
            f"{len(summary['datasets_undeletable'])} dataset(s) need app-side "
            "removal (the SDK cannot delete datasets)"
        ),
    )


@app.command("seed-demo")
def seed_demo_cmd(
    media_uri: str = typer.Option(
        ...,
        "--media-uri",
        help="s3:// prefix to stage the packaged demo starter clip under.",
    ),
    dataset: str = typer.Option(
        ...,
        "--dataset",
        help="Run-scoped demo dataset title to create and push into.",
    ),
    active_source_id: str = typer.Option(
        ...,
        "--active-source-id",
        help="The workflow's configured source id; when it differs from "
        "--dataset the operator supplied a curated source and seeding no-ops.",
    ),
    transfer: TransferMode = typer.Option(
        TransferMode.upload, "--transfer", help="Push mode for the demo clip."
    ),
    integration: str = typer.Option(
        "", "--integration", help="Cloud integration title/uuid (register mode only)."
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.json, "--output", help="Output format."
    ),
) -> None:
    """Seed the demo source dataset for encord-cosmos3-augment, or no-op."""

    from npa.cli.path_contract import PathContractError, validate_write_path

    try:
        validate_write_path(media_uri, tool="encord seed-demo", option="--media-uri", required=True)
    except PathContractError as exc:
        _fail(str(exc))

    from npa.workflows.encord_loop import EncordLoopError, seed_demo_source
    from npa.workbench.encord.schemas import EncordToolError

    try:
        summary = seed_demo_source(
            media_uri, dataset, active_source_id,
            transfer=transfer.value, integration=integration,
        )
    except (EncordLoopError, EncordToolError) as exc:
        _fail(str(exc))
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        _fail(f"encord seed-demo failed: {exc}")
    _emit(
        summary,
        output=output,
        text=summary.get("skipped") and f"seed skipped: {summary['skipped']}"
        or f"seeded demo dataset {summary.get('dataset')!r} from the packaged starter clip",
    )
