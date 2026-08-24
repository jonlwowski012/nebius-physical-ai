"""Pull curated Encord data (media + item JSON + labels) back to S3.

For register-in-place data the common case is a zero-egress server-side copy:
each item's signed URL is parsed, and when it points back at the configured
endpoint the object is copied bucket-to-bucket instead of round-tripping bytes.
The manifest is always written before a failure exit.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from npa.workbench.encord.client import (
    _default_user_client,
    resolve_collection,
    resolve_dataset,
    resolve_domain,
    resolve_project,
    resolve_public_endpoint,
)
from npa.workbench.encord.push import _parse_s3_uri, _write_json
from npa.workbench.encord.schemas import (
    PULL_MANIFEST_FILENAME,
    EncordToolError,
    PulledItem,
    PullManifest,
)

PULL_SOURCES = ("collection", "dataset", "project")
LABEL_BUNDLE_SIZE = 100


def pull_manifest_uri_for(output_path: str) -> str:
    """The exact manifest URI a given --output-path resolves to."""

    return output_path.rstrip("/") + f"/{PULL_MANIFEST_FILENAME}"


def _sanitize_name(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "._-" else "_" for c in name.strip())
    return cleaned or "item"


def enumerate_items(
    user_client: Any, *, source: str, source_id: str
) -> tuple[str, str, list[Any], Any]:
    """Resolve the source and list its storage items.

    Returns (resolved_source_id, source_name, storage_items, project_or_none).
    """

    if source == "collection":
        collection, resolved_id, name = resolve_collection(user_client, source_id)
        return resolved_id, name, list(collection.list_items(page_size=1000)), None
    if source == "dataset":
        dataset, dataset_hash, title, _ = resolve_dataset(
            user_client, source_id, create=False
        )
        backing = [
            row.backing_item_uuid
            for row in dataset.data_rows
            if getattr(row, "backing_item_uuid", None)
        ]
        items = user_client.get_storage_items(backing, sign_url=True) if backing else []
        return dataset_hash, title, list(items), None
    if source == "project":
        project, project_hash, title = resolve_project(user_client, source_id)
        label_rows = project.list_label_rows_v2()
        backing = [
            row.backing_item_uuid
            for row in label_rows
            if getattr(row, "backing_item_uuid", None)
        ]
        items = user_client.get_storage_items(backing, sign_url=True) if backing else []
        return project_hash, title, list(items), project
    raise EncordToolError(
        f"Unknown --source value {source!r}. Choices: {', '.join(PULL_SOURCES)}."
    )


def _same_endpoint_source(signed_url: str, endpoint_url: str) -> tuple[str, str] | None:
    """(bucket, key) when the signed URL points at our own endpoint, else None."""

    signed = urlparse(signed_url)
    endpoint = urlparse(endpoint_url)
    if not signed.netloc or not endpoint.netloc:
        return None
    path = unquote(signed.path.lstrip("/"))
    if signed.netloc == endpoint.netloc:
        # Path-style: /<bucket>/<key>
        bucket, _, key = path.partition("/")
        return (bucket, key) if bucket and key else None
    if signed.netloc.endswith(f".{endpoint.netloc}"):
        # Virtual-hosted: <bucket>.<endpoint>/<key>
        bucket = signed.netloc[: -len(endpoint.netloc) - 1]
        return (bucket, path) if bucket and path else None
    return None


def transfer_item(
    item: Any,
    *,
    storage_client: Any,
    output_uri: str,
    endpoint_url: str,
) -> PulledItem:
    """Copy or download one storage item into the output prefix."""

    pulled = PulledItem(
        item_uuid=str(item.uuid),
        name=str(item.name),
        item_type=str(getattr(item, "item_type", "") or ""),
        mime_type=str(getattr(item, "mime_type", "") or ""),
        file_size=int(getattr(item, "file_size", 0) or 0),
    )
    dest_uri = (
        output_uri.rstrip("/")
        + f"/media/{pulled.item_uuid}__{_sanitize_name(pulled.name)}"
    )
    dest_bucket, dest_key = _parse_s3_uri(dest_uri)

    signed_url = item.get_signed_url()
    if not signed_url:
        pulled.transfer = "error"
        pulled.error = "item has no signed URL (composite items are not supported)"
        return pulled

    source = _same_endpoint_source(signed_url, endpoint_url)
    if source is not None:
        try:
            storage_client.s3.copy_object(
                Bucket=dest_bucket,
                Key=dest_key,
                CopySource={"Bucket": source[0], "Key": source[1]},
            )
            pulled.media_uri = dest_uri
            pulled.transfer = "copy"
            return pulled
        except Exception:  # noqa: BLE001 - fall back to the download path
            source = None

    import httpx

    try:
        for attempt in (1, 2):
            try:
                with tempfile.TemporaryDirectory(prefix="npa-encord-pull-") as tmp:
                    local = Path(tmp) / _sanitize_name(pulled.name)
                    with httpx.stream(
                        "GET", signed_url, timeout=600.0, follow_redirects=True
                    ) as response:
                        response.raise_for_status()
                        with local.open("wb") as handle:
                            for chunk in response.iter_bytes(8 * 1024 * 1024):
                                handle.write(chunk)
                    storage_client.upload_file(str(local), dest_uri)
                pulled.media_uri = dest_uri
                pulled.transfer = "download"
                return pulled
            except httpx.HTTPStatusError:
                if attempt == 2:
                    raise
                # The signed URL may simply have expired mid-run; refetch once.
                signed_url = item.get_signed_url(refetch=True) or signed_url
    except Exception as exc:  # noqa: BLE001 - recorded per item, run fails closed
        pulled.transfer = "error"
        pulled.error = str(exc)
    return pulled


def export_labels(
    project: Any,
    *,
    output_uri: str,
    storage_client: Any,
) -> tuple[int, list[str]]:
    """Export every label row as Encord JSON under labels/."""

    label_rows = list(project.list_label_rows_v2())
    if not label_rows:
        return 0, []
    with project.create_bundle(bundle_size=LABEL_BUNDLE_SIZE) as bundle:
        for row in label_rows:
            row.initialise_labels(bundle=bundle)
    label_uris: list[str] = []
    for row in label_rows:
        name = str(row.label_hash or row.data_hash or len(label_uris))
        label_uri = output_uri.rstrip("/") + f"/labels/{name}.json"
        _write_json(
            row.to_encord_dict(),
            result_uri=label_uri,
            filename=f"{name}.json",
            storage_client=storage_client,
        )
        label_uris.append(label_uri)
    return len(label_rows), label_uris


def run_pull(
    *,
    source: str,
    source_id: str,
    output_path: str,
    workflow_run: str = "",
    user_client: Any = None,
    storage_client: Any = None,
    environ: dict[str, str] | None = None,
) -> PullManifest:
    """Materialize a curated Encord source into the S3 output prefix."""

    if not output_path.startswith("s3://"):
        raise EncordToolError("--output-path must be an s3:// prefix.")
    from npa.clients.storage import StorageClient

    active_storage = storage_client or StorageClient.from_environment()
    endpoint_url = resolve_public_endpoint(environ)
    client = user_client if user_client is not None else _default_user_client(environ)

    resolved_id, source_name, storage_items, project = enumerate_items(
        client, source=source, source_id=source_id
    )

    pulled: list[PulledItem] = []
    media_bytes = 0
    for item in storage_items:
        record = transfer_item(
            item,
            storage_client=active_storage,
            output_uri=output_path,
            endpoint_url=endpoint_url,
        )
        _write_json(
            {
                "item_uuid": record.item_uuid,
                "name": record.name,
                "item_type": record.item_type,
                "mime_type": record.mime_type,
                "file_size": record.file_size,
                "client_metadata": getattr(item, "client_metadata", None) or {},
            },
            result_uri=output_path.rstrip("/") + f"/items/{record.item_uuid}.json",
            filename=f"{record.item_uuid}.json",
            storage_client=active_storage,
        )
        media_bytes += record.file_size if record.transfer in ("copy", "download") else 0
        pulled.append(record)

    label_rows = 0
    label_uris: list[str] = []
    if project is not None:
        label_rows, label_uris = export_labels(
            project, output_uri=output_path, storage_client=active_storage
        )

    manifest_uri = pull_manifest_uri_for(output_path)
    manifest = PullManifest(
        generated_at=datetime.now(timezone.utc).isoformat(),
        workflow_run=workflow_run,
        encord_domain=resolve_domain(environ),
        source_kind=source,
        source_id=resolved_id,
        source_name=source_name,
        output_uri=output_path,
        manifest_uri=manifest_uri,
        items_total=len(pulled),
        media_copied=sum(1 for record in pulled if record.transfer == "copy"),
        media_downloaded=sum(1 for record in pulled if record.transfer == "download"),
        media_failed=sum(1 for record in pulled if record.transfer == "error"),
        label_rows=label_rows,
        media_bytes=media_bytes,
        label_uris=label_uris,
        items=pulled,
    )
    _write_json(
        manifest.model_dump(by_alias=True),
        result_uri=manifest_uri,
        filename=PULL_MANIFEST_FILENAME,
        storage_client=active_storage,
    )
    if manifest.media_failed > 0:
        raise EncordToolError(
            f"Encord pull failed for {manifest.media_failed} of "
            f"{manifest.items_total} item(s). Manifest written to {manifest_uri}."
        )
    if manifest.items_total == 0:
        raise EncordToolError(
            f"Encord {source} {source_id!r} contains no storage items; nothing "
            f"was pulled. Manifest written to {manifest_uri}."
        )
    return manifest
