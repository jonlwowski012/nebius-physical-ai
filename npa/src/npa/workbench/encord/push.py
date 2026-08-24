"""Register-in-place push of S3 media into an Encord storage folder.

Bytes never move: the tool lists the S3 prefix, builds public objectUrls against
the configured endpoint, and registers them with Encord through a cloud
integration. Items are then explicitly linked to a dataset (Encord never links
automatically). The receipt is always written before a failure exit so lineage
survives fail-closed runs.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

from npa.workbench.encord.client import (
    _default_user_client,
    resolve_dataset,
    resolve_domain,
    resolve_folder,
    resolve_integration,
    resolve_public_endpoint,
)
from npa.workbench.encord.schemas import (
    PUSH_RECEIPT_FILENAME,
    EncordToolError,
    PushedItem,
    PushReceipt,
)

MEDIA_CATEGORIES: dict[str, str] = {
    ".mp4": "videos",
    ".png": "images",
    ".jpg": "images",
    ".jpeg": "images",
}
MEDIA_FILTERS = ("videos-images", "mcap", "all")
TRANSFER_MODES = ("register", "upload")
# encord==0.1.201 has no cloud-registration category for a raw MCAP file: the
# upload format's `scenes` entries require a per-stream SceneBuilder document,
# not an objectUrl. Until the live spike (S1) pins a supported path, .mcap keys
# are discovered and accounted for in the receipt as experimental errors rather
# than sent with a guessed schema.
MCAP_SUFFIX = ".mcap"
MCAP_UNSUPPORTED_ERROR = (
    "MCAP cloud registration is not supported by the pinned encord SDK upload "
    "format (scenes require per-stream assets, not an objectUrl). Tracked as an "
    "experimental follow-up; push videos/images with --media videos-images."
)
BATCH_SIZE = 500


def push_receipt_uri_for(output_path: str) -> str:
    """The exact receipt URI a given --output-path resolves to."""

    if output_path.endswith(".json"):
        return output_path
    return output_path.rstrip("/") + f"/{PUSH_RECEIPT_FILENAME}"


def object_url_for(endpoint_url: str, bucket: str, key: str) -> str:
    """Path-style public URL for one object, matching the Encord integration."""

    return f"{endpoint_url.rstrip('/')}/{bucket}/{quote(key, safe='/')}"


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise EncordToolError(f"Expected an s3:// URI, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def discover_objects(
    storage_client: Any, input_uri: str, media: str
) -> tuple[list[tuple[str, str]], list[str]]:
    """List (key, category) pairs under the prefix plus skipped keys."""

    if media not in MEDIA_FILTERS:
        raise EncordToolError(
            f"Unknown --media value {media!r}. Choices: {', '.join(MEDIA_FILTERS)}."
        )
    allowed = dict(MEDIA_CATEGORIES)
    if media == "mcap":
        allowed = {MCAP_SUFFIX: "mcap"}
    elif media == "all":
        allowed[MCAP_SUFFIX] = "mcap"

    bucket, prefix = _parse_s3_uri(input_uri)
    entries: list[tuple[str, str]] = []
    skipped: list[str] = []
    paginator = storage_client.s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            key = obj["Key"]
            if key.endswith("/"):
                continue
            category = allowed.get(Path(key).suffix.lower())
            if category:
                entries.append((key, category))
            else:
                skipped.append(key)
    if not entries:
        raise EncordToolError(
            f"No supported media found under {input_uri} (media filter {media!r})."
        )
    return entries, skipped


def build_upload_json(items: list[PushedItem]) -> dict[str, Any]:
    """Encord upload-format JSON for one registration batch."""

    payload: dict[str, Any] = {"skip_duplicate_urls": True}
    for item in items:
        payload.setdefault(item.category, []).append({"objectUrl": item.object_url})
    return payload


def _chunks(items: list[PushedItem], size: int) -> list[list[PushedItem]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _write_json(
    payload: dict[str, Any],
    *,
    result_uri: str,
    filename: str,
    storage_client: Any,
) -> str:
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if result_uri.startswith("s3://"):
        from npa.clients.storage import StorageClient

        client = storage_client or StorageClient.from_environment()
        with tempfile.TemporaryDirectory(prefix="npa-encord-") as tmp:
            local_path = Path(tmp) / filename
            local_path.write_text(body, encoding="utf-8")
            return client.upload_file(str(local_path), result_uri)

    path = Path(result_uri)
    if path.suffix != ".json":
        path = path / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return str(path)


def _upload_items(
    folder_obj: Any,
    items: list[PushedItem],
    *,
    storage_client: Any,
    source_bucket: str,
) -> tuple[list[tuple[str, str]], int, int]:
    """Copy each object's bytes into Encord-hosted storage; return uuids + counts.

    Uploads are synchronous per file (the SDK returns the new item uuid), so
    unlike register mode there is no job polling; failures are recorded per
    item and the run still fails closed after the receipt is written.
    """

    uploaded: list[tuple[str, str]] = []
    done = errors = 0
    for item in items:
        try:
            with tempfile.TemporaryDirectory(prefix="npa-encord-upload-") as tmp:
                local = Path(tmp) / Path(item.key).name
                storage_client.download_file(f"s3://{source_bucket}/{item.key}", str(local))
                if item.category == "videos":
                    item_uuid = folder_obj.upload_video(str(local), title=item.key)
                else:
                    item_uuid = folder_obj.upload_image(str(local), title=item.key)
            item.item_uuid = str(item_uuid)
            item.status = "uploaded"
            uploaded.append((str(item_uuid), item.key))
            done += 1
        except Exception as exc:  # noqa: BLE001 - recorded per item, run fails closed
            item.status = "error"
            item.error = str(exc)
            errors += 1
    return uploaded, done, errors


def run_push(
    *,
    input_path: str,
    integration: str,
    folder: str,
    output_path: str,
    dataset: str = "",
    media: str = "videos-images",
    transfer: str = "register",
    poll_timeout_seconds: int = 1800,
    workflow_run: str = "",
    user_client: Any = None,
    storage_client: Any = None,
    environ: dict[str, str] | None = None,
) -> PushReceipt:
    """Register the prefix in Encord, link a dataset, and write the receipt."""

    if not input_path.startswith("s3://"):
        raise EncordToolError(
            "--input-path must be an s3:// prefix: Encord registers data by URL, "
            "so local paths cannot be pushed in place."
        )
    if transfer not in TRANSFER_MODES:
        raise EncordToolError(
            f"Unknown --transfer value {transfer!r}. Choices: {', '.join(TRANSFER_MODES)}."
        )
    from npa.clients.storage import StorageClient

    active_storage = storage_client or StorageClient.from_environment()
    # objectUrls (and therefore a public endpoint) exist only in register mode;
    # upload mode moves the bytes themselves.
    endpoint_url = resolve_public_endpoint(environ) if transfer == "register" else ""
    bucket, _ = _parse_s3_uri(input_path)

    entries, skipped = discover_objects(active_storage, input_path, media)
    items: list[PushedItem] = []
    for key, category in entries:
        if category == "mcap":
            items.append(
                PushedItem(
                    key=key,
                    object_url=object_url_for(endpoint_url, bucket, key),
                    category="mcap",
                    status="experimental_error",
                    error=MCAP_UNSUPPORTED_ERROR,
                )
            )
        else:
            items.append(
                PushedItem(
                    key=key,
                    object_url=(
                        object_url_for(endpoint_url, bucket, key)
                        if transfer == "register"
                        else ""
                    ),
                    category=category,
                )
            )
    registrable = [item for item in items if item.status == "registered"]
    by_url = {item.object_url: item for item in items}

    client = user_client if user_client is not None else _default_user_client(environ)
    integration_id = integration_title = ""
    if transfer == "register":
        integration_id, integration_title = resolve_integration(client, integration)
    folder_obj, folder_created = resolve_folder(client, folder)
    dataset_obj = None
    dataset_hash = dataset_title = ""
    dataset_created = False
    if dataset.strip():
        dataset_obj, dataset_hash, dataset_title, dataset_created = resolve_dataset(
            client, dataset
        )

    status = "done"
    units_done = units_error = 0
    registered: list[tuple[str, str]] = []  # (item_uuid, name)
    if transfer == "upload":
        registered, units_done, units_error = _upload_items(
            folder_obj,
            registrable,
            storage_client=active_storage,
            source_bucket=bucket,
        )
    for batch in _chunks(registrable if transfer == "register" else [], BATCH_SIZE):
        job_id = folder_obj.add_private_data_to_folder_start(
            integration_id=integration_id,
            private_files=build_upload_json(batch),
            ignore_errors=True,
        )
        result = folder_obj.add_private_data_to_folder_get_result(
            job_id, timeout_seconds=poll_timeout_seconds
        )
        state = getattr(result.status, "name", str(result.status)).upper()
        units_done += int(getattr(result, "units_done_count", 0) or 0)
        units_error += int(getattr(result, "units_error_count", 0) or 0)
        for entry in getattr(result, "items_with_names", None) or []:
            registered.append((str(entry.item_uuid), str(entry.name)))
        for unit_error in getattr(result, "unit_errors", None) or []:
            for url in getattr(unit_error, "object_urls", None) or []:
                matched = by_url.get(str(url))
                if matched is not None:
                    matched.status = "error"
                    matched.error = str(getattr(unit_error, "error", "") or "unit error")
        if state == "PENDING":
            status = "timeout"
            break
        if state in ("ERROR", "CANCELLED"):
            status = "failed"
            break

    # Attach item uuids to receipt rows. Encord names registered items by the
    # full object key (observed live), so match that first and fall back to the
    # basename; ambiguity leaves the row blank without affecting linking, which
    # uses the uuid list directly.
    by_name: dict[str, list[str]] = {}
    for item_uuid, name in registered:
        by_name.setdefault(name, []).append(item_uuid)
    for item in items:
        candidates = by_name.get(item.key) or by_name.get(Path(item.key).name) or []
        if len(candidates) == 1 and item.status == "registered":
            item.item_uuid = candidates[0]

    linked_count = 0
    if dataset_obj is not None and registered:
        dataset_obj.link_items([item_uuid for item_uuid, _ in registered])
        linked_count = len(registered)

    if any(item.status == "experimental_error" for item in items):
        units_error += sum(1 for item in items if item.status == "experimental_error")
    if status == "done" and units_error > 0:
        status = "failed"

    receipt_uri = push_receipt_uri_for(output_path)
    receipt = PushReceipt(
        generated_at=datetime.now(timezone.utc).isoformat(),
        workflow_run=workflow_run,
        input_uri=input_path,
        endpoint_url=endpoint_url,
        encord_domain=resolve_domain(environ),
        transfer=transfer,
        integration_id=integration_id,
        integration_title=integration_title,
        folder_uuid=str(folder_obj.uuid),
        folder_name=str(folder_obj.name),
        folder_created=folder_created,
        dataset_hash=dataset_hash,
        dataset_title=dataset_title,
        dataset_created=dataset_created,
        linked_count=linked_count,
        media_filter=media,
        status=status,
        files_discovered=len(items),
        units_done=units_done,
        units_error=units_error,
        receipt_uri=receipt_uri,
        items=items,
        skipped_unsupported=skipped,
    )
    _write_json(
        receipt.model_dump(by_alias=True),
        result_uri=receipt_uri,
        filename=PUSH_RECEIPT_FILENAME,
        storage_client=active_storage,
    )
    if status != "done" or units_error > 0:
        raise EncordToolError(
            f"Encord push {status}: {units_error} unit error(s), {units_done} "
            f"registered. Receipt written to {receipt_uri}."
        )
    return receipt
