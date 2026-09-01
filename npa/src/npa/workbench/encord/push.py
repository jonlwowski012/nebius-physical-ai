"""Register-in-place or upload push of S3 media into an Encord storage folder.

In register mode bytes never move: the tool lists the S3 prefix, builds public
objectUrls against the configured endpoint, and registers them with Encord
through a cloud integration. In upload mode the bytes are copied into
Encord-hosted storage instead. Either way, items are then explicitly linked to
a dataset (Encord never links automatically), and the receipt is written before
any failure exit so lineage survives fail-closed runs.

Identity is exact (adopted from PR #363): every item is registered with
namespaced ``npa.source_uri`` clientMetadata, and receipt lineage resolves
through that metadata or the item's normalized objectUrl — never through
display names. A write-ahead receipt lands before the first Encord mutation so
even a crash mid-mutation leaves a durable record of intent.
"""

from __future__ import annotations

import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from npa.clients.storage import StorageError, _parse_bucket_uri
from npa.workbench.encord.client import (
    _default_user_client,
    resolve_dataset,
    resolve_domain,
    resolve_folder,
    resolve_integration,
    resolve_public_endpoint,
)
from npa.workbench.encord.identity import (
    canonical_s3_uri,
    identity_metadata,
    resolve_exact_identity,
)
from npa.workbench.encord.integrity import etag_checksum, hash_file
from npa.workbench.encord.schemas import (
    DEFAULT_MEDIA_FILTER,
    DEFAULT_POLL_TIMEOUT_SECONDS,
    DEFAULT_TRANSFER,
    PUSH_RECEIPT_FILENAME,
    EncordToolError,
    PushedItem,
    PushReceipt,
)
from npa.workbench.encord.storage import artifact_uri_for, error_text, finalize_artifact

MEDIA_CATEGORIES: dict[str, str] = {
    ".mp4": "videos",
    ".png": "images",
    ".jpg": "images",
    ".jpeg": "images",
}
# encord==0.1.201 has no cloud-registration category for a raw MCAP file: the
# upload format's `scenes` entries require a per-stream SceneBuilder document,
# not an objectUrl. Until a live spike pins a supported path, .mcap keys are
# discovered and accounted for in the receipt as experimental errors rather
# than sent with a guessed schema.
MCAP_SUFFIX = ".mcap"
MCAP_UNSUPPORTED_ERROR = (
    "MCAP cloud registration is not supported by the pinned encord SDK upload "
    "format (scenes require per-stream assets, not an objectUrl). Tracked as an "
    "experimental follow-up; push videos/images with --media videos-images."
)
FILTER_CATEGORIES: dict[str, dict[str, str]] = {
    "videos-images": MEDIA_CATEGORIES,
    "mcap": {MCAP_SUFFIX: "mcap"},
    "all": {**MEDIA_CATEGORIES, MCAP_SUFFIX: "mcap"},
}
TRANSFER_MODES = ("register", "upload")
BATCH_SIZE = 500
# One short re-list absorbs folder-listing lag right after registration.
IDENTITY_RELIST_DELAY_SECONDS = 5.0


def push_receipt_uri_for(output_path: str) -> str:
    """The exact receipt URI a given --output-path resolves to."""

    return artifact_uri_for(output_path, PUSH_RECEIPT_FILENAME)


def object_url_for(endpoint_url: str, bucket: str, key: str) -> str:
    """Path-style public URL for one object, matching the Encord integration."""

    return f"{endpoint_url.rstrip('/')}/{bucket}/{quote(key, safe='/')}"


def discover_objects(
    storage_client: Any, input_uri: str, media: str
) -> tuple[list[tuple[str, str, int, str]], list[str]]:
    """List (key, category, size, etag) rows under the prefix plus skipped keys."""

    allowed = FILTER_CATEGORIES.get(media)
    if allowed is None:
        raise EncordToolError(
            f"Unknown --media value {media!r}. Choices: "
            f"{', '.join(FILTER_CATEGORIES)}."
        )
    bucket, prefix = _parse_bucket_uri(input_uri)
    entries: list[tuple[str, str, int, str]] = []
    skipped: list[str] = []
    paginator = storage_client.s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            key = obj["Key"]
            if key.endswith("/"):
                continue
            category = allowed.get(Path(key).suffix.lower())
            if category:
                entries.append(
                    (key, category, int(obj.get("Size") or 0), str(obj.get("ETag") or ""))
                )
            else:
                skipped.append(key)
    if not entries:
        raise EncordToolError(
            f"No supported media found under {input_uri} (media filter {media!r})."
        )
    return entries, skipped


def build_upload_json(items: list[PushedItem]) -> dict[str, Any]:
    """Encord upload-format JSON for one registration batch.

    Every entry carries the namespaced npa.source_uri clientMetadata and the
    full object key as title, so identity never rests on a display name.
    """

    payload: dict[str, Any] = {"skip_duplicate_urls": True}
    for item in items:
        payload.setdefault(item.category, []).append(
            {
                "objectUrl": item.object_url,
                "title": item.key,
                "clientMetadata": identity_metadata(item.source_uri),
            }
        )
    return payload


def _chunks(items: list[PushedItem], size: int) -> list[list[PushedItem]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _folder_inventory(folder_obj: Any) -> list[Any]:
    try:
        return list(folder_obj.list_items(page_size=1000, include_client_metadata=True))
    except TypeError:
        return list(folder_obj.list_items(page_size=1000))


def _resolve_identities(
    folder_obj: Any,
    items: list[PushedItem],
    *,
    attempts: int = 2,
    fail_unresolved: bool = False,
) -> int:
    """Resolve each registered item's Encord uuid by exact identity only.

    The folder inventory exposes each item's clientMetadata (npa.source_uri)
    and registered objectUrl; resolution matches on those and nothing else —
    display names, and in particular basenames, are never identity. Items the
    old code registered without metadata still resolve through their
    objectUrl. One short re-list absorbs listing lag right after registration.

    Returns the number of items failed. With ``fail_unresolved`` (the
    post-registration pass) an item that still has no exact identity is an
    error, not a silent gap: an unattributed row could never be linked,
    pulled, or verified.
    """

    failed = 0
    for attempt in range(attempts):
        pending = [
            item
            for item in items
            if item.status == "registered" and not item.item_uuid
        ]
        if not pending:
            return failed
        if attempt:
            time.sleep(IDENTITY_RELIST_DELAY_SECONDS)
        inventory = _folder_inventory(folder_obj)
        for item in pending:
            resolution = resolve_exact_identity(
                source_uri=item.source_uri,
                submitted_object_url=item.object_url,
                candidates=inventory,
            )
            if resolution.resolved:
                item.item_uuid = resolution.item_uuid
                item.identity_signal = resolution.signal
            elif resolution.error_code == "identity_conflict":
                item.status = "error"
                item.error = resolution.error
                failed += 1
    if fail_unresolved:
        for item in items:
            if item.status == "registered" and not item.item_uuid:
                item.status = "error"
                item.error = (
                    "registered but no exact metadata or object URL identity "
                    "matched in the folder inventory"
                )
                failed += 1
    return failed


def _register_items(
    folder_obj: Any,
    items: list[PushedItem],
    *,
    integration_id: str,
    poll_timeout_seconds: int,
) -> tuple[list[tuple[str, str]], int, int, str]:
    """Register objectUrls in batches; return (uuid, name) pairs, counts, status."""

    by_url = {item.object_url: item for item in items}
    done = errors = 0
    status = "done"
    for batch in _chunks(items, BATCH_SIZE):
        job_id = folder_obj.add_private_data_to_folder_start(
            integration_id=integration_id,
            private_files=build_upload_json(batch),
            ignore_errors=True,
        )
        result = folder_obj.add_private_data_to_folder_get_result(
            job_id, timeout_seconds=poll_timeout_seconds
        )
        state = getattr(result.status, "name", str(result.status)).upper()
        done += int(getattr(result, "units_done_count", 0) or 0)
        errors += int(getattr(result, "units_error_count", 0) or 0)
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
    # Lineage is attached afterwards by exact identity (_resolve_identities).
    return done, errors, status


def _upload_items(
    folder_obj: Any,
    items: list[PushedItem],
    *,
    storage_client: Any,
    source_bucket: str,
) -> tuple[list[tuple[str, str]], int, int, str]:
    """Copy each object's bytes into Encord-hosted storage; return uuids + counts.

    Uploads are synchronous per file (the SDK returns the new item uuid), so
    unlike register mode there is no job polling; failures are recorded per
    item and the run still fails closed after the receipt is written.
    """

    done = errors = 0
    for item in items:
        try:
            with tempfile.TemporaryDirectory(prefix="npa-encord-upload-") as tmp:
                local = Path(tmp) / Path(item.key).name
                storage_client.download_file(f"s3://{source_bucket}/{item.key}", str(local))
                # The bytes are local anyway: record their content digest so
                # the roundtrip verifier can compare pulled bytes exactly.
                digest = hash_file(local)
                item.source_size = digest.size
                item.source_checksum = digest.sha256
                item.source_checksum_kind = "sha256"
                metadata = identity_metadata(item.source_uri)
                upload = (
                    folder_obj.upload_video
                    if item.category == "videos"
                    else folder_obj.upload_image
                )
                try:
                    item_uuid = upload(
                        str(local), title=item.key, client_metadata=metadata
                    )
                except TypeError:
                    item_uuid = upload(str(local), title=item.key)
            item.item_uuid = str(item_uuid)
            item.identity_signal = "uploaded"
            item.status = "uploaded"
            done += 1
        except Exception as exc:  # noqa: BLE001 - recorded per item, run fails closed
            item.status = "error"
            item.error = str(exc)
            errors += 1
    return done, errors, "done"


def _link_dataset(dataset_obj: Any, items: list[PushedItem]) -> int:
    """Link this push's identity-resolved items into the dataset.

    Register mode with skip_duplicate_urls reports only newly added items, so
    a re-push would otherwise link nothing — but by link time every item
    (fresh or pre-existing) carries the uuid exact identity resolved, and
    link_items skips already-linked uuids server-side, so this is idempotent.
    """

    uuids = sorted({item.item_uuid for item in items if item.item_uuid})
    if not uuids:
        return 0
    dataset_obj.link_items(uuids)
    return len(uuids)


def run_push(
    *,
    input_path: str,
    integration: str,
    folder: str,
    output_path: str,
    dataset: str = "",
    media: str = DEFAULT_MEDIA_FILTER,
    transfer: str = DEFAULT_TRANSFER,
    poll_timeout_seconds: int = DEFAULT_POLL_TIMEOUT_SECONDS,
    workflow_run: str = "",
    user_client: Any = None,
    storage_client: Any = None,
    environ: dict[str, str] | None = None,
) -> PushReceipt:
    """Push the prefix into Encord, link a dataset, and write the receipt."""

    if transfer not in TRANSFER_MODES:
        raise EncordToolError(
            f"Unknown --transfer value {transfer!r}. Choices: {', '.join(TRANSFER_MODES)}."
        )
    try:
        bucket, _ = _parse_bucket_uri(input_path)
    except StorageError:
        raise EncordToolError(
            "--input-path must be an s3:// prefix: Encord media is pushed from "
            "object storage, not local paths."
        ) from None
    from npa.clients.storage import StorageClient

    active_storage = storage_client or StorageClient.from_environment()
    # objectUrls (and therefore a public endpoint) exist only in register mode;
    # upload mode moves the bytes themselves.
    endpoint_url = resolve_public_endpoint(environ) if transfer == "register" else ""

    entries, skipped = discover_objects(active_storage, input_path, media)
    items = []
    for key, category, size, etag in entries:
        checksum, checksum_kind = etag_checksum(etag)
        items.append(
            PushedItem(
                key=key,
                source_etag=etag.strip().strip('"'),
                source_uri=canonical_s3_uri(bucket, key),
                object_url=(
                    object_url_for(endpoint_url, bucket, key)
                    if transfer == "register" and category != "mcap"
                    else ""
                ),
                category=category,
                source_size=size,
                source_checksum=checksum,
                source_checksum_kind=checksum_kind,
                status="experimental_error" if category == "mcap" else "registered",
                error=MCAP_UNSUPPORTED_ERROR if category == "mcap" else "",
            )
        )
    registrable = [item for item in items if item.status == "registered"]

    client = user_client if user_client is not None else _default_user_client(environ)
    integration_id = integration_title = ""
    if transfer == "register":
        integration_id, integration_title = resolve_integration(client, integration)

    # Write-ahead receipt: land the plan before the first Encord mutation, so
    # even a crash mid-mutation leaves a durable record of what was attempted.
    receipt_uri = push_receipt_uri_for(output_path)
    planned = PushReceipt(
        generated_at=datetime.now(timezone.utc).isoformat(),
        workflow_run=workflow_run,
        input_uri=input_path,
        endpoint_url=endpoint_url,
        encord_domain=resolve_domain(environ),
        transfer=transfer,
        integration_id=integration_id,
        integration_title=integration_title,
        folder_name=folder.strip(),
        media_filter=media,
        status="planned",
        files_discovered=len(items),
        receipt_uri=receipt_uri,
        items=items,
        skipped_unsupported=skipped,
    )
    finalize_artifact(
        planned,
        result_uri=receipt_uri,
        filename=PUSH_RECEIPT_FILENAME,
        storage_client=active_storage,
        run_error=None,
        failure_prefix="Encord push failed",
    )
    # From this point on Encord may be mutated (folder/dataset creation).  Keep
    # every such operation inside the receipt-finalization path below: a failed
    # dataset create must not strand a newly-created folder without lineage.
    folder_obj: Any = None
    folder_created = False
    dataset_obj = None
    dataset_hash = dataset_title = ""
    dataset_created = False
    # Everything in this block can mutate Encord; the receipt must land even
    # when folder/dataset setup or a later transfer step throws.
    units_done = units_error = 0
    status = "failed"
    linked_count = 0
    run_error: Exception | None = None
    try:
        folder_obj, folder_created = resolve_folder(client, folder)
        if dataset.strip():
            dataset_obj, dataset_hash, dataset_title, dataset_created = resolve_dataset(
                client, dataset
            )
        # Caller-side idempotency: resolve exact identities BEFORE transferring
        # anything, so a retried stage re-sends only what does not already
        # exist. This is our invariant, not the SaaS's skip_duplicate_urls —
        # and it makes upload-mode re-pushes no-ops instead of duplicate
        # byte copies.
        units_error += _resolve_identities(folder_obj, registrable, attempts=1)
        pending = [
            item
            for item in registrable
            if item.status == "registered" and not item.item_uuid
        ]
        preexisting = sum(1 for item in registrable if item.item_uuid)
        if transfer == "upload":
            for item in registrable:
                if item.item_uuid:
                    item.status = "uploaded"
            units_done, units_error_new, status = _upload_items(
                folder_obj,
                pending,
                storage_client=active_storage,
                source_bucket=bucket,
            )
            units_error += units_error_new
        elif pending:
            units_done, units_error_new, status = _register_items(
                folder_obj,
                pending,
                integration_id=integration_id,
                poll_timeout_seconds=poll_timeout_seconds,
            )
            units_error += units_error_new
            # Exact identity resolution (metadata/objectUrl only) attaches the
            # Encord uuid for the freshly registered items; an item Encord
            # accepted but exact identity cannot attribute fails closed.
            units_error += _resolve_identities(
                folder_obj, pending, fail_unresolved=status == "done"
            )
        else:
            status = "done"
        units_done += preexisting

        if dataset_obj is not None:
            linked_count = _link_dataset(dataset_obj, items)
    except Exception as exc:  # noqa: BLE001 - recorded in the receipt, re-raised below
        run_error = exc
        status = "failed"

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
        folder_uuid=str(getattr(folder_obj, "uuid", "")),
        # Preserve the requested title when Encord failed before it could be
        # resolved, while recording the canonical title once it exists.
        folder_name=str(getattr(folder_obj, "name", folder.strip())),
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
        error=error_text(run_error),
        receipt_uri=receipt_uri,
        items=items,
        skipped_unsupported=skipped,
    )
    finalize_artifact(
        receipt,
        result_uri=receipt_uri,
        filename=PUSH_RECEIPT_FILENAME,
        storage_client=active_storage,
        run_error=run_error,
        failure_prefix="Encord push failed",
    )
    if status != "done":
        raise EncordToolError(
            f"Encord push {status}: {units_error} unit error(s), {units_done} "
            f"registered. Receipt written to {receipt_uri}."
        )
    return receipt
