"""Glue stages for the encord-cosmos3-augment workflow.

The Encord pull stage names media by item uuid (``media/<uuid>__<name>``), which
a workflow spec cannot know in advance, while ``cosmos3 generate`` conditions on
one exact video URI. ``stage_media_for_augment`` bridges the two: it reads the
pull manifest and server-side-copies the selected item to a deterministic URI
the spec can template.
"""

from __future__ import annotations

import json
import sys
from typing import Any
from urllib.parse import urlparse


class EncordLoopError(RuntimeError):
    """Raised when the pull manifest cannot supply the requested media item."""


def _parse_s3(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise EncordLoopError(f"expected an exact s3:// object URI, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def stage_media_for_augment(
    manifest_uri: str,
    dest_uri: str,
    index: str = "0",
    storage_client: Any = None,
) -> dict[str, Any]:
    """Copy the pull manifest's ``index``-th media item to ``dest_uri``.

    Fails closed when the manifest is missing, the index is out of range, or the
    selected item did not transfer successfully — a spec must never condition
    Cosmos on a file that is not really there.
    """

    from npa.clients.storage import StorageClient

    client = storage_client or StorageClient.from_environment()
    got = client.read_bytes_with_etag(manifest_uri)
    if got is None:
        raise EncordLoopError(f"pull manifest not found at {manifest_uri}")
    manifest = json.loads(got[0])
    items = [
        item
        for item in manifest.get("items", [])
        if item.get("transfer") in ("copy", "download") and item.get("media_uri")
    ]
    position = int(index)
    if not items or position >= len(items) or position < 0:
        raise EncordLoopError(
            f"pull manifest has {len(items)} transferred media item(s); "
            f"index {position} is unavailable"
        )
    selected = items[position]
    source_bucket, source_key = _parse_s3(str(selected["media_uri"]))
    dest_bucket, dest_key = _parse_s3(dest_uri)
    client.s3.copy_object(
        Bucket=dest_bucket,
        Key=dest_key,
        CopySource={"Bucket": source_bucket, "Key": source_key},
    )
    summary = {
        "stage": "stage_media_for_augment",
        "item_uuid": selected.get("item_uuid", ""),
        "name": selected.get("name", ""),
        "source_uri": selected["media_uri"],
        "staged_uri": dest_uri,
        "items_available": len(items),
        "index": position,
    }
    print(json.dumps(summary))
    return summary


def seed_demo_source(
    media_prefix_uri: str,
    demo_dataset_title: str,
    active_source_id: str,
    transfer: str = "upload",
    integration: str = "",
    storage_client: Any = None,
) -> dict[str, Any]:
    """Seed the workflow's default demo source into Encord, or skip.

    The committed spec defaults ``encord_source_id`` to ``demo_dataset_title``;
    when an operator overrides it with a real curated Collection/Dataset id this
    stage becomes a no-op, so the default run works out of the box without
    side effects on curated runs. The demo clip is the packaged PAIDF starter
    asset: a pinned, SHA-256-verified public sample (CC-BY-4.0).
    """

    if active_source_id.strip() != demo_dataset_title.strip():
        summary: dict[str, Any] = {
            "stage": "seed_demo_source",
            "skipped": "operator supplied a curated source id",
            "source_id": active_source_id,
        }
        print(json.dumps(summary))
        return summary

    from npa.clients.storage import StorageClient
    from npa.workflows.data_factory_input import _fetch_starter, load_starter_contract

    contract = load_starter_contract()
    local_path, cache_state = _fetch_starter(
        contract, cache_dir=None, offline=None, reporter=lambda line: print(line)
    )
    client = storage_client or StorageClient.from_environment()
    clip_uri = media_prefix_uri.rstrip("/") + "/starter-clip.mp4"
    client.upload_file(str(local_path), clip_uri)

    from npa.sdk.workbench import encord as encord_sdk

    receipt = encord_sdk.push(
        input_path=media_prefix_uri.rstrip("/") + "/",
        integration=integration,
        folder=demo_dataset_title,
        dataset=demo_dataset_title,
        output_path=media_prefix_uri.rstrip("/") + "/push/",
        transfer=transfer,
        workflow_run=demo_dataset_title,
        storage_client=client,
    )
    summary = {
        "stage": "seed_demo_source",
        "cache": cache_state,
        "clip_uri": clip_uri,
        "dataset": demo_dataset_title,
        "units_done": receipt.units_done,
        "attribution": str((contract.get("license") or {}).get("name", "")),
        "asset_sha256": str(contract["integrity"]["sha256"]),
    }
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":  # pragma: no cover - exercised through the spec argv
    stage_media_for_augment(*sys.argv[1:])
