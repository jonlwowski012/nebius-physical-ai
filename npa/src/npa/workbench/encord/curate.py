"""Headless curation: server-side quality-filter selection into a Collection.

The workbench declares filters over Encord's built-in data quality metrics
(``brightness:0.2:0.8``); curate maps them onto a run-scoped Encord filter
preset, creates/resolves the target Collection, and asks Encord to populate it
server-side (``add_preset_items``). No media bytes move, and no human clicks in
the Encord app. The existing ``pull --source collection`` consumes the result.

The per-metric filter payload is not documented or typed by the Encord SDK, so
only shapes pinned by the live spike are ever sent (an unpinned shape makes the
server-side evaluation request block indefinitely). Pinned by live evidence:

    {"include": bool, "values": [min, max], "domain": "data",
     "metric": "<metric_id>", "type": "metric"}

inside ``{"global_filters": {"filters": [...]}}``. Intrinsic metrics (width,
height, area, aspect-ratio) evaluate immediately on any folder; computed ones
(brightness, sharpness, file-size) evaluate only after quality metrics have
been computed for the folder — a one-time action in the Encord app that has no
public API. Unknown metric names fail closed before any Encord call.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from npa.workbench.encord.client import (
    _default_user_client,
    resolve_collection,
    resolve_domain,
    resolve_folder,
)
from npa.workbench.encord.schemas import (
    CURATE_RECEIPT_FILENAME,
    DEFAULT_CURATE_POLL_SECONDS,
    CurateFilter,
    CurateReceipt,
    EncordToolError,
)
from npa.workbench.encord.storage import artifact_uri_for, error_text, finalize_artifact

# CLI metric name -> (Encord metric id, requires computed quality metrics).
# Every id here was verified live: the preset is accepted and evaluated (an
# unpinned shape blocks server-side). Do not add names without re-running the
# live verification — see the module docstring.
METRIC_FILTERS: dict[str, tuple[str, bool]] = {
    "width": ("metric_width", False),
    "height": ("metric_height", False),
    "area": ("metric_area", False),
    "aspect-ratio": ("metric_aspect_ratio", False),
    "brightness": ("metric_brightness", True),
    "sharpness": ("metric_sharpness", True),
    "file-size": ("metric_file_size", True),
}
POLL_INTERVAL_SECONDS = 5.0
# Once the count changes, confirm stability quickly instead of waiting a full
# poll interval — small selections settle in a second or two.
CONFIRM_INTERVAL_SECONDS = 1.0
# add_preset_items evaluates once, server-side: items uploaded moments before
# (the push -> curate composition) may not be metric-indexed yet and are then
# missed by that evaluation, observed live. While the selection is empty the
# evaluation is re-issued on this cadence until items appear or time runs out.
REISSUE_INTERVAL_SECONDS = 15.0


def curate_receipt_uri_for(output_path: str) -> str:
    """The exact receipt URI a given --output-path resolves to."""

    return artifact_uri_for(output_path, CURATE_RECEIPT_FILENAME)


def parse_filter_specs(specs: list[str]) -> list[CurateFilter]:
    """Parse ``metric:min:max`` specs (comma-separable) into typed filters."""

    parsed: list[CurateFilter] = []
    flat = [part.strip() for spec in specs for part in spec.split(",") if part.strip()]
    if not flat:
        raise EncordToolError(
            "At least one --filter metric:min:max is required: an empty filter "
            "set would select the entire folder."
        )
    for spec in flat:
        pieces = spec.split(":")
        if len(pieces) != 3:
            raise EncordToolError(
                f"Invalid --filter {spec!r}: expected metric:min:max, e.g. "
                "brightness:0.2:0.8."
            )
        name, low_raw, high_raw = (piece.strip() for piece in pieces)
        entry = METRIC_FILTERS.get(name)
        if entry is None:
            raise EncordToolError(
                f"Unknown filter metric {name!r}. Supported: "
                f"{', '.join(sorted(METRIC_FILTERS))}. Only metrics with a "
                "live-verified Encord filter shape are allowed."
            )
        try:
            low, high = float(low_raw), float(high_raw)
        except ValueError:
            raise EncordToolError(
                f"Invalid --filter {spec!r}: min and max must be numbers."
            ) from None
        if low > high:
            raise EncordToolError(f"Invalid --filter {spec!r}: min exceeds max.")
        encord_metric, computed = entry
        parsed.append(
            CurateFilter(
                metric=name,
                encord_metric=encord_metric,
                min=low,
                max=high,
                computed=computed,
            )
        )
    return parsed


def build_filter_preset_json(filters: list[CurateFilter]) -> dict[str, Any]:
    """The exact create_preset payload for the parsed filters (pinned shape)."""

    return {
        "global_filters": {
            "filters": [
                {
                    "include": True,
                    "values": [filt.min, filt.max],
                    # "data" is the Index-level domain; "frame" (Active) blocks.
                    "domain": "data",
                    "metric": filt.encord_metric,
                    "type": "metric",
                }
                for filt in filters
            ]
        }
    }


def _poll_selection(
    collection: Any, *, preset_uuid: str, poll_seconds: float
) -> tuple[int, str]:
    """Poll the async server-side evaluation -> (items_selected, status).

    add_preset_items returns before evaluation finishes and exposes no job
    handle, so completion is inferred: a non-zero count that is stable across
    two consecutive polls is done. While the count is zero the one-shot
    evaluation is re-issued (freshly pushed items become metric-indexed a
    little after upload). A count still at zero when the timeout expires is
    indistinguishable from a genuinely empty selection and is reported as
    empty (the caller fails closed either way).
    """

    deadline = time.monotonic() + poll_seconds
    last_issue = time.monotonic()
    previous = -1
    while True:
        count = sum(1 for _ in collection.list_items(page_size=1000))
        if count > 0 and count == previous:
            return count, "done"
        if time.monotonic() >= deadline:
            return count, "empty" if count == 0 else "timeout"
        if count == 0 and time.monotonic() - last_issue >= REISSUE_INTERVAL_SECONDS:
            collection.add_preset_items(preset_uuid)
            last_issue = time.monotonic()
        interval = CONFIRM_INTERVAL_SECONDS if count != previous else POLL_INTERVAL_SECONDS
        previous = count
        time.sleep(min(interval, max(deadline - time.monotonic(), 0.1)))


def _zero_selection_remedy(filters: list[CurateFilter]) -> str:
    remedy = (
        "No items matched the filters. Check the ranges against the folder's "
        "media"
    )
    if any(filt.computed for filt in filters):
        computed = ", ".join(sorted(f.metric for f in filters if f.computed))
        remedy += (
            f"; note that {computed} are computed quality metrics, which "
            "evaluate to no matches until quality metrics have been computed "
            "for the folder (a one-time action in the Encord app with no "
            "public API). Intrinsic metrics (width, height, area, "
            "aspect-ratio) work on any folder"
        )
    return remedy + "."


def run_curate(
    *,
    folder: str,
    filters: list[str],
    collection: str,
    output_path: str,
    workflow_run: str = "",
    poll_seconds: float = DEFAULT_CURATE_POLL_SECONDS,
    user_client: Any = None,
    storage_client: Any = None,
    environ: dict[str, str] | None = None,
) -> CurateReceipt:
    """Curate the folder into a Collection server-side; write the receipt."""

    # Filter validation happens before any Encord call: an unpinned shape must
    # never reach the server (it blocks the evaluation request indefinitely).
    parsed = parse_filter_specs(filters)
    preset_json = build_filter_preset_json(parsed)
    if not collection.strip():
        raise EncordToolError("--collection must not be empty.")

    from npa.clients.storage import StorageClient

    active_storage = storage_client or StorageClient.from_environment()
    client = user_client if user_client is not None else _default_user_client(environ)

    # Write-ahead receipt: land the plan before the first Encord mutation
    # (collection/preset creation), so an uncatchable kill mid-mutation still
    # leaves a durable record of intent.
    receipt_uri = curate_receipt_uri_for(output_path)
    preset_name = f"npa-curate-{workflow_run.strip() or 'adhoc'}"
    planned = CurateReceipt(
        generated_at=datetime.now(timezone.utc).isoformat(),
        workflow_run=workflow_run,
        encord_domain=resolve_domain(environ),
        folder_name=folder.strip(),
        collection_name=collection.strip(),
        preset_name=preset_name,
        filters=parsed,
        filter_preset_json=preset_json,
        status="planned",
        receipt_uri=receipt_uri,
    )
    finalize_artifact(
        planned,
        result_uri=receipt_uri,
        filename=CURATE_RECEIPT_FILENAME,
        storage_client=active_storage,
        run_error=None,
        failure_prefix="Encord curate failed",
    )

    # Everything below can mutate Encord (collection/preset creation); the
    # receipt must land even when a later step throws.
    folder_obj: Any = None
    collection_obj: Any = None
    collection_uuid = collection_name = ""
    collection_created = False
    preset_uuid = ""
    items_selected = 0
    status = "failed"
    run_error: Exception | None = None
    try:
        # Curate never creates the folder: an absent folder means there is
        # nothing to curate, not a fresh namespace to make.
        folder_obj, _ = resolve_folder(client, folder, create=False)
        # An empty folder is a definitive fail-fast: no filter can select
        # anything, and burning the poll window would only blur the diagnosis.
        if next(iter(folder_obj.list_items(page_size=1)), None) is None:
            raise EncordToolError(
                f"Encord folder {folder_obj.name!r} contains no storage items; "
                "nothing to curate."
            )
        collection_obj, collection_uuid, collection_name, collection_created = (
            resolve_collection(
                client, collection, create_in_folder_uuid=str(folder_obj.uuid)
            )
        )
        preset = client.create_preset(
            name=preset_name,
            description="Created by npa workbench encord curate",
            filter_preset_json=preset_json,
        )
        preset_uuid = str(preset.uuid)
        collection_obj.add_preset_items(preset_uuid)
        items_selected, status = _poll_selection(
            collection_obj, preset_uuid=preset_uuid, poll_seconds=poll_seconds
        )
        # The preset is transient scaffolding: the receipt's filter_preset_json
        # is the reproducibility record, so don't accumulate one per run.
        try:
            client.delete_preset(preset_uuid)
        except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
            logging.getLogger(__name__).debug(
                "preset %s cleanup failed: %s", preset_uuid, exc
            )
    except Exception as exc:  # noqa: BLE001 - recorded in the receipt, re-raised below
        run_error = exc

    receipt = CurateReceipt(
        generated_at=datetime.now(timezone.utc).isoformat(),
        workflow_run=workflow_run,
        encord_domain=resolve_domain(environ),
        folder_uuid=str(getattr(folder_obj, "uuid", "")),
        folder_name=str(getattr(folder_obj, "name", folder.strip())),
        collection_uuid=collection_uuid,
        collection_name=collection_name or collection.strip(),
        collection_created=collection_created,
        preset_uuid=preset_uuid,
        preset_name=preset_name,
        filters=parsed,
        filter_preset_json=preset_json,
        items_selected=items_selected,
        status=status,
        error=error_text(run_error),
        receipt_uri=receipt_uri,
    )
    finalize_artifact(
        receipt,
        result_uri=receipt_uri,
        filename=CURATE_RECEIPT_FILENAME,
        storage_client=active_storage,
        run_error=run_error,
        failure_prefix="Encord curate failed",
    )
    if status == "empty":
        raise EncordToolError(
            f"Encord curate selected 0 items. {_zero_selection_remedy(parsed)} "
            f"Receipt written to {receipt_uri}."
        )
    if status != "done":
        raise EncordToolError(
            f"Encord curate {status}: selection still changing after "
            f"{poll_seconds:.0f}s ({items_selected} item(s) so far). Receipt "
            f"written to {receipt_uri}."
        )
    return receipt
