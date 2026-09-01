"""Shared S3 JSON writer for the Encord tool's receipts, manifests, and labels."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from npa.workbench.encord.schemas import EncordToolError


def artifact_uri_for(output_path: str, filename: str) -> str:
    """The exact artifact URI a receipt-style --output-path resolves to.

    A ``.json`` output path is the artifact itself; anything else is a prefix
    the artifact lands under. (Pull's manifest helper is deliberately not this:
    its output path is always a directory prefix holding media/ and items/.)
    """

    if output_path.endswith(".json"):
        return output_path
    return output_path.rstrip("/") + f"/{filename}"


def error_text(run_error: Exception | None) -> str:
    """The durable artifact's ``error`` field for a mid-run exception."""

    return f"{type(run_error).__name__}: {run_error}" if run_error else ""


def finalize_artifact(
    model: Any,
    *,
    result_uri: str,
    filename: str,
    storage_client: Any,
    run_error: Exception | None,
    failure_prefix: str,
    artifact_noun: str = "Receipt",
) -> None:
    """Write the durable artifact, then re-raise a mid-run failure.

    This is the cross-verb contract SKILL.md advertises: the artifact lands
    before any failure exit, and a crash after Encord was mutated is recorded
    in the artifact's ``error`` field. Verb-specific post-write checks (unit
    errors, zero selections, ...) stay at the call sites.
    """

    write_json(
        model.model_dump(by_alias=True),
        result_uri=result_uri,
        filename=filename,
        storage_client=storage_client,
    )
    if run_error is not None:
        raise EncordToolError(
            f"{failure_prefix}: {model.error}. {artifact_noun} written to {result_uri}."
        ) from run_error


def read_json(uri: str, *, storage_client: Any) -> dict[str, Any]:
    """Read a JSON document from an s3:// URI or a local path."""

    if uri.startswith("s3://"):
        from npa.clients.storage import _parse_bucket_uri

        bucket, key = _parse_bucket_uri(uri)
        body = storage_client.s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(body)
    return json.loads(Path(uri).read_text(encoding="utf-8"))


def write_json(
    payload: dict[str, Any],
    *,
    result_uri: str,
    filename: str,
    storage_client: Any,
) -> str:
    """Write ``payload`` to an s3:// URI or a local path; return the written URI."""

    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if result_uri.startswith("s3://"):
        with tempfile.TemporaryDirectory(prefix="npa-encord-") as tmp:
            local_path = Path(tmp) / filename
            local_path.write_text(body, encoding="utf-8")
            return storage_client.upload_file(str(local_path), result_uri)

    path = Path(result_uri)
    if path.suffix != ".json":
        path = path / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return str(path)
