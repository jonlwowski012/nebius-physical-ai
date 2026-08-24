"""Shared S3 JSON writer for the Encord tool's receipts, manifests, and labels."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any


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
