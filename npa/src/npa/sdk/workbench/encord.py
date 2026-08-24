"""SDK surface for the Encord workbench tool."""

from __future__ import annotations

from typing import Any

from npa.workbench.encord.schemas import (
    DEFAULT_MEDIA_FILTER,
    DEFAULT_POLL_TIMEOUT_SECONDS,
    DEFAULT_TRANSFER,
    PullManifest,
    PushReceipt,
)


def push(
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
) -> PushReceipt:
    """Register S3 media in place into an Encord folder; write a receipt."""

    from npa.workbench.encord.push import run_push

    return run_push(
        input_path=input_path,
        integration=integration,
        folder=folder,
        output_path=output_path,
        dataset=dataset,
        media=media,
        transfer=transfer,
        poll_timeout_seconds=poll_timeout_seconds,
        workflow_run=workflow_run,
        user_client=user_client,
        storage_client=storage_client,
    )


def pull(
    *,
    source: str,
    source_id: str,
    output_path: str,
    workflow_run: str = "",
    user_client: Any = None,
    storage_client: Any = None,
) -> PullManifest:
    """Materialize a curated Encord source to S3; write a lineage manifest."""

    from npa.workbench.encord.pull import run_pull

    return run_pull(
        source=source,
        source_id=source_id,
        output_path=output_path,
        workflow_run=workflow_run,
        user_client=user_client,
        storage_client=storage_client,
    )


__all__ = ["PullManifest", "PushReceipt", "pull", "push"]
