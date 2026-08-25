"""Schemas and constants for the Encord workbench tool.

The Encord tool registers Nebius object-store media in place (bytes stay in the
bucket; Encord references them through an S3-compatible cloud integration) and
pulls curated data plus labels back to S3. These models are the durable receipt
and manifest contracts other stages consume.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

PUSH_RECEIPT_SCHEMA = "npa.encord.push_receipt.v1"
PULL_MANIFEST_SCHEMA = "npa.encord.pull_manifest.v1"
PUSH_RECEIPT_FILENAME = "push_receipt.json"
PULL_MANIFEST_FILENAME = "manifest.json"
DEFAULT_MEDIA_FILTER = "videos-images"
DEFAULT_TRANSFER = "register"
DEFAULT_POLL_TIMEOUT_SECONDS = 1800


class EncordToolError(RuntimeError):
    """Raised when an Encord push/pull operation fails."""


class EncordAuthError(EncordToolError):
    """Raised when no usable Encord credential can be resolved."""


class PushedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    object_url: str = ""
    category: str
    item_uuid: str = ""
    # registered | uploaded | error | experimental_error
    status: str = "registered"
    error: str = ""


class PushReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: str = Field(default=PUSH_RECEIPT_SCHEMA, alias="schema")
    tool: str = "encord"
    stage: str = "push"
    generated_at: str
    workflow_run: str = ""
    input_uri: str
    endpoint_url: str
    encord_domain: str
    # register (in-place by objectUrl) | upload (bytes copied into Encord storage)
    transfer: str = "register"
    integration_id: str = ""
    integration_title: str = ""
    folder_uuid: str
    folder_name: str
    folder_created: bool = False
    dataset_hash: str = ""
    dataset_title: str = ""
    dataset_created: bool = False
    linked_count: int = 0
    media_filter: str
    # done | failed | timeout
    status: str
    files_discovered: int = 0
    units_done: int = 0
    units_error: int = 0
    # Populated when the run failed partway: the exception that ended it, so the
    # receipt still explains a crash that happened after Encord was mutated.
    error: str = ""
    receipt_uri: str = ""
    items: list[PushedItem] = Field(default_factory=list)
    skipped_unsupported: list[str] = Field(default_factory=list)


class PulledItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_uuid: str
    name: str
    item_type: str = ""
    mime_type: str = ""
    file_size: int = 0
    media_uri: str = ""
    # copy | download | error
    transfer: str = ""
    error: str = ""


class PullManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: str = Field(default=PULL_MANIFEST_SCHEMA, alias="schema")
    tool: str = "encord"
    stage: str = "pull"
    generated_at: str
    workflow_run: str = ""
    encord_domain: str
    # collection | dataset | project
    source_kind: str
    source_id: str
    source_name: str = ""
    output_uri: str
    manifest_uri: str = ""
    items_total: int = 0
    media_copied: int = 0
    media_downloaded: int = 0
    media_failed: int = 0
    label_rows: int = 0
    media_bytes: int = 0
    # Populated when the run failed partway (see PushReceipt.error).
    error: str = ""
    label_uris: list[str] = Field(default_factory=list)
    items: list[PulledItem] = Field(default_factory=list)
