"""Unit tests for the encord-cosmos3-augment glue stage."""

from __future__ import annotations

import json
from typing import Any

import pytest

from npa.workflows.encord_loop import EncordLoopError, stage_media_for_augment

MANIFEST_URI = "s3://bkt/run/pull/manifest.json"
DEST_URI = "s3://bkt/run/augment-input/source.mp4"


class FakeS3:
    def __init__(self) -> None:
        self.copy_calls: list[dict[str, Any]] = []

    def copy_object(self, *, Bucket: str, Key: str, CopySource: dict[str, str]) -> None:
        self.copy_calls.append({"Bucket": Bucket, "Key": Key, "CopySource": CopySource})


class FakeStorage:
    def __init__(self, manifest: dict[str, Any] | None) -> None:
        self._manifest = manifest
        self.s3 = FakeS3()

    def read_bytes_with_etag(self, uri: str):
        if self._manifest is None:
            return None
        return json.dumps(self._manifest).encode(), "etag"


def _manifest(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"schema": "npa.encord.pull_manifest.v1", "items": items}


def test_stages_selected_item_server_side() -> None:
    storage = FakeStorage(
        _manifest(
            [
                {"item_uuid": "u1", "name": "a.mp4", "transfer": "copy",
                 "media_uri": "s3://bkt/run/pull/media/u1__a.mp4"},
                {"item_uuid": "u2", "name": "b.mp4", "transfer": "download",
                 "media_uri": "s3://bkt/run/pull/media/u2__b.mp4"},
            ]
        )
    )
    summary = stage_media_for_augment(MANIFEST_URI, DEST_URI, "1", storage_client=storage)
    assert summary["item_uuid"] == "u2" and summary["items_available"] == 2
    call = storage.s3.copy_calls[0]
    assert call["CopySource"] == {"Bucket": "bkt", "Key": "run/pull/media/u2__b.mp4"}
    assert (call["Bucket"], call["Key"]) == ("bkt", "run/augment-input/source.mp4")


def test_failed_items_are_not_selectable() -> None:
    storage = FakeStorage(
        _manifest(
            [
                {"item_uuid": "bad", "name": "x", "transfer": "error", "media_uri": ""},
                {"item_uuid": "u1", "name": "a.mp4", "transfer": "copy",
                 "media_uri": "s3://bkt/p/media/u1__a.mp4"},
            ]
        )
    )
    summary = stage_media_for_augment(MANIFEST_URI, DEST_URI, "0", storage_client=storage)
    assert summary["item_uuid"] == "u1"  # the error row is filtered out


def test_missing_manifest_fails_closed() -> None:
    with pytest.raises(EncordLoopError, match="not found"):
        stage_media_for_augment(MANIFEST_URI, DEST_URI, storage_client=FakeStorage(None))


def test_index_out_of_range_fails_closed() -> None:
    storage = FakeStorage(_manifest([]))
    with pytest.raises(EncordLoopError, match="0 transferred media"):
        stage_media_for_augment(MANIFEST_URI, DEST_URI, "0", storage_client=storage)


def test_seed_skips_when_operator_supplied_a_curated_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import npa.workflows.data_factory_input as dfi
    from npa.workflows.encord_loop import seed_demo_source

    def explode(*args, **kwargs):
        raise AssertionError("seed must not fetch when skipping")

    monkeypatch.setattr(dfi, "_fetch_starter", explode)
    summary = seed_demo_source(
        "s3://bkt/run/seed/", "npa-demo-src-run", "my-curated-collection",
        storage_client=FakeStorage(None),
    )
    assert summary["skipped"]


def test_seed_fetches_uploads_and_pushes_default_demo(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import npa.sdk.workbench.encord as encord_sdk
    import npa.workflows.data_factory_input as dfi
    from npa.workflows.encord_loop import seed_demo_source

    clip = tmp_path / "starter.mp4"
    clip.write_bytes(b"starter-bytes")
    monkeypatch.setattr(dfi, "_fetch_starter", lambda contract, **kw: (clip, "verified_hit"))
    monkeypatch.setattr(
        dfi, "load_starter_contract",
        lambda: {"integrity": {"sha256": "a" * 64}, "license": {"name": "CC-BY-4.0"}},
    )
    pushed: dict = {}

    class Receipt:
        units_done = 1

    def fake_push(**kwargs):
        pushed.update(kwargs)
        return Receipt()

    monkeypatch.setattr(encord_sdk, "push", fake_push)

    class UploadingStorage(FakeStorage):
        def __init__(self) -> None:
            super().__init__(None)
            self.uploads: list[tuple[str, str]] = []

        def upload_file(self, local_file: str, bucket_uri: str) -> str:
            self.uploads.append((local_file, bucket_uri))
            return bucket_uri

    storage = UploadingStorage()
    summary = seed_demo_source(
        "s3://bkt/run/seed/", "npa-demo-src-run", "npa-demo-src-run",
        transfer="upload", storage_client=storage,
    )
    assert storage.uploads[0][1] == "s3://bkt/run/seed/starter-clip.mp4"
    assert pushed["dataset"] == "npa-demo-src-run" and pushed["transfer"] == "upload"
    assert summary["units_done"] == 1 and summary["attribution"] == "CC-BY-4.0"
