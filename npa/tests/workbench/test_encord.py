"""Unit tests for the Encord workbench tool (SaaS and S3 mocked at the seam)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from npa.workbench.encord.client import (
    _default_user_client,
    resolve_dataset,
    resolve_folder,
    resolve_integration,
    resolve_public_endpoint,
)
from npa.workbench.encord.pull import (
    _same_endpoint_source,
    enumerate_items,
    pull_manifest_uri_for,
    run_pull,
    transfer_item,
)
from npa.workbench.encord.push import (
    BATCH_SIZE,
    build_upload_json,
    discover_objects,
    object_url_for,
    push_receipt_uri_for,
    run_push,
)
from npa.workbench.encord.schemas import (
    EncordAuthError,
    EncordToolError,
    PushedItem,
)

ENDPOINT = "https://storage.test.example"
ENVIRON = {"AWS_ENDPOINT_URL": ENDPOINT}


def _uuid(seed: int) -> str:
    return str(uuid.UUID(int=seed))


# --- fakes -------------------------------------------------------------------


class FakePaginator:
    def __init__(self, keys: list[str]) -> None:
        self._keys = keys

    def paginate(self, *, Bucket: str, Prefix: str) -> list[dict[str, Any]]:
        return [{"Contents": [{"Key": key} for key in self._keys]}]


class FakeS3:
    def __init__(self, keys: list[str] | None = None) -> None:
        self.keys = keys or []
        self.copy_calls: list[dict[str, Any]] = []

    def get_paginator(self, name: str) -> FakePaginator:
        assert name == "list_objects_v2"
        return FakePaginator(self.keys)

    def copy_object(self, *, Bucket: str, Key: str, CopySource: dict[str, str]) -> None:
        self.copy_calls.append({"Bucket": Bucket, "Key": Key, "CopySource": CopySource})


class FakeStorage:
    """StorageClient stand-in: raw .s3 plus upload_file recording."""

    def __init__(self, keys: list[str] | None = None) -> None:
        self.s3 = FakeS3(keys)
        self.uploads: list[tuple[str, str]] = []

    def upload_file(self, local_file: str, bucket_uri: str) -> str:
        self.uploads.append((local_file, bucket_uri))
        return bucket_uri


class FakePollResult:
    def __init__(
        self,
        status: str = "DONE",
        done: int = 0,
        errors: int = 0,
        items: list[tuple[str, str]] | None = None,
        unit_errors: list[tuple[list[str], str]] | None = None,
    ) -> None:
        self.status = SimpleNamespace(name=status)
        self.units_done_count = done
        self.units_error_count = errors
        self.units_pending_count = 0
        self.items_with_names = [
            SimpleNamespace(item_uuid=item_uuid, name=name)
            for item_uuid, name in (items or [])
        ]
        self.unit_errors = [
            SimpleNamespace(object_urls=urls, error=error)
            for urls, error in (unit_errors or [])
        ]


class FakeFolder:
    def __init__(self, name: str = "folder-a", results: list[FakePollResult] | None = None) -> None:
        self.uuid = uuid.UUID(int=1)
        self.name = name
        self.results = results or [FakePollResult()]
        self.start_calls: list[dict[str, Any]] = []
        self._result_index = 0

    def add_private_data_to_folder_start(self, *, integration_id, private_files, ignore_errors):
        self.start_calls.append(
            {
                "integration_id": integration_id,
                "private_files": private_files,
                "ignore_errors": ignore_errors,
            }
        )
        return uuid.UUID(int=100 + len(self.start_calls))

    def add_private_data_to_folder_get_result(self, job_id, timeout_seconds):
        result = self.results[min(self._result_index, len(self.results) - 1)]
        self._result_index += 1
        return result


class FakeDataset:
    def __init__(self, dataset_hash: str = _uuid(7), title: str = "ds-a") -> None:
        self.dataset_hash = dataset_hash
        self.title = title
        self.linked: list[list[str]] = []
        self.data_rows: list[Any] = []

    def link_items(self, item_uuids):
        self.linked.append([str(u) for u in item_uuids])
        return item_uuids


class FakeItem:
    def __init__(
        self,
        item_uuid: str,
        name: str,
        signed_url: str | None,
        file_size: int = 10,
    ) -> None:
        self.uuid = item_uuid
        self.name = name
        self.item_type = "VIDEO"
        self.mime_type = "video/mp4"
        self.file_size = file_size
        self.client_metadata = {}
        self._signed_url = signed_url
        self.refetch_calls = 0

    def get_signed_url(self, refetch: bool = False) -> str | None:
        if refetch:
            self.refetch_calls += 1
        return self._signed_url


class FakeCollection:
    def __init__(self, items: list[FakeItem]) -> None:
        self.uuid = uuid.UUID(int=9)
        self.name = "keepers"
        self._items = items

    def list_items(self, page_size: int | None = None):
        return iter(self._items)


class FakeUserClient:
    def __init__(
        self,
        *,
        integrations=None,
        folders=None,
        datasets=None,
        dataset_rows=None,
        collection=None,
        items=None,
    ) -> None:
        self.integrations = integrations or [
            SimpleNamespace(id=_uuid(3), title="nebius-s3")
        ]
        self.folders = folders or []
        self.datasets = datasets or {}
        self.dataset_rows = dataset_rows or []
        self.collection = collection
        self.items = items or []
        self.created_folders: list[str] = []
        self.created_datasets: list[str] = []

    def get_cloud_integrations(self):
        return list(self.integrations)

    def list_storage_folders(self, *, search: str = "", page_size: int = 100):
        return iter([f for f in self.folders if search in str(f.name)])

    def get_storage_folder(self, folder_uuid):
        for folder in self.folders:
            if str(folder.uuid) == str(folder_uuid):
                return folder
        raise KeyError(folder_uuid)

    def create_storage_folder(self, name, description=""):
        folder = FakeFolder(name=name)
        self.created_folders.append(name)
        self.folders.append(folder)
        return folder

    def get_dataset(self, dataset_hash):
        return self.datasets[str(dataset_hash)]

    def get_datasets(self, *, title_eq: str = ""):
        return [
            {"dataset": SimpleNamespace(dataset_hash=h, title=d.title)}
            for h, d in self.datasets.items()
            if d.title == title_eq
        ]

    def create_dataset(self, title, storage_location, dataset_description="", create_backing_folder=True):
        dataset_hash = _uuid(50 + len(self.created_datasets))
        self.created_datasets.append(title)
        self.datasets[dataset_hash] = FakeDataset(dataset_hash, title)
        return {"dataset_hash": dataset_hash}

    def get_collection(self, collection_uuid):
        return self.collection

    def list_collections(self, **kwargs):
        return iter([self.collection] if self.collection else [])

    def get_storage_items(self, item_uuids, sign_url=False):
        wanted = {str(u) for u in item_uuids}
        return [item for item in self.items if str(item.uuid) in wanted]


# --- url + discovery ---------------------------------------------------------


def test_object_url_for_is_path_style_and_encoded() -> None:
    url = object_url_for(ENDPOINT + "/", "bkt", "runs/a b/frame#1.png")
    assert url == f"{ENDPOINT}/bkt/runs/a%20b/frame%231.png"
    assert " " not in url


def test_receipt_and_manifest_uri_helpers() -> None:
    assert push_receipt_uri_for("s3://b/p/") == "s3://b/p/push_receipt.json"
    assert push_receipt_uri_for("s3://b/p/r.json") == "s3://b/p/r.json"
    assert pull_manifest_uri_for("s3://b/p/") == "s3://b/p/manifest.json"


def test_discover_objects_maps_suffixes_and_skips() -> None:
    storage = FakeStorage(["p/a.mp4", "p/b.PNG", "p/c.jpeg", "p/d.txt", "p/e.mcap"])
    entries, skipped = discover_objects(storage, "s3://bkt/p/", "videos-images")
    assert entries == [("p/a.mp4", "videos"), ("p/b.PNG", "images"), ("p/c.jpeg", "images")]
    assert skipped == ["p/d.txt", "p/e.mcap"]


def test_discover_objects_mcap_gating() -> None:
    storage = FakeStorage(["p/a.mp4", "p/e.mcap"])
    entries, _ = discover_objects(storage, "s3://bkt/p/", "mcap")
    assert entries == [("p/e.mcap", "mcap")]
    entries, skipped = discover_objects(storage, "s3://bkt/p/", "all")
    assert ("p/e.mcap", "mcap") in entries and ("p/a.mp4", "videos") in entries
    assert skipped == []


def test_discover_objects_empty_prefix_fails() -> None:
    with pytest.raises(EncordToolError, match="No supported media"):
        discover_objects(FakeStorage(["p/readme.md"]), "s3://bkt/p/", "videos-images")


def test_build_upload_json_shape() -> None:
    items = [
        PushedItem(key="a.mp4", object_url="u1", category="videos"),
        PushedItem(key="b.png", object_url="u2", category="images"),
    ]
    payload = build_upload_json(items)
    assert payload["skip_duplicate_urls"] is True
    assert payload["videos"] == [{"objectUrl": "u1"}]
    assert payload["images"] == [{"objectUrl": "u2"}]


# --- client resolution -------------------------------------------------------


def test_resolve_integration_by_title_and_id() -> None:
    client = FakeUserClient()
    integration_id, title = resolve_integration(client, "nebius-s3")
    assert (integration_id, title) == (_uuid(3), "nebius-s3")
    assert resolve_integration(client, _uuid(3)) == (_uuid(3), "nebius-s3")
    with pytest.raises(EncordToolError, match="No Encord cloud integration titled"):
        resolve_integration(client, "missing")


def test_resolve_folder_creates_on_missing_title_only() -> None:
    client = FakeUserClient()
    folder, created = resolve_folder(client, "fresh")
    assert created is True and client.created_folders == ["fresh"]
    again, created = resolve_folder(client, "fresh")
    assert created is False and again is folder
    with pytest.raises(KeyError):
        resolve_folder(client, _uuid(99))


def test_resolve_dataset_title_create_and_pull_no_create() -> None:
    client = FakeUserClient()
    _, dataset_hash, title, created = resolve_dataset(client, "new-ds")
    assert created is True and title == "new-ds"
    _, _, _, created = resolve_dataset(client, "new-ds")
    assert created is False
    with pytest.raises(EncordToolError, match="No Encord dataset titled"):
        resolve_dataset(client, "absent", create=False)


def test_default_user_client_requires_secret_and_decodes_b64() -> None:
    with pytest.raises(EncordAuthError, match="No Encord credential"):
        _default_user_client({})
    with pytest.raises(EncordAuthError, match="not valid base64"):
        _default_user_client({"ENCORD_SSH_KEY_B64": "!!!not-base64!!!"})


def test_resolve_public_endpoint_prefers_env() -> None:
    assert resolve_public_endpoint({"AWS_ENDPOINT_URL": ENDPOINT + "/"}) == ENDPOINT
    with pytest.raises(EncordToolError, match="No S3 endpoint"):
        resolve_public_endpoint({})


# --- run_push ----------------------------------------------------------------


def _push_kwargs(tmp_path: Path, storage: FakeStorage, client: FakeUserClient, **overrides):
    kwargs = dict(
        input_path="s3://bkt/p/",
        integration="nebius-s3",
        folder="fresh",
        output_path=str(tmp_path / "receipt.json"),
        user_client=client,
        storage_client=storage,
        environ=dict(ENVIRON),
    )
    kwargs.update(overrides)
    return kwargs


def test_run_push_happy_path_links_dataset(tmp_path: Path) -> None:
    storage = FakeStorage(["p/a.mp4", "p/b.png"])
    folder = FakeFolder(
        results=[
            FakePollResult(
                status="DONE",
                done=2,
                items=[(_uuid(21), "p/a.mp4"), (_uuid(22), "b.png")],
            )
        ]
    )
    client = FakeUserClient(folders=[])
    client.create_storage_folder = lambda name, description="": folder  # type: ignore[assignment]
    receipt = run_push(**_push_kwargs(tmp_path, storage, client, dataset="new-ds"))
    assert receipt.status == "done"
    assert receipt.units_done == 2 and receipt.units_error == 0
    assert receipt.dataset_created is True and receipt.linked_count == 2
    dataset = next(iter(client.datasets.values()))
    assert dataset.linked == [[_uuid(21), _uuid(22)]]
    # uuids attached to receipt rows by basename
    assert {item.item_uuid for item in receipt.items} == {_uuid(21), _uuid(22)}
    # objectUrls are path-style against the environ endpoint
    assert receipt.items[0].object_url == f"{ENDPOINT}/bkt/p/a.mp4"
    # receipt written locally
    payload = json.loads((tmp_path / "receipt.json").read_text())
    assert payload["schema"] == "npa.encord.push_receipt.v1"
    assert payload["status"] == "done"


def test_run_push_unit_errors_write_receipt_then_raise(tmp_path: Path) -> None:
    storage = FakeStorage(["p/a.mp4"])
    bad_url = f"{ENDPOINT}/bkt/p/a.mp4"
    folder = FakeFolder(
        results=[FakePollResult(status="DONE", done=0, errors=1, unit_errors=[([bad_url], "403 from integration")])]
    )
    client = FakeUserClient(folders=[folder])
    with pytest.raises(EncordToolError, match="1 unit error"):
        run_push(**_push_kwargs(tmp_path, storage, client, folder=str(folder.uuid)))
    payload = json.loads((tmp_path / "receipt.json").read_text())
    assert payload["status"] == "failed"
    assert payload["items"][0]["status"] == "error"
    assert "403" in payload["items"][0]["error"]


def test_run_push_timeout_is_fail_closed(tmp_path: Path) -> None:
    storage = FakeStorage(["p/a.mp4"])
    folder = FakeFolder(results=[FakePollResult(status="PENDING")])
    client = FakeUserClient(folders=[folder])
    with pytest.raises(EncordToolError, match="timeout"):
        run_push(**_push_kwargs(tmp_path, storage, client, folder=str(folder.uuid)))
    payload = json.loads((tmp_path / "receipt.json").read_text())
    assert payload["status"] == "timeout"


def test_run_push_mcap_is_experimental_error(tmp_path: Path) -> None:
    storage = FakeStorage(["p/e.mcap"])
    folder = FakeFolder()
    client = FakeUserClient(folders=[folder])
    with pytest.raises(EncordToolError):
        run_push(**_push_kwargs(tmp_path, storage, client, folder=str(folder.uuid), media="mcap"))
    payload = json.loads((tmp_path / "receipt.json").read_text())
    assert payload["items"][0]["status"] == "experimental_error"
    assert folder.start_calls == []  # nothing guessed onto the wire


def test_run_push_batches_at_500(tmp_path: Path) -> None:
    keys = [f"p/{index:04d}.png" for index in range(BATCH_SIZE + 1)]
    storage = FakeStorage(keys)
    folder = FakeFolder(results=[FakePollResult(status="DONE", done=BATCH_SIZE + 1)])
    client = FakeUserClient(folders=[folder])
    run_push(**_push_kwargs(tmp_path, storage, client, folder=str(folder.uuid)))
    assert len(folder.start_calls) == 2
    first = folder.start_calls[0]["private_files"]
    assert len(first["images"]) == BATCH_SIZE


def test_run_push_rejects_local_input(tmp_path: Path) -> None:
    with pytest.raises(EncordToolError, match="s3:// prefix"):
        run_push(
            input_path=str(tmp_path),
            integration="i",
            folder="f",
            output_path=str(tmp_path / "r.json"),
            user_client=FakeUserClient(),
            storage_client=FakeStorage(),
            environ=dict(ENVIRON),
        )


# --- pull --------------------------------------------------------------------


def test_same_endpoint_source_path_and_virtual_hosted() -> None:
    assert _same_endpoint_source(
        f"{ENDPOINT}/bkt/p/a.mp4?X-Sig=abc", ENDPOINT
    ) == ("bkt", "p/a.mp4")
    host = ENDPOINT.removeprefix("https://")
    assert _same_endpoint_source(
        f"https://bkt.{host}/p/a%20b.mp4?X-Sig=abc", ENDPOINT
    ) == ("bkt", "p/a b.mp4")
    assert _same_endpoint_source("https://elsewhere.example/x", ENDPOINT) is None


def test_transfer_item_same_bucket_copies_server_side() -> None:
    storage = FakeStorage()
    item = FakeItem(_uuid(31), "a.mp4", f"{ENDPOINT}/bkt/p/a.mp4?sig=1", file_size=5)
    record = transfer_item(
        item, storage_client=storage, output_uri="s3://out/pull", endpoint_url=ENDPOINT
    )
    assert record.transfer == "copy"
    assert storage.s3.copy_calls[0]["CopySource"] == {"Bucket": "bkt", "Key": "p/a.mp4"}
    assert storage.s3.copy_calls[0]["Bucket"] == "out"
    assert record.media_uri.startswith("s3://out/pull/media/")


def test_transfer_item_composite_without_signed_url_is_error() -> None:
    record = transfer_item(
        FakeItem(_uuid(32), "group", None),
        storage_client=FakeStorage(),
        output_uri="s3://out/pull",
        endpoint_url=ENDPOINT,
    )
    assert record.transfer == "error" and "signed URL" in record.error


def test_transfer_item_downloads_cross_origin(monkeypatch: pytest.MonkeyPatch) -> None:

    class FakeResponse:
        def raise_for_status(self) -> None: ...
        def iter_bytes(self, chunk_size: int):
            yield b"payload"

    class FakeStream:
        def __init__(self, *args, **kwargs) -> None: ...
        def __enter__(self) -> FakeResponse:
            return FakeResponse()
        def __exit__(self, *args) -> None: ...

    import httpx

    monkeypatch.setattr(httpx, "stream", FakeStream)
    storage = FakeStorage()
    record = transfer_item(
        FakeItem(_uuid(33), "far.mp4", "https://cdn.encord.example/x?sig=1"),
        storage_client=storage,
        output_uri="s3://out/pull",
        endpoint_url=ENDPOINT,
    )
    assert record.transfer == "download"
    assert storage.uploads and storage.uploads[0][1] == record.media_uri


def test_enumerate_items_per_source() -> None:
    items = [FakeItem(_uuid(41), "a.mp4", "u")]
    collection = FakeCollection(items)
    dataset = FakeDataset()
    dataset.data_rows = [SimpleNamespace(backing_item_uuid=_uuid(41))]
    client = FakeUserClient(
        collection=collection, datasets={dataset.dataset_hash: dataset}, items=items
    )
    source_id, name, found, project = enumerate_items(
        client, source="collection", source_id=str(collection.uuid)
    )
    assert name == "keepers" and len(found) == 1 and project is None
    source_id, name, found, project = enumerate_items(
        client, source="dataset", source_id=dataset.dataset_hash
    )
    assert source_id == dataset.dataset_hash and found == items


def test_run_pull_writes_manifest_and_fails_closed_on_errors(tmp_path: Path) -> None:
    good = FakeItem(_uuid(51), "a.mp4", f"{ENDPOINT}/bkt/p/a.mp4", file_size=7)
    bad = FakeItem(_uuid(52), "b.mp4", None)
    collection = FakeCollection([good, bad])
    client = FakeUserClient(collection=collection)
    storage = FakeStorage()
    out_uri = "s3://out/pull"
    with pytest.raises(EncordToolError, match="failed for 1 of 2"):
        run_pull(
            source="collection",
            source_id=str(collection.uuid),
            output_path=out_uri,
            user_client=client,
            storage_client=storage,
            environ=dict(ENVIRON),
        )
    # manifest + per-item JSON were still uploaded before the raise
    uploaded = [uri for _, uri in storage.uploads]
    assert f"{out_uri}/manifest.json" in uploaded
    assert f"{out_uri}/items/{_uuid(51)}.json" in uploaded


def test_run_pull_happy_path_counts(tmp_path: Path) -> None:
    good = FakeItem(_uuid(53), "a.mp4", f"{ENDPOINT}/bkt/p/a.mp4", file_size=7)
    collection = FakeCollection([good])
    client = FakeUserClient(collection=collection)
    storage = FakeStorage()
    manifest = run_pull(
        source="collection",
        source_id=str(collection.uuid),
        output_path="s3://out/pull",
        user_client=client,
        storage_client=storage,
        environ=dict(ENVIRON),
    )
    assert manifest.items_total == 1
    assert manifest.media_copied == 1 and manifest.media_failed == 0
    assert manifest.media_bytes == 7
    assert manifest.manifest_uri == "s3://out/pull/manifest.json"


class FakeUploadFolder(FakeFolder):
    def __init__(self) -> None:
        super().__init__()
        self.uploads: list[tuple[str, str, str]] = []  # (kind, path, title)

    def upload_image(self, file_path, title=None, **kwargs):
        self.uploads.append(("image", str(file_path), str(title)))
        return uuid.UUID(int=200 + len(self.uploads))

    def upload_video(self, file_path, title=None, **kwargs):
        self.uploads.append(("video", str(file_path), str(title)))
        return uuid.UUID(int=200 + len(self.uploads))


class FakeDownloadStorage(FakeStorage):
    def __init__(self, keys=None) -> None:
        super().__init__(keys)
        self.downloads: list[str] = []

    def download_file(self, bucket_uri: str, local_path: str) -> str:
        self.downloads.append(bucket_uri)
        Path(local_path).write_bytes(b"media-bytes")
        return local_path


def test_run_push_upload_mode_copies_bytes_and_links(tmp_path: Path) -> None:
    storage = FakeDownloadStorage(["p/a.mp4", "p/b.png"])
    folder = FakeUploadFolder()
    client = FakeUserClient(folders=[folder])
    receipt = run_push(
        input_path="s3://bkt/p/",
        integration="",  # unused in upload mode
        folder=str(folder.uuid),
        dataset="new-ds",
        transfer="upload",
        output_path=str(tmp_path / "receipt.json"),
        user_client=client,
        storage_client=storage,
        environ={},  # upload mode needs no public endpoint
    )
    assert receipt.status == "done" and receipt.transfer == "upload"
    assert receipt.units_done == 2 and receipt.units_error == 0
    # bytes moved: each object downloaded from S3 then uploaded by kind, titled by key
    assert storage.downloads == ["s3://bkt/p/a.mp4", "s3://bkt/p/b.png"]
    assert [(kind, title) for kind, _, title in folder.uploads] == [
        ("video", "p/a.mp4"),
        ("image", "p/b.png"),
    ]
    # no registration jobs, no objectUrls
    assert folder.start_calls == []
    assert all(item.object_url == "" for item in receipt.items)
    assert all(item.status == "uploaded" and item.item_uuid for item in receipt.items)
    dataset = next(iter(client.datasets.values()))
    assert len(dataset.linked[0]) == 2


def test_run_push_upload_mode_per_item_error_fails_closed(tmp_path: Path) -> None:
    storage = FakeDownloadStorage(["p/a.mp4"])

    class BrokenFolder(FakeUploadFolder):
        def upload_video(self, file_path, title=None, **kwargs):
            raise RuntimeError("507 storage quota")

    folder = BrokenFolder()
    client = FakeUserClient(folders=[folder])
    with pytest.raises(EncordToolError, match="1 unit error"):
        run_push(
            input_path="s3://bkt/p/",
            integration="",
            folder=str(folder.uuid),
            transfer="upload",
            output_path=str(tmp_path / "receipt.json"),
            user_client=client,
            storage_client=storage,
            environ={},
        )
    payload = json.loads((tmp_path / "receipt.json").read_text())
    assert payload["status"] == "failed"
    assert "507" in payload["items"][0]["error"]


def test_run_push_rejects_unknown_transfer(tmp_path: Path) -> None:
    with pytest.raises(EncordToolError, match="Unknown --transfer"):
        run_push(
            input_path="s3://bkt/p/",
            integration="i",
            folder="f",
            transfer="teleport",
            output_path=str(tmp_path / "r.json"),
            user_client=FakeUserClient(),
            storage_client=FakeStorage(["p/a.mp4"]),
            environ=dict(ENVIRON),
        )
