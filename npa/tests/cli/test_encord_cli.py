"""CLI tests for `npa workbench encord` (SaaS and S3 never touched)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import npa.clients.credentials as credentials_module
from npa.cli.main import app
from npa.workbench.encord.schemas import (
    CurateReceipt,
    EncordToolError,
    PullManifest,
    PushReceipt,
)

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", tmp_path / "missing.yaml")
    for name in ("ENCORD_SSH_KEY", "ENCORD_SSH_KEY_B64", "ENCORD_SSH_KEY_FILE"):
        monkeypatch.delenv(name, raising=False)


def _receipt(**overrides) -> PushReceipt:
    payload = dict(
        generated_at="2026-08-24T00:00:00+00:00",
        input_uri="s3://bkt/p/",
        endpoint_url="https://storage.test",
        encord_domain="https://api.encord.com",
        integration_id="00000000-0000-0000-0000-000000000003",
        integration_title="nebius-s3",
        folder_uuid="00000000-0000-0000-0000-000000000001",
        folder_name="fresh",
        media_filter="videos-images",
        status="done",
        files_discovered=2,
        units_done=2,
        receipt_uri="s3://bkt/out/push_receipt.json",
    )
    payload.update(overrides)
    return PushReceipt(**payload)


def _manifest(**overrides) -> PullManifest:
    payload = dict(
        generated_at="2026-08-24T00:00:00+00:00",
        encord_domain="https://api.encord.com",
        source_kind="collection",
        source_id="00000000-0000-0000-0000-000000000009",
        source_name="keepers",
        output_uri="s3://bkt/pull",
        manifest_uri="s3://bkt/pull/manifest.json",
        items_total=1,
        media_copied=1,
    )
    payload.update(overrides)
    return PullManifest(**payload)


def _curate_receipt(**overrides) -> CurateReceipt:
    payload = dict(
        generated_at="2026-08-24T00:00:00+00:00",
        encord_domain="https://api.encord.com",
        folder_uuid="00000000-0000-0000-0000-000000000001",
        folder_name="src",
        collection_uuid="00000000-0000-0000-0000-0000000000c8",
        collection_name="keepers",
        collection_created=True,
        preset_uuid="00000000-0000-0000-0000-00000000012c",
        preset_name="npa-curate-run-1",
        items_selected=2,
        status="done",
        receipt_uri="s3://bkt/curate/curate_receipt.json",
    )
    payload.update(overrides)
    return CurateReceipt(**payload)


def test_encord_group_registered() -> None:
    result = runner.invoke(app, ["workbench", "encord", "--help"])
    assert result.exit_code == 0
    assert "push" in result.output and "pull" in result.output
    assert "curate" in result.output


def test_push_help_contains_contract_options() -> None:
    result = runner.invoke(app, ["workbench", "encord", "push", "--help"])
    assert result.exit_code == 0
    for option in ("input-path", "integration", "folder", "output-path", "media"):
        assert option in result.output


def test_pull_help_contains_source_options() -> None:
    result = runner.invoke(app, ["workbench", "encord", "pull", "--help"])
    assert result.exit_code == 0
    for option in ("source", "source-id", "output-path"):
        assert option in result.output


def test_push_happy_path_json(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_push(**kwargs):
        captured.update(kwargs)
        return _receipt()

    monkeypatch.setattr("npa.sdk.workbench.encord.push", fake_push)
    result = runner.invoke(
        app,
        [
            "workbench", "encord", "push",
            "--input-path", "s3://bkt/p/",
            "--integration", "nebius-s3",
            "--folder", "fresh",
            "--dataset", "ds-a",
            "--output-path", "s3://bkt/out/",
            "--workflow-run", "run-1",
            "--output", "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema"] == "npa.encord.push_receipt.v1"
    assert captured["media"] == "videos-images"
    assert captured["dataset"] == "ds-a"
    assert captured["workflow_run"] == "run-1"


def test_push_rejects_local_input_path() -> None:
    result = runner.invoke(
        app,
        [
            "workbench", "encord", "push",
            "--input-path", "/tmp/media",
            "--integration", "i",
            "--folder", "f",
            "--output-path", "s3://bkt/out/",
        ],
    )
    assert result.exit_code == 1
    assert "S3" in result.output


def test_push_tool_error_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_push(**kwargs):
        raise EncordToolError("Encord push failed: 1 unit error(s)")

    monkeypatch.setattr("npa.sdk.workbench.encord.push", fake_push)
    result = runner.invoke(
        app,
        [
            "workbench", "encord", "push",
            "--input-path", "s3://bkt/p/",
            "--integration", "i",
            "--folder", "f",
            "--output-path", "s3://bkt/out/",
        ],
    )
    assert result.exit_code == 1
    assert "unit error" in result.output


def test_curate_help_contains_contract_options() -> None:
    result = runner.invoke(app, ["workbench", "encord", "curate", "--help"])
    assert result.exit_code == 0
    for option in ("folder", "filter", "collection", "output-path", "poll-seconds"):
        assert option in result.output


def test_curate_happy_path_json(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_curate(**kwargs):
        captured.update(kwargs)
        return _curate_receipt()

    monkeypatch.setattr("npa.sdk.workbench.encord.curate", fake_curate)
    result = runner.invoke(
        app,
        [
            "workbench", "encord", "curate",
            "--folder", "src",
            "--filter", "brightness:0.2:0.8,sharpness:0.3:1",
            "--filter", "width:32:4096",
            "--collection", "keepers",
            "--output-path", "s3://bkt/curate/",
            "--workflow-run", "run-1",
            "--output", "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema"] == "npa.encord.curate_receipt.v1"
    assert captured["folder"] == "src"
    assert captured["filters"] == ["brightness:0.2:0.8,sharpness:0.3:1", "width:32:4096"]
    assert captured["collection"] == "keepers"
    assert captured["workflow_run"] == "run-1"


def test_curate_text_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "npa.sdk.workbench.encord.curate", lambda **kwargs: _curate_receipt()
    )
    result = runner.invoke(
        app,
        [
            "workbench", "encord", "curate",
            "--folder", "src",
            "--filter", "width:1:100000",
            "--collection", "keepers",
            "--output-path", "s3://bkt/curate/",
            "--output", "text",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "curated 2 item(s)" in result.output


def test_curate_rejects_local_output_path() -> None:
    result = runner.invoke(
        app,
        [
            "workbench", "encord", "curate",
            "--folder", "src",
            "--filter", "width:1:100000",
            "--collection", "keepers",
            "--output-path", "/tmp/out",
        ],
    )
    assert result.exit_code == 1


def test_curate_tool_error_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_curate(**kwargs):
        raise EncordToolError("Encord curate selected 0 items.")

    monkeypatch.setattr("npa.sdk.workbench.encord.curate", fake_curate)
    result = runner.invoke(
        app,
        [
            "workbench", "encord", "curate",
            "--folder", "src",
            "--filter", "brightness:0.2:0.8",
            "--collection", "keepers",
            "--output-path", "s3://bkt/curate/",
        ],
    )
    assert result.exit_code == 1
    assert "selected 0 items" in result.output


def test_verify_happy_and_failure_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    from npa.workbench.encord.schemas import RoundtripReport

    def fake_verify(**kwargs):
        return RoundtripReport(
            generated_at="t",
            receipt_uri=kwargs["receipt_uri"],
            manifest_uri=kwargs["manifest_uri"],
            report_uri="s3://bkt/verify/roundtrip_report.json",
            status="passed",
            expected=2,
            matched=2,
            checksum_verified=2,
        )

    monkeypatch.setattr("npa.sdk.workbench.encord.verify", fake_verify)
    result = runner.invoke(
        app,
        [
            "workbench", "encord", "verify",
            "--receipt-uri", "s3://bkt/push/push_receipt.json",
            "--manifest-uri", "s3://bkt/pull/manifest.json",
            "--output-path", "s3://bkt/verify/",
            "--output", "text",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "roundtrip passed: 2/2 matched" in result.output

    def failing_verify(**kwargs):
        raise EncordToolError("Encord roundtrip verification failed: 1 missing")

    monkeypatch.setattr("npa.sdk.workbench.encord.verify", failing_verify)
    result = runner.invoke(
        app,
        [
            "workbench", "encord", "verify",
            "--receipt-uri", "s3://bkt/push/push_receipt.json",
            "--manifest-uri", "s3://bkt/pull/manifest.json",
            "--output-path", "s3://bkt/verify/",
        ],
    )
    assert result.exit_code == 1 and "1 missing" in result.output
    # local paths violate the contract
    result = runner.invoke(
        app,
        [
            "workbench", "encord", "verify",
            "--receipt-uri", "/tmp/push_receipt.json",
            "--manifest-uri", "s3://bkt/pull/manifest.json",
            "--output-path", "s3://bkt/verify/",
        ],
    )
    assert result.exit_code == 1


def test_pull_happy_path_text(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_pull(**kwargs):
        captured.update(kwargs)
        return _manifest()

    monkeypatch.setattr("npa.sdk.workbench.encord.pull", fake_pull)
    result = runner.invoke(
        app,
        [
            "workbench", "encord", "pull",
            "--source", "collection",
            "--source-id", "keepers",
            "--output-path", "s3://bkt/pull/",
            "--output", "text",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "pulled 1 item(s)" in result.output
    assert captured["source"] == "collection"
    assert captured["source_id"] == "keepers"


def test_pull_rejects_local_output_and_bad_source() -> None:
    result = runner.invoke(
        app,
        [
            "workbench", "encord", "pull",
            "--source", "collection",
            "--source-id", "x",
            "--output-path", "/tmp/out",
        ],
    )
    assert result.exit_code == 1
    result = runner.invoke(
        app,
        [
            "workbench", "encord", "pull",
            "--source", "everything",
            "--source-id", "x",
            "--output-path", "s3://bkt/pull/",
        ],
    )
    assert result.exit_code != 0


def test_pull_missing_credential_message(monkeypatch: pytest.MonkeyPatch) -> None:
    # No monkeypatched SDK: run_pull resolves the endpoint from environ then
    # fails on auth; both are acceptable fail-closed messages, never a traceback.
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://storage.test")
    result = runner.invoke(
        app,
        [
            "workbench", "encord", "pull",
            "--source", "collection",
            "--source-id", "00000000-0000-0000-0000-000000000009",
            "--output-path", "s3://bkt/pull/",
        ],
    )
    assert result.exit_code == 1
    assert "ENCORD_SSH_KEY" in result.output or "credential" in result.output.lower()


def test_seed_demo_skips_and_seeds(monkeypatch: pytest.MonkeyPatch) -> None:
    import npa.workflows.encord_loop as loop

    captured: dict = {}

    def fake_seed(media_uri, dataset, active_source_id, *, transfer, integration):
        captured.update(dict(media_uri=media_uri, dataset=dataset,
                             active=active_source_id, transfer=transfer))
        if active_source_id != dataset:
            return {"stage": "seed_demo_source", "skipped": "operator supplied a curated source id"}
        return {"stage": "seed_demo_source", "dataset": dataset, "units_done": 1}

    monkeypatch.setattr(loop, "seed_demo_source", fake_seed)
    result = runner.invoke(
        app,
        ["workbench", "encord", "seed-demo",
         "--media-uri", "s3://bkt/run/seed/",
         "--dataset", "npa-demo-src-run",
         "--active-source-id", "npa-demo-src-run",
         "--output", "text"],
    )
    assert result.exit_code == 0, result.output
    assert "seeded demo dataset" in result.output
    assert captured["transfer"] == "upload"
    result = runner.invoke(
        app,
        ["workbench", "encord", "seed-demo",
         "--media-uri", "s3://bkt/run/seed/",
         "--dataset", "npa-demo-src-run",
         "--active-source-id", "my-curated", "--output", "text"],
    )
    assert result.exit_code == 0 and "seed skipped" in result.output
    result = runner.invoke(
        app,
        ["workbench", "encord", "seed-demo",
         "--media-uri", "/tmp/local", "--dataset", "d", "--active-source-id", "d"],
    )
    assert result.exit_code == 1  # path contract holds
