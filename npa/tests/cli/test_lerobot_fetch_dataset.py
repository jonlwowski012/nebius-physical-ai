"""``npa workbench lerobot fetch-dataset`` stages a reproducible dataset.

The command's job is to make a later training run provable: an immutable
revision, a validated LeRobot contract, and a provenance receipt that travels
with the staged bytes.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from npa.cli.main import app

runner = CliRunner()

REVISION = "728583b5eaf9e739a7f119e2def466fa1d552402"
REPO = "lerobot/svla_so100_pickplace"


@pytest.mark.parametrize(
    "revision",
    ["main", "v1.0", "728583b5", "", "728583b5eaf9e739a7f119e2def466fa1d55240Z"],
)
def test_fetch_dataset_requires_an_immutable_revision(revision: str) -> None:
    """A branch or short SHA can move, so a "reproduced" run would not be."""
    result = runner.invoke(
        app,
        [
            "workbench",
            "lerobot",
            "fetch-dataset",
            "--hf-repo",
            REPO,
            "--revision",
            revision,
            "--output-path",
            "s3://bucket/datasets/so100/",
        ],
    )

    assert result.exit_code != 0
    assert "40-character commit SHA" in result.output


def test_fetch_dataset_stages_and_records_provenance(tmp_path: Path, mocker) -> None:
    local = tmp_path / "dataset"
    (local / "meta").mkdir(parents=True)
    download = mocker.patch(
        "npa.workflows.lerobot_dataset.download_public_lerobot_dataset",
        return_value=local,
    )
    summarize = mocker.patch(
        "npa.workflows.lerobot_dataset.summarize_lerobot_dataset",
        return_value=SimpleNamespace(
            total_episodes=50,
            total_frames=19631,
            fps=30,
            camera_keys=["observation.images.top", "observation.images.wrist"],
        ),
    )
    stage = mocker.patch(
        "npa.workflows.lerobot_dataset.stage_dataset_to_s3",
        return_value="s3://bucket/datasets/so100/",
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "lerobot",
            "fetch-dataset",
            "--hf-repo",
            REPO,
            "--revision",
            REVISION,
            "--output-path",
            "s3://bucket/datasets/so100/",
            "--license",
            "apache-2.0",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert download.call_args.kwargs["repo_id"] == REPO
    assert download.call_args.kwargs["revision"] == REVISION
    assert summarize.call_args.kwargs["license"] == "apache-2.0"

    payload = json.loads(result.output)
    assert payload["status"] == "staged"
    assert payload["revision"] == REVISION
    assert payload["license"] == "apache-2.0"
    assert payload["total_episodes"] == 50
    assert payload["camera_keys"] == [
        "observation.images.top",
        "observation.images.wrist",
    ]
    assert payload["source_uri"] == f"hf://datasets/{REPO}@{REVISION}"

    # The receipt must travel with the bytes, not only in stdout.
    receipt = json.loads((local / "meta" / "npa_dataset_provenance.json").read_text())
    assert receipt["schema"] == "npa.lerobot.dataset_provenance.v1"
    assert receipt["revision"] == REVISION
    assert receipt["staged_uri"] == "s3://bucket/datasets/so100/"
    stage.assert_called_once()


def test_fetch_dataset_reports_a_rejected_dataset_without_staging(
    tmp_path: Path, mocker
) -> None:
    from npa.workflows.lerobot_dataset import LeRobotDatasetError

    local = tmp_path / "dataset"
    (local / "meta").mkdir(parents=True)
    mocker.patch(
        "npa.workflows.lerobot_dataset.download_public_lerobot_dataset",
        return_value=local,
    )
    mocker.patch(
        "npa.workflows.lerobot_dataset.summarize_lerobot_dataset",
        side_effect=LeRobotDatasetError("missing required feature(s): action"),
    )
    stage = mocker.patch("npa.workflows.lerobot_dataset.stage_dataset_to_s3")

    result = runner.invoke(
        app,
        [
            "workbench",
            "lerobot",
            "fetch-dataset",
            "--hf-repo",
            REPO,
            "--revision",
            REVISION,
            "--output-path",
            "s3://bucket/datasets/so100/",
        ],
    )

    assert result.exit_code != 0
    assert "missing required feature" in result.output
    stage.assert_not_called()
