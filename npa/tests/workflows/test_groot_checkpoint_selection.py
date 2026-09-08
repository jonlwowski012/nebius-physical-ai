"""Per-checkpoint validation and the rule that picks one.

Falling training loss does not tell a company which checkpoint to ship. These
tests cover the two things that do: measuring every saved checkpoint on the
validation split, and a selection rule that refuses to crown the least bad
checkpoint when none of them actually beat the base model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from npa.workflows import groot_learning as learning
from npa.workflows.groot_learning import (
    CHECKPOINT_SELECTION_SCHEMA,
    GrootVisualizationError,
    VALIDATION_CURVE_SCHEMA,
    select_checkpoint,
    validate_checkpoints,
)

RUN_ID = "run-1"


# --------------------------------------------------------------------------
# select_checkpoint
# --------------------------------------------------------------------------


def _candidates(*pairs: tuple[int, float]) -> list[dict[str, Any]]:
    return [{"step": step, "mse": mse} for step, mse in pairs]


def test_selection_takes_the_lowest_validation_error() -> None:
    result = select_checkpoint(
        _candidates((100, 0.5), (200, 0.3), (300, 0.4)),
        baseline_mse=1.0,
        minimum_relative_improvement=0.10,
    )

    assert result["selected_step"] == 200
    assert result["selected_mse"] == pytest.approx(0.3)
    assert result["relative_improvement"] == pytest.approx(0.7)
    assert result["outcome"] == "selected"


def test_a_tie_resolves_to_the_earlier_step() -> None:
    """Same error from less training is the cheaper and less overfit model."""
    result = select_checkpoint(
        _candidates((400, 0.3), (200, 0.3), (600, 0.3)),
        baseline_mse=1.0,
        minimum_relative_improvement=0.10,
    )

    assert result["selected_step"] == 200


def test_selection_refuses_when_nothing_beats_the_baseline() -> None:
    result = select_checkpoint(
        _candidates((100, 0.95), (200, 0.91)),
        baseline_mse=1.0,
        minimum_relative_improvement=0.10,
    )

    assert result["selected_step"] is None
    assert result["outcome"] == "none_beat_baseline"
    # The report names the closest candidate so the next run has somewhere to start.
    assert "step 200" in result["detail"]
    assert result["eligibility_ceiling_mse"] == pytest.approx(0.9)


def test_a_candidate_exactly_on_the_margin_is_eligible() -> None:
    result = select_checkpoint(
        _candidates((100, 0.9)), baseline_mse=1.0, minimum_relative_improvement=0.10
    )

    assert result["selected_step"] == 100


def test_selection_ignores_a_non_finite_candidate() -> None:
    result = select_checkpoint(
        _candidates((100, float("nan")), (200, 0.2)),
        baseline_mse=1.0,
        minimum_relative_improvement=0.10,
    )

    assert result["selected_step"] == 200
    assert result["candidates_considered"] == 1


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        pytest.param(
            {
                "candidates": [],
                "baseline_mse": 1.0,
                "minimum_relative_improvement": 0.1,
            },
            "at least one candidate",
            id="no-candidates",
        ),
        pytest.param(
            {
                "candidates": _candidates((100, 0.2)),
                "baseline_mse": 0.0,
                "minimum_relative_improvement": 0.1,
            },
            "finite and positive",
            id="zero-baseline",
        ),
        pytest.param(
            {
                "candidates": _candidates((100, 0.2)),
                "baseline_mse": float("nan"),
                "minimum_relative_improvement": 0.1,
            },
            "finite and positive",
            id="non-finite-baseline",
        ),
        pytest.param(
            {
                "candidates": _candidates((100, 0.2)),
                "baseline_mse": 1.0,
                "minimum_relative_improvement": 1.0,
            },
            "below one",
            id="impossible-margin",
        ),
    ],
)
def test_selection_rejects_an_unusable_comparison(kwargs: dict, expected: str) -> None:
    with pytest.raises(GrootVisualizationError, match=expected):
        select_checkpoint(
            kwargs.pop("candidates"),
            **kwargs,
        )


# --------------------------------------------------------------------------
# validate_checkpoints
# --------------------------------------------------------------------------


class _Body:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def seed_json(self, bucket: str, key: str, payload: dict[str, Any]) -> None:
        self.objects[(bucket, key)] = json.dumps(payload).encode()

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        if (Bucket, Key) not in self.objects:
            raise KeyError(f"missing s3://{Bucket}/{Key}")
        return {"Body": _Body(self.objects[(Bucket, Key)])}

    def put_object(
        self, *, Bucket: str, Key: str, Body: bytes, ContentType: str = ""
    ) -> dict[str, Any]:
        self.objects[(Bucket, Key)] = Body
        return {}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        body = self.objects[(Bucket, Key)]
        return {
            "ContentLength": len(body),
            "ETag": f'"{hashlib.md5(body).hexdigest()}"',
        }

    def json_at(self, bucket: str, key: str) -> dict[str, Any]:
        return json.loads(self.objects[(bucket, key)])


@pytest.fixture()
def validation_run(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeS3, dict[int, float]]:
    """A completed 3-checkpoint run whose candidate errors the test controls."""

    client = FakeS3()
    client.seed_json(
        "bucket",
        "reports/split.json",
        {
            "schema": learning.SPLIT_SCHEMA,
            "status": "prepared",
            "run_id": RUN_ID,
            "split_hash": "s" * 64,
            "heldout": {"uri": "s3://bucket/data/validation/"},
        },
    )
    client.seed_json(
        "bucket",
        "checkpoints/manifest.json",
        {
            "schema": "npa.groot.finetune.v1",
            "status": "completed",
            "run_id": RUN_ID,
            "checkpoint_steps": [100, 200, 300],
        },
    )
    client.seed_json(
        "bucket",
        "offline/baseline/evaluation.json",
        {
            "schema": learning.EVAL_SCHEMA,
            "status": "completed",
            "run_id": RUN_ID,
            "phase": "baseline",
            "metrics": {"mse": 1.0, "mae": 0.8},
        },
    )

    errors = {100: 0.8, 200: 0.2, 300: 0.25}

    monkeypatch.setattr(learning, "_download_prefix", lambda *_a, **_k: [Path("x")])
    monkeypatch.setattr(
        learning,
        "_checkpoint_identity",
        lambda path: {"sha256": "a" * 64, "weights_sha256": "b" * 64},
    )
    monkeypatch.setattr(learning, "checkpoint_model_config_contract", lambda path: {})
    monkeypatch.setattr(learning, "validate_evaluation", lambda *_a, **_k: None)

    def fake_evaluate(*, checkpoint_path: Path, **_kwargs: Any) -> dict[str, Any]:
        step = int(str(checkpoint_path).rsplit("-", 1)[-1])
        return {"metrics": {"mse": errors[step], "mae": errors[step] / 2}}

    monkeypatch.setattr(learning, "_evaluate_checkpoint", fake_evaluate)
    return client, errors


def _run(client: FakeS3, **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "split_manifest_uri": "s3://bucket/reports/split.json",
        "training_manifest_uri": "s3://bucket/checkpoints/manifest.json",
        "checkpoint_uri": "s3://bucket/checkpoints/candidate",
        "baseline_eval_uri": "s3://bucket/offline/baseline/evaluation.json",
        "validation_uri": "s3://bucket/validation",
        "curve_uri": "s3://bucket/validation/curve.json",
        "selection_uri": "s3://bucket/reports/selected-checkpoint.json",
        "run_id": RUN_ID,
        "robot_embodiment": "NEW_EMBODIMENT",
        "s3_client": client,
    }
    kwargs.update(overrides)
    return validate_checkpoints(**kwargs)


def test_validation_scores_every_checkpoint_and_selects_the_best(
    validation_run: tuple[FakeS3, dict[int, float]],
) -> None:
    client, _errors = validation_run

    result = _run(client)

    assert result["schema"] == CHECKPOINT_SELECTION_SCHEMA
    assert result["status"] == "selected"
    assert result["checkpoint_steps"] == [100, 200, 300]
    assert result["selected_step"] == 200
    assert result["selected_checkpoint"]["uri"] == (
        "s3://bucket/checkpoints/candidate/checkpoint-200/"
    )

    curve = client.json_at("bucket", "validation/curve.json")
    assert curve["schema"] == VALIDATION_CURVE_SCHEMA
    assert [item["step"] for item in curve["candidates"]] == [100, 200, 300]
    assert [item["mse"] for item in curve["candidates"]] == [0.8, 0.2, 0.25]
    assert curve["baseline"]["mse"] == 1.0

    # Every candidate leaves its own evaluation behind, not just the winner.
    for step in (100, 200, 300):
        published = client.json_at(
            "bucket", f"validation/checkpoint-{step}/evaluation.json"
        )
        assert published["phase"] == "validation"
        assert published["checkpoint"]["uri"].endswith(f"checkpoint-{step}/")


def test_validation_publishes_a_curve_even_when_nothing_learned(
    validation_run: tuple[FakeS3, dict[int, float]],
) -> None:
    """A run that did not learn still owes the operator its evidence."""
    client, errors = validation_run
    errors.update({100: 1.2, 200: 0.98, 300: 1.1})

    result = _run(client)

    assert result["status"] == "none"
    assert result["selected_step"] is None
    assert "selected_checkpoint" not in result
    assert len(client.json_at("bucket", "validation/curve.json")["candidates"]) == 3


def test_validation_averages_repeated_passes(
    validation_run: tuple[FakeS3, dict[int, float]],
) -> None:
    client, _errors = validation_run

    result = _run(client, validation_repeats=3)

    curve = client.json_at("bucket", "validation/curve.json")
    assert curve["passes_per_candidate"] == 3
    assert all(item["passes"] == 3 for item in curve["candidates"])
    assert result["selected_step"] == 200


@pytest.mark.parametrize(
    "overrides, seed, expected",
    [
        pytest.param(
            {"run_id": "other"}, None, "this run's split manifest", id="wrong-run"
        ),
        pytest.param(
            {"validation_repeats": 0}, None, "at least one pass", id="zero-passes"
        ),
        pytest.param(
            {},
            (
                "bucket",
                "checkpoints/manifest.json",
                {
                    "schema": "npa.groot.finetune.v1",
                    "status": "completed",
                    "run_id": RUN_ID,
                    "checkpoint_steps": [],
                },
            ),
            "records no saved checkpoints",
            id="no-checkpoints",
        ),
        pytest.param(
            {},
            (
                "bucket",
                "checkpoints/manifest.json",
                {
                    "schema": "npa.groot.finetune.v1",
                    "status": "failed",
                    "run_id": RUN_ID,
                    "checkpoint_steps": [100],
                },
            ),
            "completed training manifest",
            id="incomplete-training",
        ),
    ],
)
def test_validation_fails_closed_on_an_untrustworthy_run(
    validation_run: tuple[FakeS3, dict[int, float]],
    overrides: dict[str, Any],
    seed: tuple[str, str, dict[str, Any]] | None,
    expected: str,
) -> None:
    client, _errors = validation_run
    if seed is not None:
        client.seed_json(*seed)

    with pytest.raises(GrootVisualizationError, match=expected):
        _run(client, **overrides)


def test_validation_rejects_a_non_finite_candidate_error(
    validation_run: tuple[FakeS3, dict[int, float]],
) -> None:
    client, errors = validation_run
    errors[200] = float("nan")

    with pytest.raises(GrootVisualizationError, match="non-finite validation error"):
        _run(client)
