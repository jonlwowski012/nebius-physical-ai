"""The Encord to GR00T fine-tuning spec.

The thing this spec has to get right is cohort discipline: the validation
episodes choose a checkpoint, and the final episodes are read exactly once, by
the two stages that form the headline comparison. If those ever collide, the
run reports its own selection back to itself. Most of what follows checks that
the rendered commands actually keep them apart.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.scheduler import build_scheduler_plan
from npa.orchestration.npa_workflow.skypilot_render import SkypilotRenderOptions
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.npa_workflow.submit import prepare_npa_workflow_for_submit

REPO_ROOT = Path(__file__).resolve().parents[4]
SPEC_PATH = (
    REPO_ROOT / "npa/workflows/workbench/npa-workflows/encord-groot-finetune.yaml"
)

STATES = [
    "prepare-dataset",
    "push",
    "curate",
    "pull",
    "verify",
    "prepare-split",
    "preflight",
    "baseline-validation",
    "baseline-final",
    "train",
    "validate-checkpoints",
    "resolve-checkpoint",
    "final-eval",
    "compare",
    "emit-rrd",
    "emit-mcap",
    "publish",
]


def _rendered(run_id: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, dict]:
    """Render the spec to SkyPilot documents keyed by stage name."""
    registry = "cr.ci.invalid/workbench"
    monkeypatch.setenv("NPA_REGISTRY", registry)
    monkeypatch.setenv("NPA_PUBLIC_REGISTRY", "ghcr.io/nebius/nebius-physical-ai")
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/source/npa")
    prepared = prepare_npa_workflow_for_submit(
        SPEC_PATH,
        run_id=run_id,
        render_options=SkypilotRenderOptions(
            registry=registry, materialize_registry_secrets=False
        ),
    )
    try:
        documents = [
            doc
            for doc in yaml.safe_load_all(prepared.skypilot_yaml_path.read_text())
            if doc
        ]
        assert len(documents) == len(STATES) + 1
        return {stage["name"]: stage for stage in documents[1:]}
    finally:
        prepared.temp_dir.cleanup()


def test_spec_orders_conversion_before_curation() -> None:
    """Encord curates media items, so per-episode videos must exist first."""
    spec = load_spec(SPEC_PATH)

    assert list(spec.states) == STATES
    assert spec.initial == "prepare-dataset"
    assert spec.states["publish"].terminal is True
    order = {name: index for index, name in enumerate(STATES)}
    assert order["prepare-dataset"] < order["push"]
    assert order["verify"] < order["prepare-split"]
    # Selection happens before anything reads the final cohort.
    assert order["validate-checkpoints"] < order["final-eval"]


def test_every_stage_uses_a_catalog_tool() -> None:
    """No inline argv: each stage goes through an audited toolRef."""
    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    spec = load_spec(SPEC_PATH)
    for name, state in spec.states.items():
        assert state.tool_ref, f"{name} has no toolRef"
        assert state.tool_ref in TOOL_CATALOG, f"{name}: {state.tool_ref}"


def test_the_validation_cohort_never_reports_the_headline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cohort that chose the checkpoint must not also grade it."""
    by_name = _rendered("encord-groot-cohorts", monkeypatch)

    assert "--split-role heldout" in by_name["baseline-validation"]["run"]
    assert "--split-role final" in by_name["baseline-final"]["run"]
    assert "--split-role final" in by_name["final-eval"]["run"]
    # Exactly two stages read the final cohort.
    reads_final = [
        name for name, stage in by_name.items() if "--split-role final" in stage["run"]
    ]
    assert sorted(reads_final) == ["baseline-final", "final-eval"]


def test_selection_ceiling_comes_from_the_validation_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A final-cohort ceiling would leak the final split into the selection."""
    by_name = _rendered("encord-groot-ceiling", monkeypatch)
    run = by_name["validate-checkpoints"]["run"]

    assert "/validation/baseline/evaluation.json" in run
    assert "/offline/baseline/evaluation.json" not in run
    # The comparison, by contrast, reads the final-cohort baseline.
    assert "/offline/baseline/evaluation.json" in by_name["compare"]["run"]


def test_training_is_curated_and_tracked(monkeypatch: pytest.MonkeyPatch) -> None:
    by_name = _rendered("encord-groot-training", monkeypatch)

    split = by_name["prepare-split"]["run"]
    assert "--curation-manifest-uri" in split
    assert "--curation-report-uri" in split
    assert "/encord/pull/manifest.json" in split

    train = by_name["train"]["run"]
    assert "workbench groot finetune" in train
    # W&B is this application's default, expressed in its own spec.
    assert "--wandb-mode online" in train
    assert "--save-steps 100" in train
    assert "--max-steps 500" in train

    resolve = by_name["resolve-checkpoint"]["run"]
    assert "--selection-uri" in resolve
    assert "/reports/selected-checkpoint.json" in resolve


def test_gpu_stages_are_the_expensive_ones_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Curation and reporting must not hold a GPU."""
    spec = load_spec(SPEC_PATH)
    gpu_states = {
        name for name, state in spec.states.items() if state.resources == "gpu"
    }

    assert gpu_states == {
        "baseline-validation",
        "baseline-final",
        "train",
        "validate-checkpoints",
        "final-eval",
    }
    scheduler = build_scheduler_plan(
        spec,
        build_plan(spec, run_id="encord-groot-gpu").steps,
        run_id="encord-groot-gpu",
    )
    for task in scheduler["tasks"]:
        accelerators = task["resources"].get("accelerators")
        assert bool(accelerators) is (task["name"] in gpu_states), task["name"]
        if accelerators:
            assert accelerators == "H100:1"


def test_the_checkpoint_schedule_leaves_candidates_to_select_from() -> None:
    """Selecting a checkpoint is meaningless if only one is retained."""
    spec = load_spec(SPEC_PATH)
    max_steps = int(spec.config["max_steps"])
    save_steps = int(spec.config["save_steps"])

    assert max_steps % save_steps == 0
    saved = max_steps // save_steps
    assert saved > 1
    assert int(spec.config["save_total_limit"]) >= saved


def test_the_final_cohort_is_actually_materialised() -> None:
    """A once-touched cohort that nobody splits out would never be read."""
    spec = load_spec(SPEC_PATH)

    assert int(spec.config["final_episodes"]) > 0
    assert int(spec.config["heldout_episodes"]) > 0
    assert spec.config["evaluation_split_role"] == "final"


@pytest.mark.parametrize("gpu_count", [1, 2, 8])
def test_spec_plans_across_gpu_counts(gpu_count: int) -> None:
    spec = copy.deepcopy(load_spec(SPEC_PATH))
    spec.config.update(
        {
            "gpu_count": str(gpu_count),
            "per_device_batch_size": "2",
            "gradient_accumulation_steps": "1",
            "global_batch_size": str(gpu_count * 2),
        }
    )
    run_id = f"encord-groot-gpu-{gpu_count}"

    plan = build_plan(spec, run_id=run_id)
    scheduler = build_scheduler_plan(spec, plan.steps, run_id=run_id)

    assert [task["name"] for task in scheduler["tasks"]] == STATES
    train = next(t for t in scheduler["tasks"] if t["name"] == "train")
    assert train["resources"]["accelerators"] == f"H100:{gpu_count}"


def test_manual_curation_variant_still_plans() -> None:
    """The guide documents pausing for a human pass; it must remain valid."""
    spec = copy.deepcopy(load_spec(SPEC_PATH))
    spec.config.update(
        {"encord_curate_filters": "", "encord_curate_poll_seconds": "86400"}
    )
    run_id = "encord-groot-manual"

    scheduler = build_scheduler_plan(
        spec, build_plan(spec, run_id=run_id).steps, run_id=run_id
    )

    assert [task["name"] for task in scheduler["tasks"]] == STATES


def test_spec_is_registered_for_live_submission() -> None:
    """A shipped reference spec needs a live submit case, not just a file."""
    from npa.orchestration.npa_workflow import submit_matrix

    source = Path(submit_matrix.__file__).read_text(encoding="utf-8")
    assert "encord-groot-finetune.yaml" in source
    assert "ENCORD_SSH_KEY_B64" in source


def test_gpu_stages_get_the_groot_image_not_the_default_one() -> None:
    """A stage that imports gr00t or shells to ffmpeg cannot run image-less.

    Caught on the way to a live submit: `workflow.groot.validate_checkpoints`
    runs real `Gr00tPolicy` forwards and `workflow.groot.prepare_dataset` shells
    out to ffmpeg to split packed LeRobot v3 video, yet both resolved to no
    image and would have landed on SkyPilot's default one.
    """
    from npa.orchestration.npa_workflow.skypilot_render import tool_image_key

    spec = load_spec(SPEC_PATH)
    needs_groot = {
        "prepare-dataset",
        "baseline-validation",
        "baseline-final",
        "train",
        "validate-checkpoints",
        "final-eval",
    }

    for name in needs_groot:
        tool = spec.states[name].tool_ref
        assert tool_image_key(tool) == "groot", (
            f"{name} ({tool}) must run in the GR00T image; it resolved to "
            f"{tool_image_key(tool)!r}"
        )

    # The reporting stages deliberately stay on the default image plus staged
    # source, which is where their [viz] extra comes from.
    for name in ("compare", "emit-rrd", "emit-mcap", "publish", "prepare-split"):
        tool = spec.states[name].tool_ref
        assert tool_image_key(tool) is None, f"{name} ({tool}) should stay image-less"


def test_every_dataset_reader_after_conversion_reads_the_converted_dataset() -> None:
    """Only `prepare-dataset` may read the raw input.

    Found by live run `encord-groot-finetune-20260908T181240Z`, which failed at
    `prepare-split` with `NoSuchKey` on
    `datasets/so100-pickplace/meta/modality.json`. `modality.json` is a GR00T
    artifact that *conversion creates*, so a raw LeRobot v3 dataset has none.
    The shared `workflow.groot.prepare_split` catalog entry defaults
    `--source-uri` to `config.source_data_uri`, which is correct for
    `groot-1-7-finetune.yaml` (already GR00T, no conversion stage) and wrong
    here, so this spec overrides it per state.

    The dangerous part is that the spec validates, plans, and renders cleanly
    either way; the failure only appears minutes into a live submit.
    """
    spec = load_spec(SPEC_PATH)
    steps = {step.state: list(step.argv) for step in build_plan(spec, run_id="conv").steps}

    raw = "datasets/lerobot-source/"
    prepared = "data/prepared/"

    source = steps["prepare-dataset"]
    assert raw in source[source.index("--source-uri") + 1], (
        "prepare-dataset must read the raw input dataset"
    )

    split = steps["prepare-split"]
    read = split[split.index("--source-uri") + 1]
    assert prepared in read and raw not in read, (
        f"prepare-split reads {read!r}; it must read the converted dataset, "
        "because meta/modality.json only exists after conversion"
    )

    for state, argv in steps.items():
        if state == "prepare-dataset":
            continue
        offenders = [value for value in argv if raw in str(value)]
        assert not offenders, (
            f"state {state!r} reads the raw dataset {offenders!r}; every stage "
            "after conversion must consume the converted dataset"
        )
