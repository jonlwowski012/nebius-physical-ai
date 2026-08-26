from __future__ import annotations

from pathlib import Path

from npa.orchestration.npa_workflow import build_plan, load_spec, validate_spec
from npa.orchestration.npa_workflow.catalog import argv_for_tool

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"
PUSH = WORKFLOWS / "encord-push.yaml"
PULL = WORKFLOWS / "encord-pull.yaml"
ROUNDTRIP = WORKFLOWS / "encord-roundtrip-smoke.yaml"
AUGMENT = WORKFLOWS / "encord-cosmos3-augment.yaml"


def test_push_spec_is_a_single_terminal_state() -> None:
    spec = load_spec(PUSH)
    validate_spec(spec)
    assert spec.name == "encord-push"
    steps = [step.state for step in build_plan(spec, run_id="t").steps]
    assert steps == ["push"]
    outputs = spec.states["push"].outputs
    assert outputs and outputs[0].schema == "npa.encord.push_receipt.v1"
    assert outputs[0].uri.endswith("push_receipt.json")
    # Human curation happens between the workflows: push declares no inputs.
    assert not spec.states["push"].inputs


def test_pull_spec_is_a_single_terminal_state() -> None:
    spec = load_spec(PULL)
    validate_spec(spec)
    assert spec.name == "encord-pull"
    steps = [step.state for step in build_plan(spec, run_id="t").steps]
    assert steps == ["pull"]
    outputs = spec.states["pull"].outputs
    assert outputs and outputs[0].schema == "npa.encord.pull_manifest.v1"
    assert outputs[0].uri.endswith("manifest.json")


def test_roundtrip_smoke_chains_push_then_pull() -> None:
    spec = load_spec(ROUNDTRIP)
    validate_spec(spec)
    assert spec.name == "encord-roundtrip-smoke"
    steps = [step.state for step in build_plan(spec, run_id="t").steps]
    assert steps == ["push", "pull"]
    assert spec.states["pull"].needs == ["push"]
    # pull consumes the receipt schema push produces (schema-chaining contract).
    assert spec.states["pull"].inputs[0].schema == "npa.encord.push_receipt.v1"
    assert spec.states["push"].outputs[0].schema == "npa.encord.push_receipt.v1"
    # The e2e pulls the dataset push just created, resolved by run-scoped title.
    assert spec.config["encord_source"] == "dataset"
    assert spec.config["encord_source_id"] == spec.config["encord_dataset"]


def test_specs_declare_cpu_resource_blocks() -> None:
    for path in (PUSH, PULL, ROUNDTRIP):
        spec = load_spec(path)
        assert "cpu" in spec.resources, path.name
        profile = spec.resources["cpu"]
        assert profile.get("cloud") == "kubernetes", path.name
        assert "accelerators" not in profile, f"{path.name}: encord stages are CPU-only"
        for state in spec.states.values():
            assert state.resources == "cpu", f"{path.name}:{state.name}"


def test_push_argv_renders_every_flag() -> None:
    argv = argv_for_tool("workbench.encord.push")
    assert argv[:4] == ["npa", "workbench", "encord", "push"]
    for flag in (
        "--input-path",
        "--integration",
        "--folder",
        "--dataset",
        "--media",
        "--poll-timeout-seconds",
        "--output-path",
        "--workflow-run",
        "--output",
    ):
        assert flag in argv, flag
    assert "{{run.id}}" in argv


def test_pull_argv_renders_every_flag() -> None:
    argv = argv_for_tool("workbench.encord.pull")
    assert argv[:4] == ["npa", "workbench", "encord", "pull"]
    for flag in ("--source", "--source-id", "--output-path", "--workflow-run", "--output"):
        assert flag in argv, flag


def test_push_plan_omits_dataset_flag_when_empty() -> None:
    spec = load_spec(PUSH)
    assert "--dataset" in build_plan(spec, run_id="t").steps[0].argv
    spec.config["encord_dataset"] = ""
    argv = build_plan(spec, run_id="t").steps[0].argv
    assert "--dataset" not in argv


def test_augment_loop_chains_pull_stage_two_generations_push() -> None:
    spec = load_spec(AUGMENT)
    validate_spec(spec)
    plan = build_plan(spec, run_id="t")
    steps = [step.state for step in plan.steps]
    assert steps == ["seed-source", "pull", "stage-input", "augment", "augment", "push-augmented"]
    assert spec.states["pull"].needs == ["seed-source"]
    assert spec.states["stage-input"].needs == ["pull"]
    assert spec.states["augmentations"].needs == ["stage-input"]
    assert spec.config["augmentation_count"] == "2"
    assert spec.states["augmentations"].loop.max == "{{config.augmentation_count}}"
    assert spec.states["augmentations"].sequence == ["augment"]
    assert spec.states["push-augmented"].needs == ["augmentations"]
    assert spec.states["augment"].resources == "gpu"
    # Default source is the self-seeded demo dataset; the seed argv carries the
    # same run-scoped title so an operator override makes seeding a no-op.
    seed_argv = next(
        s.argv for s in build_plan(spec, run_id="t").steps if s.state == "seed-source"
    )
    assert seed_argv[:4] == ["npa", "workbench", "encord", "seed-demo"]
    assert seed_argv.count("npa-demo-src-t") == 2  # demo title + defaulted source id
    assert "--integration" not in seed_argv  # omitted while the default is empty
    for name in ("seed-source", "pull", "stage-input", "push-augmented"):
        assert spec.states[name].resources == "cpu", name
    # Schema chain: manifest -> staged video -> generated attestations -> receipt.
    assert spec.states["stage-input"].inputs[0].schema == "npa.encord.pull_manifest.v1"
    assert spec.states["augment"].inputs[0].schema == "video/mp4"
    assert spec.states["augment"].outputs[0].schema == "npa.cosmos3.generate.v1"
    assert spec.states["push-augmented"].outputs[0].schema == "npa.encord.push_receipt.v1"
    # Each loop iteration conditions on the staged video, has a distinct seed,
    # and publishes a distinct video and attestation for push.
    stage_argv = next(s.argv for s in plan.steps if s.state == "stage-input")
    staged_uri = stage_argv[-2]  # the glue's dest_uri positional
    assert staged_uri.endswith("augment-input/source.mp4")
    variant_steps = [step for step in plan.steps if step.state == "augment"]
    assert all(staged_uri in step.argv for step in variant_steps)
    seeds = [step.argv[step.argv.index("--seed") + 1] for step in variant_steps]
    output_paths = [step.argv[step.argv.index("--output-path") + 1] for step in variant_steps]
    assert seeds == ["1", "2"]
    assert len(set(output_paths)) == 2
    assert all("/generated/variant-" in path for path in output_paths)
    attestation_paths = [step.outputs[0]["uri"] for step in variant_steps]
    assert len(set(attestation_paths)) == 2
    assert all("/generated/variant-" in path for path in attestation_paths)
