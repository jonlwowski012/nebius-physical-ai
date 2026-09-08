"""The fine-tuning spec must be schedulable on the nodes it asks for.

Found the hard way. The spec originally requested `cpus: 16` for its GPU
profile and `cpus: 8` / `memory: 32Gi` for its CPU profile. A Nebius node
advertises roughly `vcpu - 0.1` CPU and about 1Gi less memory after kubelet and
system reserve, so a 16-vCPU node offers 15.9 and an 8-vCPU node offers 7.9 with
about 30.7Gi. Every GPU stage would have sat unschedulable forever, and the CPU
stages would have landed or not depending on which node the scheduler picked.

A request that is only *just* too big is the dangerous case: `plan-spec` is
happy, `validate-spec` is happy, and the failure appears minutes into a real
submit as `ResourcesUnavailableError`. This pins the sizes against the presets
the spec is expected to run on.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC = REPO_ROOT / "npa/workflows/workbench/npa-workflows/encord-groot-finetune.yaml"

#: Headroom a Kubernetes node never offers to pods (kubelet + system reserve).
#: Measured on a live Nebius cluster: a 16-vCPU node advertised 15900m and a
#: 32GB node advertised about 30.7Gi.
CPU_RESERVE = 0.5
MEMORY_RESERVE_GIB = 2.0

#: The node shapes this spec is documented to run on. The GPU entries are the
#: two accelerators the guide names, and the CPU entry is the smallest node a
#: default cluster provisions.
GPU_PRESETS = {"H100": (16.0, 200.0), "L40S": (16.0, 92.0)}
CPU_PRESET = (8.0, 32.0)


def _memory_gib(value: object) -> float:
    match = re.match(r"([\d.]+)\s*(Gi|G|Mi|M)?$", str(value).strip())
    assert match, f"unrecognized memory request {value!r}"
    amount = float(match.group(1))
    return amount / 1024 if match.group(2) in {"Mi", "M"} else amount


def _profiles() -> dict[str, dict]:
    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8")) or {}
    return {
        name: profile
        for name, profile in (spec.get("resources") or {}).items()
        if isinstance(profile, dict)
    }


def _is_gpu(profile: dict) -> bool:
    return bool(str(profile.get("accelerators", "") or "").strip())


def test_cpu_stages_fit_the_smallest_default_node() -> None:
    vcpu, memory_gib = CPU_PRESET
    profiles = {n: p for n, p in _profiles().items() if not _is_gpu(p)}
    assert profiles, "no CPU-only resource profile found"

    for name, profile in profiles.items():
        cpus = float(profile.get("cpus", 0) or 0)
        memory = _memory_gib(profile.get("memory", "0Gi"))
        assert cpus <= vcpu - CPU_RESERVE, (
            f"profile {name!r} requests {cpus:g} CPU; a {vcpu:g}-vCPU node offers "
            f"about {vcpu - 0.1:g} allocatable and cannot schedule it"
        )
        assert memory <= memory_gib - MEMORY_RESERVE_GIB, (
            f"profile {name!r} requests {memory:g}Gi; a {memory_gib:g}GB node offers "
            "about 1Gi less after reserve"
        )


@pytest.mark.parametrize("accelerator", sorted(GPU_PRESETS))
def test_gpu_stages_fit_every_documented_accelerator(accelerator: str) -> None:
    """The guide offers `--var gpu_type=L40S`, so both shapes have to work."""
    vcpu, memory_gib = GPU_PRESETS[accelerator]
    profiles = {n: p for n, p in _profiles().items() if _is_gpu(p)}
    assert profiles, "no GPU resource profile found"

    for name, profile in profiles.items():
        cpus = float(profile.get("cpus", 0) or 0)
        memory = _memory_gib(profile.get("memory", "0Gi"))
        assert cpus <= vcpu - CPU_RESERVE, (
            f"profile {name!r} requests {cpus:g} CPU, which a {accelerator} node "
            f"({vcpu:g} vCPU) cannot schedule after reserve"
        )
        assert memory <= memory_gib - MEMORY_RESERVE_GIB, (
            f"profile {name!r} requests {memory:g}Gi, which a {accelerator} node "
            f"({memory_gib:g}GB) cannot schedule after reserve"
        )


def test_the_guard_would_catch_the_original_mistake() -> None:
    """A request 0.1 CPU over allocatable must fail, not squeak through."""
    vcpu = CPU_PRESET[0]

    just_too_big = vcpu - 0.05
    assert not (just_too_big <= vcpu - CPU_RESERVE), (
        "the reserve is too small to catch a request that only just exceeds "
        "allocatable capacity, which is the failure mode this guard exists for"
    )
