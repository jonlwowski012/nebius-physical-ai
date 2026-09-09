"""The rank wrapper must force the configured W&B project.

The vendor GR00T trainer calls `wandb.init(project=...)` with its own name, and
an explicit argument beats `WANDB_PROJECT`. Live run 20260908T222914Z therefore
asked for project "npa-groot" and logged to "finetune-gr00t-n1d7", so anyone
looking in the configured project found nothing at all.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

from npa.cli.groot.training_evidence import render_training_rank_wrapper


def _run_wrapper(tmp_path: Path, env: dict[str, str]) -> tuple[str, dict]:
    """Execute the rendered wrapper with a stub wandb and vendor trainer."""
    import json
    import os

    # A stub `wandb` that records the kwargs the vendor call ends up with.
    (tmp_path / "wandb.py").write_text(
        textwrap.dedent(
            """
            import json, os
            def init(*args, **kwargs):
                path = os.environ["NPA_TEST_CAPTURE"]
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(kwargs, fh)
                return object()
            """
        ),
        encoding="utf-8",
    )
    # A stub vendor trainer that calls wandb.init the way upstream does.
    vendor = tmp_path / "gr00t" / "experiment"
    vendor.mkdir(parents=True)
    (vendor / "launch_finetune.py").write_text(
        'import wandb\nwandb.init(project="finetune-gr00t-n1d7", name="vendor")\n',
        encoding="utf-8",
    )

    capture = tmp_path / "captured.json"
    script = tmp_path / "wrapper.py"
    script.write_text(render_training_rank_wrapper(str(tmp_path)), encoding="utf-8")

    full_env = {
        **os.environ,
        "NPA_TEST_CAPTURE": str(capture),
        "PYTHONPATH": str(tmp_path),
        **env,
    }
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=full_env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    recorded = json.loads(capture.read_text()) if capture.is_file() else {}
    return proc.stdout, recorded


def test_configured_project_overrides_the_vendor_default(tmp_path: Path) -> None:
    out, kwargs = _run_wrapper(
        tmp_path,
        {
            "NPA_TRAINING_WANDB_ENABLED": "1",
            "NPA_TRAINING_WANDB_PROJECT": "npa-groot",
        },
    )
    assert "NPA_GROOT_WANDB_PROJECT_PINNED npa-groot" in out
    assert kwargs["project"] == "npa-groot", (
        f"vendor default survived: {kwargs!r}; runs would land in the wrong project"
    )


def test_configured_run_name_is_applied_when_set(tmp_path: Path) -> None:
    _, kwargs = _run_wrapper(
        tmp_path,
        {
            "NPA_TRAINING_WANDB_ENABLED": "1",
            "NPA_TRAINING_WANDB_PROJECT": "npa-groot",
            "NPA_TRAINING_WANDB_RUN_NAME": "run-42",
        },
    )
    assert kwargs["name"] == "run-42"


def test_vendor_default_is_left_alone_when_tracking_is_off(tmp_path: Path) -> None:
    """No pin without the enabled flag: this must not alter a disabled run."""
    out, kwargs = _run_wrapper(
        tmp_path, {"NPA_TRAINING_WANDB_PROJECT": "npa-groot"}
    )
    assert "PINNED" not in out
    assert kwargs["project"] == "finetune-gr00t-n1d7"


def test_a_broken_wandb_never_fails_training(tmp_path: Path) -> None:
    """Tracking is never worth killing a multi-hour job for."""
    (tmp_path / "wandb.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    vendor = tmp_path / "gr00t" / "experiment"
    vendor.mkdir(parents=True)
    (vendor / "launch_finetune.py").write_text("print('trained')\n", encoding="utf-8")

    import os

    script = tmp_path / "wrapper.py"
    script.write_text(render_training_rank_wrapper(str(tmp_path)), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": str(tmp_path),
            "NPA_TRAINING_WANDB_ENABLED": "1",
            "NPA_TRAINING_WANDB_PROJECT": "npa-groot",
        },
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "NPA_GROOT_WANDB_PROJECT_PIN_FAILED" in proc.stdout
    assert "trained" in proc.stdout
