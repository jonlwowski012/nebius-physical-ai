"""W&B has to be switched on at the vendor launcher, not through the environment.

The pinned GR00T launcher owns the decision:

    finetune_config.py:  use_wandb: bool = False
                         wandb_project: str = "finetune-gr00t-n1d7"
    launch_finetune.py:  config.training.use_wandb = ft_config.use_wandb
                         config.training.wandb_project = ft_config.wandb_project

Exporting `WANDB_MODE` / `WANDB_PROJECT` / `WANDB_API_KEY` never reaches that
switch, so every run through 20260908T222914Z logged nothing: its 80 KB
`training.log` contains no `wandb.init`, no login line and no run URL. The
vendor still wrote a `wandb_config.json` naming its own default project, which
looks like evidence of a run and is not one.
"""

from __future__ import annotations

from npa.cli.groot import _build_finetune_command
from npa.workbench.training_config import build_training_config

_COMMON = dict(
    input_path="s3://bucket/train/",
    output_path="s3://bucket/out/",
    base_model="nvidia/GR00T-N1.7-3B",
    robot_embodiment="NEW_EMBODIMENT",
    num_gpus=1,
    config="cfg",
    endpoint_url="https://endpoint",
)


def _trainer_flags(command: str) -> list[str]:
    """Return only the launcher argument lines, not the env exports."""
    return [
        line.strip()
        for line in command.splitlines()
        if line.strip().startswith("--")
    ]


def test_enabled_wandb_switches_on_the_launcher_flag() -> None:
    config = build_training_config(
        wandb_enabled=True, wandb_project="npa-groot", wandb_mode="online"
    )
    flags = _trainer_flags(_build_finetune_command(training_config=config, **_COMMON))

    assert any(f.startswith("--use-wandb") for f in flags), (
        "no --use-wandb passed; FinetuneConfig.use_wandb defaults to False, so "
        "the trainer never calls wandb.init and the run logs nothing"
    )
    assert any("--wandb-project npa-groot" in f for f in flags), (
        f"configured project not passed to the launcher: {flags!r}; the run "
        "would land in the vendor default 'finetune-gr00t-n1d7'"
    )


def test_naming_a_mode_is_enough_to_switch_it_on() -> None:
    """`--wandb-mode online` alone must enable it; there is no bare --wandb."""
    config = build_training_config(wandb_enabled=False, wandb_mode="online")
    # build_training_config records mode; the CLI derives `wandb_on` from it.
    assert config.wandb.mode == "online"


def test_disabled_wandb_passes_no_launcher_flags() -> None:
    config = build_training_config(wandb_enabled=False, wandb_mode="disabled")
    flags = _trainer_flags(_build_finetune_command(training_config=config, **_COMMON))

    assert not any("wandb" in f for f in flags), (
        f"a disabled run must not switch the launcher's W&B on: {flags!r}"
    )


def test_enabled_without_a_project_still_switches_on() -> None:
    """Omitting the project is allowed; the vendor default then applies."""
    config = build_training_config(wandb_enabled=True, wandb_mode="online")
    flags = _trainer_flags(_build_finetune_command(training_config=config, **_COMMON))

    assert any(f.startswith("--use-wandb") for f in flags)
    assert not any("--wandb-project" in f for f in flags)
