---
name: encord-groot-finetune
description: Use when fine-tuning GR00T N1.7 on a customer's own LeRobot demonstrations with Encord curating which episodes train — the three-cohort contract, per-checkpoint selection, and reading whether the policy actually learned.
---

# Encord → GR00T fine-tuning

## When To Use

Load when running, adapting, debugging, or explaining
`npa/workflows/workbench/npa-workflows/encord-groot-finetune.yaml`, or when
someone asks how to fine-tune GR00T on their own robot data and know whether it
worked.

Operator guide: `docs/workbench/guides/encord-groot-finetune.md`.

For the shorter plumbing validation this grew out of, see
`docs/workbench/cookbooks/groot-1-7-training.md` and
`npa/workflows/workbench/npa-workflows/groot-1-7-finetune.yaml`. For the Encord
verbs on their own, `skills/tools/encord/SKILL.md`. For GR00T deployment,
serving, and image concerns, `skills/tools/groot/SKILL.md`.

## The Contract That Matters

Three cohorts. Confusing them is the single failure mode that silently
invalidates a run's headline number.

| Cohort | Config | Role |
| --- | --- | --- |
| train | `train_episodes` | the episodes the optimizer sees |
| validation | `heldout_episodes` | scores every saved checkpoint and picks one |
| final | `final_episodes` | read exactly once, by two stages, and that is the headline |

The validation cohort **informs the selection**, so it cannot also be the
evidence the selection was good. That is why the base model is evaluated twice:

- `baseline-validation` (`--split-role heldout`) is the ceiling
  `validate-checkpoints` must beat, and **nothing else reads it**.
- `baseline-final` (`--split-role final`) is one half of the headline
  comparison. It reuses the checkpoint the previous stage published rather than
  rebuilding a multi-billion-parameter model.

**Never point `validate_checkpoints --baseline-eval-uri` at the final-cohort
baseline.** The ceiling would then encode information from the cohort that
reports the result, which is precisely the leak the final split exists to
prevent. The spec routes this through a dedicated
`config.validation_baseline_eval_uri`.

## Stage Order, And Why

```text
prepare-dataset → push → curate → pull → verify → prepare-split → preflight
→ baseline-validation → baseline-final → train → validate-checkpoints
→ resolve-checkpoint → final-eval → compare → emit-rrd → emit-mcap → publish
```

**Conversion must precede Encord push.** Encord curates media *items*, and
LeRobot v3 packs every episode into one video file per camera (all 50 SO-100
episodes share one MP4 per camera, 49 at a non-zero offset). Episode-level
curation is impossible until the videos are split per episode. This is a hard
ordering constraint, not a preference.

Conversion also publishes the dataset audit, which is the cheapest place to
learn the data cannot train.

## Fail-Closed Behaviour To Preserve

Do not soften any of these when adapting the pipeline:

- **The dataset audit** raises on non-finite state/action values, a declared
  dimension the data does not carry, an empty or miscounted episode, a
  non-increasing timebase, non-positive fps, and a declared camera with no
  video bytes. Constant dimensions and an irregular timebase are advisories,
  because only a human can judge them.
- **Curation attribution** resolves an Encord item to an episode through its
  registered `npa.source_uri`, never a display name. An unattributable item
  fails the run. See `skills/tools/encord/SKILL.md` for why identity is never a
  filename.
- **`prepare_split`** refuses a curation whose roundtrip report did not pass,
  and folds the eligible episode ids into the split hash, so the same seed with
  different curation is a different experiment.
- **Checkpoint selection** requires a candidate to beat the baseline by
  `minimum_relative_improvement`. When none does, the run selects **nothing**
  and still publishes its curve. That is a real outcome; do not make it pick
  the least bad checkpoint.
- **The checkpoint schedule** requires `save_steps` to divide `max_steps` and
  retention to cover every checkpoint saved, because a deleted checkpoint
  cannot be selected.
- **A declared language annotation must have task text.** When
  `meta/modality.json` maps an annotation onto `task_index`, GR00T resolves
  every frame's instruction through `meta/tasks.jsonl`, so empty task text
  trains a language-conditioned policy that never sees its task. Loss still
  falls and the run still reports an improvement, which is why this raises
  instead of warning.

## Reading The Result

Read in this order; each answers a different question.

1. `reports/dataset-audit.json` — is the data usable. Per-dimension ranges
   first; a constant gripper is a recording bug.
2. W&B, or the trainer manifest — is training working. Loss is stochastic, so
   compare windows. Low samples/s means data starvation, not a slow model.
3. `validation/curve.json` — did anything learn. Still falling means train
   longer; flat means more data; rising while train loss falls means
   overfitting and the earlier checkpoint is better.
4. `reports/selected-checkpoint.json` — which step, and why, or an explicit
   nothing-qualified with the closest candidate named.
5. `reports/learning-report.json` — the headline on the final cohort, with
   trivial-predictor floors and a repeat-noise band.

**An improvement smaller than the repeat-noise band is not an improvement.**
Beating the base model is a much lower bar than beating a trivial predictor,
which is what the skill score measures.

**`improved` is not a success rate.** Nothing here closes the loop, so it does
not measure recovery, timing, or contact, and it says nothing about hardware.

## Gotchas

- **`--max-wait-seconds 0` is required.** A cold image pull plus model download
  plus training exceeds the runtime's default one-hour per-wave deadline, and a
  run cancelled at the deadline looks like a failure that is really a timeout.
- **The default curate filter is width-based**, which is intrinsic so it works
  on any folder, but it is an identity and plumbing gate, **not** a quality
  classifier. Point it at app-computed metrics, or set it empty with a long
  `encord_curate_poll_seconds` for a human pass.
- **W&B is on in this spec only.** The shared `workbench.groot.finetune` entry
  declares `--wandb-mode` and `--wandb-project` as
  `omit_flags_when_empty`, so a GR00T workflow that says nothing renders no
  W&B flags. Naming any mode but `disabled` is the switch; there is no bare
  `--wandb` in the argv because a fixed template could never drop one.
- **Conversion re-encodes packed video** (one lossy generation, real CPU time)
  and copies per-episode video byte-for-byte. It is idempotent, so converting
  once with `npa workbench groot convert` and pointing `source_data_uri` at the
  result makes later runs skip the work.
- **Only `prepare-dataset` may read `source_data_uri`.** `modality.json` is a
  GR00T artifact that conversion *creates*, so every later stage must consume
  `prepared_data_uri`. The shared `workflow.groot.prepare_split` catalog entry
  defaults `--source-uri` to the raw input, which is right for
  `groot-1-7-finetune.yaml` and wrong here, so this spec overrides it with a
  per-state `params` overlay. The spec validates and plans either way; the
  mistake only surfaces as `NoSuchKey` minutes into a live submit.
- **LeRobot v3 stores task text as the pandas index** of `meta/tasks.parquet`,
  arriving as `__index_level_0__` rather than a `task` column. Reading only
  `task` silently yields `""` for every v3 dataset.
- **`encord_media_uri` hardcodes `chunk-000`**, correct below 1000 episodes
  since the chunk index is `episode_index // chunks_size`. A larger dataset
  needs a fan-out here.
- **Controller-normalized action error is not metres or radians.** Translate
  through the controller's scales before quoting a physical tolerance.

## Verify

```bash
npa/.venv/bin/npa workbench workflow validate-spec \
  npa/workflows/workbench/npa-workflows/encord-groot-finetune.yaml
npa/.venv/bin/python -m pytest \
  npa/tests/orchestration/npa_workflow/test_encord_groot_finetune_workflow.py \
  npa/tests/workflows/test_groot_dataset_prepare.py \
  npa/tests/workflows/test_groot_checkpoint_selection.py -q
```

The spec tests assert cohort discipline against the **rendered** commands, not
the template: which stages carry `--split-role final`, and that
`validate-checkpoints` reads the validation baseline.

## Status

The spec validates and plans and its stages have unit and guardrail coverage.
**No end-to-end live run has been performed**, so the guide quotes no measured
numbers and the defaults are a starting recipe rather than a tuned one. Do not
add measured claims to the guide without citing a run id.
