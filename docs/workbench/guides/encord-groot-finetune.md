# Fine-tune GR00T on your own robot demonstrations

You have robot demonstrations and you want a policy. This guide takes a LeRobot
dataset in your bucket, curates which episodes train, fine-tunes GR00T N1.7, and
tells you whether it helped. One YAML, one submit.

It is written for a robotics engineer who knows their robot and their data but
has not trained a vision-language-action policy before.

> **Validation status: run end to end.** All 17 stages succeeded on run
> `encord-groot-finetune-20260908T222914Z` (L40S, 1 GPU, `max_steps=1000`,
> `save_steps=200`) against `lerobot/svla_so100_pickplace`. Every number below
> is measured from that run's artifacts. The defaults are a working recipe, but
> read [Scale up](#scale-up) before treating them as tuned: 1000 steps is 1.117
> epochs over 36 episodes, and the run's own preflight records
> `statistical_learning_claim: false`.

## What you get, and what you do not

The run ends with one number that matters: how much better the fine-tuned
policy predicts held-out actions than the base model did, measured on episodes
that informed nothing else in the run.

That is real evidence the policy learned something about your task. It is **not**
a success rate, because nothing here closes the loop. A policy can predict
recorded actions well and still fail on the robot, because prediction error does
not measure recovery, timing, or contact. Treat an `improved` outcome as
permission to try the policy on hardware under supervision, not as a result
about hardware.

## Before you start

You need a set-up workbench: a Kubernetes context, a validated `npa-groot`
image in your registry, and credentials in `~/.npa/credentials.yaml` for
Hugging Face, S3, Encord, and optionally Weights & Biases. You also need an
Encord S3-compatible cloud integration that can read your bucket. See
[Encord setup](../encord.md) and, for a fresh machine, the
[platform quickstart](../../quickstart.md).

```bash
npa workbench health preflight --checks s3,hf,encord,wandb --json
npa workbench health access --capability groot --json
```

The `wandb` check warns rather than fails when the key is absent. Training
publishes its evidence to S3 either way; the key only buys live curves.

## Your data

The pipeline takes a **standard LeRobot dataset** in S3, either v3.0 or v2.1.
That is what `lerobot-record` writes, and what Hugging Face hosts. You do not
need to convert anything first.

To try it on a public dataset:

```bash
npa workbench lerobot fetch-dataset \
  --hf-repo lerobot/svla_so100_pickplace \
  --revision 728583b5eaf9e739a7f119e2def466fa1d552402 \
  --output-path s3://<bucket>/datasets/so100-pickplace/
```

That is an SO-100 pick-and-place set: 50 episodes, 19,631 frames, 30 fps, a top
and a wrist camera. A full commit SHA is required, because a branch or tag can
move and a "reproduced" run then would not be. The command leaves a provenance
receipt inside the staged dataset.

For your own recordings, `aws s3 sync ./my-dataset s3://<bucket>/datasets/my-task/`
is enough.

## Submit

```bash
npa workbench workflow submit \
  npa/workflows/workbench/npa-workflows/encord-groot-finetune.yaml \
  --run-id <unique-id> --max-wait-seconds 0 \
  --var bucket=<bucket> \
  --var source_data_uri=s3://<bucket>/datasets/so100-pickplace/ \
  --var encord_integration=<your-encord-integration-title> \
  --image-override workbench.groot=<registry>/npa-groot:<tag> \
  --image-override workflow.groot.prepare_dataset=<registry>/npa-groot:<tag> \
  --image-override workflow.groot.validate_checkpoints=<registry>/npa-groot:<tag> \
  --secret-env HF_TOKEN --secret-env ENCORD_SSH_KEY_B64 \
  --secret-env WANDB_API_KEY
```

`--max-wait-seconds 0` is not optional. The runtime's default one-hour per-wave
deadline does not survive a cold image pull plus a model download plus training,
and a run cancelled at the deadline looks like a failure that is really a
timeout.

**Why three `--image-override` flags and not `--registry`.** `--registry` takes
a registry *prefix* and appends `npa-groot:<pinned-version>` itself, so passing
a full image reference renders the malformed
`<registry>/npa-groot:<tag>/npa-groot:0.1.0`. `--image` accepts a full
reference but applies it to **every** stage, which puts the Encord stages inside
the GR00T image -- where they fail with `No such command 'encord'`, because that
image ships a light `npa workbench` exposing one tool group. `--image-override
TOOL_REF=IMAGE` is prefix-matched and repeatable, so these three flags put the
GR00T image on exactly the six stages that need it and leave the eleven CPU
stages on the default image with the full CLI.

Verify the routing before you submit, rather than discovering it in a pod:

```bash
npa workbench workflow submit <spec> --run-id probe --plan-only --runtime \
  --output-format json <your flags> | python3 -c '
import sys, json, yaml
d = json.load(sys.stdin)
for doc in yaml.safe_load_all(d["skypilot_yaml"]):
    for t in (doc.get("tasks") or [doc]) if isinstance(doc, dict) else []:
        img = (t.get("resources") or {}).get("image_id", "")
        print(f"{t.get(\"name\",\"?\"):24} {img.split(\"/\")[-1] or \"default\"}")'
```

Add `--stage-src` only if your CPU image predates the Encord tools.

Watch it:

```bash
npa workbench workflow status <run-id> --json
npa workbench workflow logs <run-id> --stage train
```

The W&B run link is printed by the training stage. `--var wandb_mode=""` turns
tracking off without changing anything else about the run.

## How long it takes

Measured on the reference run (L40S, warm image cache, 50 episodes):

| Stage | Wall time |
| --- | --- |
| prepare-dataset | ~6 min (27 GB image pull on a cold node adds ~5) |
| push, curate, pull, verify | ~2 min each |
| prepare-split | ~10 min (copies ~600 MB into three cohorts) |
| preflight | seconds |
| baseline-validation | ~6 min (12.6 GB model download plus 173 forwards) |
| baseline-final | ~4 min (reuses the baseline checkpoint) |
| train, 1000 steps | ~20 min |
| validate-checkpoints, 5 candidates | ~10 min (~90 s each) |
| resolve-checkpoint | ~8 min (downloads the selected 12.6 GB checkpoint) |
| final-eval, compare | ~4 min each |
| emit-rrd, emit-mcap, publish | ~5 min total |

About 90 minutes end to end, and roughly 91 GB of artifacts: 75.5 GB of
checkpoints, a 1.3 GB `.rrd`, a 659 MB `.mcap`, and the rest dataset copies.
`prepare-split` and `resolve-checkpoint` are slower than they look because both
move bulk data, not because they are stuck.

## What the run does

```text
prepare-dataset → push → curate → pull → verify → prepare-split → preflight
→ baseline-validation → baseline-final → train → validate-checkpoints
→ resolve-checkpoint → final-eval → compare → emit-rrd → emit-mcap → publish
```

**Conversion runs first, before Encord.** Encord curates media *items*, and
LeRobot v3 packs every episode into a single video file per camera. All 50
SO-100 episodes share one MP4 per camera, 49 of them at a non-zero offset, so
there is no way to curate episode 7 until the videos are split per episode.
Conversion also publishes the dataset audit, which is the cheapest possible
place to discover the data cannot train.

**Three cohorts, and the difference between them is the whole point.**

| Cohort | What it does |
| --- | --- |
| train | the episodes the optimizer sees |
| validation | scores every saved checkpoint and picks one |
| final | read exactly once, by two stages, and that is the headline |

The validation cohort chooses a checkpoint. Because it informed that choice, it
cannot also be the evidence the choice was good, so the headline comparison
happens on the final cohort instead. That is why the run evaluates the base
model twice: once on validation, to give selection a ceiling to beat, and once
on final, as one half of the comparison. Both reuse the same initialized
weights, so the second costs an evaluation rather than a model build.

## Is my data usable?

Read this before you spend GPU time. `reports/dataset-audit.json` reports the
episode inventory, the recorded timebase, per-dimension state and action ranges,
and camera coverage.

The audit **fails the run** on defects no optimizer recovers from:

| Defect | Why it is fatal |
| --- | --- |
| Non-finite state or action values | Training on NaN or inf produces a broken policy |
| A declared dimension the data does not carry | The policy input contract is wrong |
| An empty episode, or a declared length that disagrees with its data file | The metadata cannot be trusted |
| Timestamps that do not increase within an episode | Observations cannot be ordered |
| `fps` of zero or less | The action timebase is undefined |
| A declared camera with no video bytes | Training without required perception |

It **records advisories** for things only you can judge:

- A dimension that never changes, named by joint. If `gripper` never moves,
  nothing can learn to use it, and that is usually a recording bug.
- Inter-frame intervals more than 25% off `1/fps`, which points at the
  recorder's clock rather than the model.

The reading rule: **look at the per-dimension ranges before training.** A
saturated joint or a constant gripper is a data problem, and no amount of
training fixes it.

**What a passing audit looks like.** The reference run's
`reports/dataset-audit.json`:

```json
{ "tasks": ["Pick up the cube and place it in the box."],
  "dataset": { "episodes": 50, "frames": 19631, "fps": 30.0,
               "duration_seconds": 654.366667,
               "episode_frames_min": 326, "episode_frames_max": 575,
               "episode_frames_median": 382 },
  "advisories": [] }
```

Both cameras reported 640x480 with all 50 episodes, and state and action were
6-dimensional with no constant dimension. Seven checks passed, including
`declared language annotation has task text` -- that one matters more than it
looks. `modality.json` maps `human.task_description` onto `task_index`, so GR00T
resolves every frame's instruction through `meta/tasks.jsonl`. Empty task text
there trains a language-conditioned policy that never sees its task, and the
loss still falls, so nothing downstream catches it. The audit now fails closed
on that combination. If your `tasks` array is empty, stop and fix conversion.

The audit does not establish task coverage, operator quality, or label
correctness. For your own recordings, measure coverage across object poses,
lighting, camera placement, operator and session, and include recoveries. Do not
relabel an unsuccessful attempt as a demonstration.

## Is training working?

Live in W&B, and in the trainer manifest afterwards: loss, learning rate,
gradient norm, samples per second, GPU memory.

What to actually conclude:

- **Loss is stochastic.** Compare windows of tens of steps, not single steps. A
  step that goes up means nothing.
- **Gradient norm should be finite** and neither collapsing to zero nor
  exploding.
- **Low samples per second means the GPU is data-starved**, not that the model
  is slow. Raise `dataloader_num_workers` before you raise anything else.
- **A NaN anywhere means stop.** Start a new run; do not resume through it.

**Measured on the reference run** (`npa_groot_finetune_manifest.json`):

| Field | Value |
| --- | --- |
| `initial_loss` | 1.3847 |
| `final_loss` | 0.0634 |
| `robust_early_loss` -> `robust_late_loss` | 1.2348 -> 0.0697 |
| `aggregate_train_loss` | 0.4408 |
| `loss_decreased` / `loss_finite` | `true` / `true` |
| `loss_steps_real` | `true` |

The manifest records `loss_step_source:
trainer_state.log_history.explicit_global_step`, so the curve is the trainer's
own steps rather than an interpolation. Training 1000 steps at global batch 16
on one L40S took about 20 minutes.

**W&B lands in the trainer's project, not the spec's.** `--wandb-project` sets
`WANDB_PROJECT`, but the GR00T trainer calls `wandb.init(project=...)` with its
own name and wins. The reference run logged to project
`finetune-gr00t-n1d7`, and `checkpoints/candidate/wandb_config.json` records the
project and run id it actually used. Read that file rather than assuming your
configured project.

Falling loss is not success. It is the precondition for asking the next
question.

## Did it learn anything?

Three artifacts, in the order you should read them.

**`validation/curve.json`** is the validation error of every saved checkpoint
against the baseline line. Three shapes and what each means:

| Shape | Reading |
| --- | --- |
| Still falling at the last checkpoint | Train longer; raise `max_steps` |
| Flat from early on | More data, or a higher learning rate |
| Falling then rising, while train loss keeps falling | Overfitting; the earlier checkpoint is the better model, and you need more data |

**`reports/selected-checkpoint.json`** says which step was chosen and why.
Lowest validation error wins, a tie goes to the earlier step because the same
error from less training is the cheaper and less overfit model, and a candidate
has to beat the baseline by `minimum_relative_improvement` to be eligible at
all.

If nothing qualifies, the run selects **nothing** and says so, naming the
closest candidate and the ceiling it missed. That is a real outcome, not a
failure to report, and the curve is still published.

**`reports/learning-report.json`** is the headline, on the final cohort:
baseline against selected checkpoint, relative improvement, a skill score
against trivial predictors (zero, mean, last action), the repeat-noise band, and
per-dimension and per-horizon errors, ending in `improved` or `not_improved`.

Two traps worth naming. An improvement smaller than the repeat-noise band is
not an improvement. And beating the base model is a much lower bar than beating
a trivial predictor, which is why the skill score is there.

### The reference run's numbers

The validation curve that chose the checkpoint, with the skill score computed
against that cohort's train-mean predictor (MSE 493.83):

| Candidate | validation MSE | MAE | skill |
| --- | --- | --- | --- |
| base model | 1716.55 | 31.401 | -2.476 |
| checkpoint-200 | 655.86 | 18.237 | -0.328 |
| checkpoint-400 | 370.75 | 12.690 | +0.249 |
| checkpoint-600 | 229.41 | 8.947 | +0.535 |
| checkpoint-800 | 43.54 | 4.503 | +0.912 |
| **checkpoint-1000** | **38.64** | **4.443** | **+0.922** |

Note the shape: a large drop through step 800, then a small one to 1000
(43.54 -> 38.64). That is a curve beginning to flatten, which says more steps
will help less than more episodes would. At step 600 it was still falling
steeply; the same run supports opposite conclusions depending on where you stop,
which is why you read the curve and not the last number.

Selection recorded:

```json
{ "outcome": "selected", "selected_step": 1000, "selected_mse": 38.6393,
  "baseline_mse": 1716.5498, "eligibility_ceiling_mse": 1544.8948,
  "relative_improvement": 0.9775, "candidates_considered": 5 }
```

The headline, on the final cohort of 7 episodes and 2588 samples read exactly
once:

| | base model | fine-tuned | |
| --- | --- | --- | --- |
| action MSE | 1734.28 | **39.88** | -97.7% |
| MAE | 31.531 | **4.471** | 7x lower |
| skill score | -2.902 | **+0.910** | |

And the gate that makes it credible rather than merely large:

```json
{ "gate_passed": true, "gate_failures": [], "regressions": [],
  "repeat_noise_spread": 44.4520,
  "required_absolute_improvement_over_noise": 133.3561,
  "absolute_improvement": 1694.4013 }
```

The improvement is 12.7x the required margin over noise, every one of the 6
action dimensions improved, and each side ran 5 repeats across 4 independent
seeds (830 model forwards each) with `real_model_forward: true`.

**What this does not say.** The report ships its own limitations, and they are
the honest frame for the number above:

```json
"limitations": [
  "This report is offline action-matching evidence, not a robot rollout.",
  "The short optimizer smoke is an operational pipeline validation and does
   not establish statistical learning."
],
"candidate_promoted": false,
"closed_loop": false
```

The policy predicts held-out expert actions far better than the base model and
far better than any trivial predictor, on episodes it never saw. That is real
evidence of learning. It is not a task success rate, it says nothing about
recovery, timing or contact, and the run deliberately does not promote the
checkpoint.

## When it says `not_improved`

Work in this order:

1. **Re-read the audit.** Constant dimensions and an irregular timebase explain
   more failures than hyperparameters do.
2. **Check the training signals** above. A flat loss with only one episode being
   sampled is a data-consumption bug, not a model problem.
3. **Train longer.** The default 1000 steps is 1.117 epochs over 36 episodes --
   barely one pass. 2,000 to 10,000 is a normal range once the pipeline is
   proven. Remember `save_steps` must divide `max_steps` and
   `save_total_limit` must cover every checkpoint saved, or preflight rejects
   the run.
4. **Add data**, especially coverage of the situations it fails in.
5. **Then** touch the learning rate.

Every change is a new run id. Compare `learning-report.json` across runs with
the same `split_seed`, so the cohorts are identical and the comparison means
something.

## Scale up

Preserve the batch invariant when you change GPUs:

```text
global_batch_size = gpu_count × per_device_batch_size × gradient_accumulation_steps
```

The preflight rejects a violation before any GPU is provisioned. It also
requires `save_steps` to divide `max_steps`, and retention to cover every
checkpoint the schedule saves, because a deleted checkpoint cannot be selected.

`--var gpu_type=B200` and `--var gpu_type=L40S` both work; check your image and
model against the GPU first, since changing the accelerator does not prove its
kernels are compatible. Plan S3 capacity from your first measured checkpoint
size, and note that optimizer state can exceed the weights.

**Check the node's disk before you choose the GPU.** This is the sizing trap
that bites, because it has nothing to do with GPU memory. The GR00T image is
about 27 GB compressed and unpacks to roughly 55 to 80 GB. Add the base model
and its runtime dependency, then the checkpoints the schedule retains: at five
retained weights-only checkpoints for a 3B model, budget on the order of

```text
55-80 GB (image) + ~10 GB (models) + 5 x ~13 GB (checkpoints) = 130-155 GB
```

A node advertising around 118 GB of ephemeral storage cannot hold that, and a
node advertising around 238 GB can. Check yours before submitting:

```bash
kubectl get nodes -o custom-columns=\
'NAME:.metadata.name,GPU:.metadata.labels.nebius\.com/gpu-name,DISK:.status.allocatable.ephemeral-storage'
```

If the GPU you want is short on disk, the levers are a lower
`save_total_limit` (at the cost of selection candidates), the
[durable model cache](../model-weight-cache.md) on a mounted volume, or simply
picking the node with more room. Training that dies on `no space left on
device` after an hour looks like a training failure and is not one.

Two costs to know about. Conversion **re-encodes** video when the source packs
episodes together, which costs CPU time and one lossy generation; a dataset
already recorded per-episode is copied byte-for-byte instead. And conversion is
idempotent, so you can convert once with `npa workbench groot convert` and point
`source_data_uri` at the result to make every later run skip that work.

## Troubleshooting

| Symptom | Look at | Do |
| --- | --- | --- |
| Run cancelled around the one-hour mark | The per-wave deadline | Resubmit with `--max-wait-seconds 0` |
| Conversion fails naming an episode or feature | `reports/dataset-audit.json`, and the message itself | Fix the recording; the message names the offending episode |
| Curation selects nothing | The curate receipt | The width filter is an identity gate; point it at app-computed metrics or curate by hand |
| Verify fails on checksum | The roundtrip report | Do not proceed; the media Encord returned is not what you pushed |
| Split has too few episodes | `selection.json` and the cohort counts | Curation excluded too many; the three cohorts must fit inside what survived |
| Selection chose nothing | `validation/curve.json` | Follow the `not_improved` order above |
| GPU out of memory | `per_device_batch_size` | Lower it and raise accumulation to hold the effective batch |
| GPU mostly idle | Samples per second | Raise `dataloader_num_workers`; profile before adding GPUs |
| Training loss goes NaN | Input finiteness, learning rate | Stop; start a new run rather than resuming |

### Resuming, exactly

Repeat the submit with `--resume-run <original-run-id>` in place of `--run-id`
(the two are mutually exclusive). Completed waves replay from the ledger in
seconds -- you will see `replayed from ledger (job N)` per stage -- so verified
curation, saved checkpoints and finished evaluations are reused rather than
recomputed.

**Pass the original run id, not the id the error message suggests.** A failed
wave's `operator_remedy` names a wave-scoped id like
`<run-id>-07-preflight`. `--resume-run` validates its argument *as a run id*, so
that value becomes a brand-new run with no ledger history and the pipeline
restarts from stage 1. This cost a full 35-minute replay before we noticed.

Two failure modes need an explicit authorization flag, because the runtime will
not invent an attempt it cannot tie to a known job:

| Recorded state | Meaning | Flag |
| --- | --- | --- |
| `resume_block_terminal_or_legacy_absence` with `job_id: ""` | The launch died before SkyPilot assigned an id, so there is no identity to reconcile | Unrecoverable; start a fresh run |
| Wave failed but the job actually succeeded | An unconfirmable launch that did complete | `--adopt-absent-in-flight-outputs` |
| Wave terminal-failed and the job is genuinely gone | Safe to re-attempt | `--retries N`, or `--retry-absent-in-flight` |

The first row is worth internalizing: if a wave never got a job id, that run is
finished no matter what you fix afterwards. We lost a run to it.

### Kubernetes credential traps

Two thirds of the interruptions in the reference session were credentials, not
the pipeline. All three symptoms look like network faults and are not.

| Symptom | Real cause | Fix |
| --- | --- | --- |
| `managed-job launch indeterminate ... transport: authentication handshake failed: context deadline exceeded`, naming `.nebius/bin/nebius ... exit code 60` | The kubeconfig authenticates through an exec plugin that shells out per call, and it intermittently times out | Replace the `exec` user in `~/.kube/config` with an inline bearer token |
| Launches fail while `kubectl` works fine | The SkyPilot API server caches the credential it started with; a long-lived server holds an expired token | `sky api stop && sky api start` |
| `You must be logged in to the server (Unauthorized)` mid-run | The inline token expired | Mint a new one and restart the API server, or it keeps serving the old one |

`KUBECONFIG` does not help here. SkyPilot 0.12.2 calls
`list_kube_config_contexts()` with no `config_file`, so the Kubernetes client
reads `~/.kube/config` and ignores the environment variable -- npa's
`sky_environment()` documents this. The file itself has to carry the credential.

Mint a token with:

```bash
nebius mk8s v1 cluster get-token --profile <profile> --format json
```

Keep the kubeconfig **user name unchanged** when you edit it. SkyPilot derives
cluster ownership from that name, so renaming it orphans the existing jobs
controller with
`ClusterOwnerIdentityMismatchError`.

**The durable direction is a service-account token**, but the service account
SkyPilot ships is **not sufficient as-is**, and this is worth getting right
before you try it.

```bash
kubectl --context <ctx> create token skypilot-service-account \
  -n default --duration=2160h        # 90 days; this cluster allows up to 8760h
```

That token authenticates, passes SkyPilot's ownership check (ownership derives
from the kubeconfig *user entry name*, not the token subject, so keep the entry
name unchanged), and serves `sky jobs queue` correctly. It then fails
provisioning:

```
pods is forbidden: User "system:serviceaccount:default:skypilot-service-account"
cannot list resource "pods" in API group "" at the cluster scope
```

`skypilot-service-account` is bound namespace-scoped in `default`, so
`list pods -n default` is allowed and `list pods --all-namespaces` is not --
and SkyPilot's GPU discovery lists cluster-wide. The failure surfaces as
`accelerator readiness failed: Timed out after 600s waiting for SkyPilot to
discover a compatible GPU`, which reads like a capacity problem and is a
permissions problem.

Before adopting a service account, grant it cluster-scoped read on the
resources SkyPilot discovers against (`pods`, `nodes`) and **verify with
`sky gpus list --infra k8s`, not just `kubectl` and `sky jobs queue`.** A
credential can pass every check you thought to run and still fail the one the
launcher makes.

Until that binding exists, keep the user-account token and rotate it:

```bash
nebius mk8s v1 cluster get-token --profile <profile> --format json
```

Either way, restart the API server afterwards (`sky api stop && sky api
start`) -- it serves the credential it booted with. Prefer 90 days over a year
for any long-lived token in a plaintext kubeconfig.

## Curating by hand

The default filter is width-based. It is intrinsic, so it works on any folder
without precomputed metrics, but it is an **identity and plumbing gate, not a
quality classifier**. Two ways to make curation mean something:

- Compute quality metrics in the Encord app, then
  `--var encord_curate_filters="brightness:0.2:0.8,sharpness:0.3:1"`.
- Curate by hand: `--var encord_curate_filters=""` with
  `--var encord_curate_poll_seconds=86400`, then populate the run-scoped
  Collection in the Encord app. The run waits, then continues.

Either way `selection.json` records which episodes trained and which Encord
items they came from, and the selection enters the split hash, so the same seed
with different curation is correctly a different experiment.

## Adapting to your robot

Document the joint ordering, controller API, action space, reference frames,
rotation convention, gripper behaviour, camera calibration, synchronisation,
control rate, and latency. Then map them onto `camera_key`, `state_key` and
`action_key`, and let the audit tell you whether the result is coherent.

`NEW_EMBODIMENT` means GR00T learns your action space from the data rather than
assuming a registered robot's convention. Controller-normalized action error is
not metres or radians; translate through your controller's scales before quoting
a physical tolerance.

## Where everything lands

```text
s3://<bucket>/encord-groot-finetune/<run-id>/
  data/prepared/                    per-episode GR00T dataset
  reports/dataset-audit.json        is my data usable
  encord/{push,curate,pull,verify}/ curation lineage and the checksum proof
  selection.json                    which episodes trained, with Encord identities
  data/{train,validation,final}/    the three cohorts
  checkpoints/baseline/             initialized base weights
  checkpoints/candidate/            checkpoint-100 … checkpoint-500
  validation/baseline/              the selection ceiling
  validation/checkpoint-<N>/        every candidate's validation error
  validation/curve.json             the curve you read
  reports/selected-checkpoint.json  which step, and why
  offline/{baseline,trained}/       the final-cohort comparison
  reports/learning-report.json      improved or not_improved
  reports/*.rrd, *.mcap             open these in the NPA agent viewer
  reports/publish-manifest.json     hashes for everything above
```

## Related

- [GR00T N1.7 operational training pipeline](../cookbooks/groot-1-7-training.md)
  — the shorter plumbing validation this guide's stages grew out of.
- [Encord curation](../encord.md) — credentials, integrations, and the curation
  verbs on their own.
- [Physical AI Data Factory](physical-ai-data-factory.md) — augmentation and
  curation for perception data rather than policy training.
