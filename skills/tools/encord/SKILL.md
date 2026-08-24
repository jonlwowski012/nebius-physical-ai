---
name: encord
description: Use when pushing Nebius object-store media into the Encord curation SaaS (register-in-place, bytes stay in the bucket), pulling curated Collections/Datasets/Project labels back to S3, or wiring the encord-push / encord-pull / encord-roundtrip-smoke workflows.
---

# Encord (curation SaaS push/pull)

Encord is a third-party labeling/curation platform. The workbench integration
is a **register-in-place** loop: `push` lists an S3 prefix, registers public
objectUrls with Encord through a cloud integration (bytes never leave the
bucket), and links the items into an Encord dataset; a human curates in the
Encord app; `pull` materializes the curated Collection, Dataset, or a Project's
labels back to S3 as media + per-item JSON + a lineage manifest.

## Three-access pattern

Implementation lives in `npa/src/npa/workbench/encord/` (`push.py`, `pull.py`,
with the SaaS seam in `client.py`). The CLI
(`npa/src/npa/cli/workbench/encord.py`) and SDK
(`npa/src/npa/sdk/workbench/encord.py`) are thin wrappers over `run_push` /
`run_pull`. There is no service tier: the UI is Encord's own app. The `encord`
PyPI package is an optional extra (`npa[encord]`), lazy-imported; workflow
stages on the default image install it via `TOOL_REF_PIP_EXTRAS`.

Operator walkthrough (account setup through troubleshooting):
`docs/workbench/encord.md`.

## One-time Encord-side setup (operator)

1. In the Encord app create an **S3-compatible cloud integration** (the
   MinIO/OTC pattern) pointing at the Nebius endpoint
   (`https://storage.<region>.nebius.cloud`) with a dedicated key pair that has
   read access to the media bucket. Prefer "strict client-only access" so Encord
   signs URLs client-side instead of copying media server-side.
2. Give the bucket a read policy for that key pair, and if the Encord viewer
   loads media directly in the browser, a CORS rule allowing `*.encord.com`.
3. Note the integration **title** — the tool accepts it directly
   (`--integration nebius-s3`); no ids need to be copied around.

## Auth

Generate a key pair in the Encord app (public keys) and store the PEM:

```yaml
# ~/.npa/credentials.yaml
tokens:
  ENCORD_SSH_KEY: |
    -----BEGIN OPENSSH PRIVATE KEY-----
    ...
```

For workflow submits, forward the secret by name only. The base64 form is the
multi-line-safe transport (`base64 < key.pem | tr -d '\n'`):

```bash
--secret-env ENCORD_SSH_KEY_B64
```

`ENCORD_SSH_KEY_FILE` (a path) also works for local CLI runs, and
`ENCORD_DOMAIN` selects a non-default (e.g. US) API domain. Verify before
spending time:

```bash
npa workbench health preflight --checks encord
```

## Interfaces

```bash
# Register a prefix in place and link a dataset for annotation.
npa workbench encord push \
  --input-path s3://<bucket>/raw-media/ \
  --integration nebius-s3 \
  --folder my-batch --dataset my-batch \
  --output-path s3://<bucket>/encord/push/

# After curating in the Encord app: materialize the keeper Collection.
npa workbench encord pull \
  --source collection --source-id <collection-uuid-or-name> \
  --output-path s3://<bucket>/encord/pull/

# Or pull a whole dataset / a project's labels (labels export iff project).
npa workbench encord pull --source dataset --source-id my-batch ...
npa workbench encord pull --source project --source-id <project-hash> ...
```

Push has two transfer modes: `--transfer register` (default — bytes stay in the
bucket; Encord references objectUrls through the integration) and
`--transfer upload` (bytes are copied into Encord-hosted storage; no
integration needed). Pull is mode-agnostic: registered items come back as
zero-egress server-side copies, uploaded items stream back through Encord
signed URLs.

Titles resolve wherever they are unique; UUID/hash-shaped values must exist.
`push --folder/--dataset` titles are created when absent; `pull --source-id`
never creates.

## Data contract

- Push receipt: `push_receipt.json` (`npa.encord.push_receipt.v1`) — per-item
  objectUrl/uuid/status, unit counts, folder/dataset lineage. Written **before**
  any failure exit; any unit error fails the command closed (exit 1).
- Pull output under `--output-path`: `media/<item_uuid>__<name>`,
  `items/<item_uuid>.json`, `labels/<label_hash>.json` (project source only),
  and `manifest.json` (`npa.encord.pull_manifest.v1`) with copy/download/failed
  counts. Media registered from this bucket returns as zero-egress
  **server-side copies**; anything else streams through the Encord signed URL.
- Re-push is idempotent (`skip_duplicate_urls`); re-pull overwrites.

## Workflows

- `npa/workflows/workbench/npa-workflows/encord-push.yaml` — production push;
  terminal after the receipt (curation is human-in-the-loop).
- `npa/workflows/workbench/npa-workflows/encord-pull.yaml` — production pull,
  run after curation with the Collection uuid (or dataset/project reference).
- `npa/workflows/workbench/npa-workflows/encord-roundtrip-smoke.yaml` — the
  live e2e test: push fixture media into a fresh `npa-e2e-<run-id>` folder +
  dataset, then pull that dataset back by title in the same run. Submit with
  `--secret-env ENCORD_SSH_KEY_B64 --secret-env AWS_ACCESS_KEY_ID
  --secret-env AWS_SECRET_ACCESS_KEY`. Clean up `npa-e2e-*` folders/datasets in
  Encord afterwards.

toolRefs: workbench.encord.push, workbench.encord.pull

## GPU routing

CPU-only, both verbs. No container image; stages run on the SkyPilot default
image with the `npa[encord]` extra installed at setup.

## Known issues and boundaries

- **MCAP is experimental and currently fails closed.** The pinned
  `encord==0.1.x` upload format has no cloud-registration category for a raw
  `.mcap` (scenes require per-stream SceneBuilder assets). `--media mcap|all`
  discovers `.mcap` keys and records them in the receipt as
  `experimental_error` without sending a guessed schema; the receipt-visible
  failure is the honest v1 boundary pending a live spike.
- Composite Encord items (image groups, DICOM series) expose no single signed
  URL and are recorded as per-item pull errors.
- Licensing/egress: the `encord` SDK wheel is Apache-2.0 (verified from package
  metadata). Push sends object URLs + metadata only; media bytes leave the
  bucket only on the cross-origin download path during pull. Review your Encord
  agreement for any field-of-use terms on exported labels before training on
  them.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/workbench/test_encord.py npa/tests/cli/test_encord_cli.py npa/tests/workflows/test_encord_workflow.py -q
```
