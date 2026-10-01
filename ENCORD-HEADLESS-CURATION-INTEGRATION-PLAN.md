# Headless Encord curation

## Goal

Select media in Encord using its metadata filters and available image quality metrics. Save the selection as a Collection, pull it to S3, and verify the exact items and bytes against the original push. The workflow is `push → curate → pull → verify`.

## Design

`npa workbench encord curate` accepts one or more `metric:min:max` filters. The supported metrics are `width`, `height`, `area`, `aspect-ratio`, `brightness`, `sharpness`, and `file-size`. Validate names and finite bounds before changing Encord state. Keep the preset JSON in the receipt so an operator can see what Encord evaluated.

Use a fresh Collection in the selected folder for each run. Reject a Collection in another folder or one that already has items. The command writes a provisional receipt before creating remote objects, checkpoints the Collection and preset IDs, and stops changing Encord if a checkpoint fails. It records the selected item UUIDs, source push receipt URI, counts, status, and preset cleanup result. An empty selection fails with a receipt and a useful diagnostic. Delete the temporary preset on success or failure, and record any cleanup error.

Expose the command through the CLI, SDK, and `workbench.encord.curate` tool reference. The workflow uses a run-scoped Collection title in both curate and pull, then checks the resolved Collection ID during verification. Its default `width:16:64` filter selects the 32×32 MP4 fixture and excludes the 1×1 PNG fixture.

`verify-roundtrip` accepts an optional curation receipt. With one, it requires final push and curation receipts, a nonempty selection drawn from the push receipt, and an exact UUID match in the pull manifest. It also checks source and destination size and checksum evidence for every selected item. Without a curation receipt, it verifies the full pushed set as before.

Encord does not provide a documented completion handle for preset insertion. The UUIDs in the curation receipt are a snapshot. If the Collection changes after that snapshot, verification fails and the run needs a new selection and receipt.

## Image quality metrics

Width, height, area, and aspect ratio come from intrinsic media metadata. Brightness, sharpness, and file size require quality metrics already computed for the folder in Encord. Encord's [Collections SDK guide](https://docs.encord.com/sdk-documentation/index-sdk/sdk-collections) describes preset evaluation and explicit UUID insertion. Its [embedding guide](https://docs.encord.com/platform-documentation/Curate/embedding-plots) describes starting computation in the app. Neither documents an SDK call that starts the computation. A fresh folder can use intrinsic filters without an app step; the quality filters need that preparation. Zero matches may mean missing metrics or a range that excluded everything, so the error should name both possibilities.

If new folders must use quality criteria without an Encord app step, calculate defined scores in an NPA CPU stage, join them to the push receipt's item UUIDs, and insert those UUIDs with Encord's documented `Collection.add_items` method. Give those scores distinct, versioned NPA metric names because they may not match Encord's scales. Add this path when that unattended requirement exists, or use an Encord computation API if one becomes documented and passes a live check.

## Checks before a live run

- Reject invalid or nonfinite filters before any Encord mutation. Check wrong-folder and populated Collections, delayed indexing, empty selection, preset cleanup failure, and artifact checkpoint failure.
- Verify that a strict subset passes, while a missing, extra, or wrong UUID fails. Keep full roundtrip behavior intact when no curation receipt is supplied.
- Run the Encord tests, workflow and catalog guardrails, spec validation, docs check, and Ruff against the worktree source.
- For live acceptance, confirm Encord can read the registered media, inspect the Collection and final S3 receipts, and prove the workflow selected a strict subset. Local tests and a rendered workflow do not establish a live roundtrip.
