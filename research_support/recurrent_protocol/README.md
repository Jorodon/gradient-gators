# Frozen recurrent protocol source reference

This folder provides a small source-verification utility and the recorded software inventory for `recurrent_ppo_frozen_protocol_v1`. The scientific implementation is supplied in the separate source release asset; it is not duplicated into the application's current `src/` tree.

## Files

- `source_catalog.json` pins the public source manifest and the original first-party file identities. It records publication transformations without claiming the published bytes are all identical to the executed bytes.
- `recorded_runtime.json` records the observed Python and package versions. It is evidence of the executed environment, not a tested installation lock or an installer.
- `verify_source.py` checks the extracted source asset against the pinned catalog and parses Python syntax without importing scientific modules.
- `tests/test_source_verifier.py` checks rejected corruption, identity changes, path traversal and accidental execution risks using small synthetic files.

## Verify the source download

Download the study's source release asset and verify its archive checksum against the published checksum list. Extract `recurrent_ppo_frozen_protocol_v1_source.zip` to a location of your choice, preserving its top-level folder. From the repository root:

```powershell
$SourceRoot = Read-Host 'Extracted recurrent_ppo_frozen_protocol_v1_source folder'
python -B research_support/recurrent_protocol/verify_source.py --source-root "$SourceRoot"
```

This command reads files and reports JSON. It does not import the trainer, load a checkpoint, step an environment or write into the source asset. Python 3.10 or newer is required for the verifier; scientific code syntax must also be supported by the chosen interpreter. The experiment's recorded Python version is listed separately in `recorded_runtime.json`.

The source archive is an auditable implementation reference. Its scientific module paths are preserved, including `engine.recurrent_complete_episode` and `src.training.geometry_features`. Some archived files have declared path-only publication transformations; placeholders are archival labels and must not be treated as working launch paths. Do not add this archive to the running dashboard's import path or overwrite the application's current source with it.

## Reproduction status

The frozen study completed, and its results and saved replay were verified by the producer. Publication checks verify source identities and saved data, not fresh-machine training reproduction. This package does not yet supply a qualified portable campaign launcher or a verified dependency-installation recipe. Do not run an archived controller as a substitute.

Issue #115 remains responsible for a portable setup and separately documented bounded reproduction: resolve supported dependencies, isolate frozen source imports, supply data/output paths explicitly, verify model loading and environment behavior, and record another reviewer's result. No full-study retraining is required merely to verify publication integrity. Preserve the original protocol when qualifying that path.

For saved-results browsing, use the existing dashboard's results/worlds bundles and one-folder loading workflow; neither this source asset nor model weights are required.
