# Research results and saved replay contract

`research_results.schema.json` describes the minimum transport structure. The dependency-free `readers/results_adapter.py` validates counts and joins. The canonical wire ID is `research-results-v1`; `m4-results-v1` remains accepted as a backward-compatible alias with exactly the same validation. Scientific protocol/formula IDs and hash domains are preserved, including `m4-common-ppo-v1`; changing those would change identity or selection semantics.

This package supplies reusable readers, metrics, and one stored-state grid component. It does not supply an HTML dashboard host, HTTP service, production Streamlit bridge, or source-discovery catalog. Streamlit remains the intended application UI.

## Explicit file access

Use `BundleIndex(bundle_directory, releases_root=...)` for a v2 artifact bundle. Repository paths are relative to the bundle directory. Release paths are relative to the explicitly provided releases root, then the indexed `release_id`. That field identifies a bundle folder, not a GitHub release title or tag. Read by artifact ID; original scientific path strings are provenance, never open instructions. Missing release files remain unavailable. Readers use explicit roots without network access or implicit search paths.

`load_bundle_export(bundle, artifact_id)` reads a normalized results export. `load_selected(study, index, bundle=bundle)` validates its selected replay artifacts. An explicitly configured local export can instead use `load_export(path, allowed_root=...)`; legacy trace paths must remain within the same explicit root. Traversal, hidden paths, network paths, root escapes, oversized files, and changed hashes are rejected. The legacy CSV reader retains its stricter final/held-out/OOD path and row exclusions.

## Counts, missingness, and joins

- Models have unique `model_id` and reward/seed/checkpoint dimensions. Separate attempts/configurations belong in separate study exports.
- Datasets have one entry per role. For the frozen protocol, real TRAIN/DEV/LEARNING_VALIDATION counts are 300/100/100; each of the five FINAL roles has 200 scenarios. Explicit other roles use their supplied counts. Labeled synthetic fixtures may use small counts.
- Every outcome and learning row joins a model and dataset. Expected counts equal the dataset count; success ≤ completed ≤ expected. Success + death + timeout must equal completed for outcomes. Learning rows match their model checkpoint. Duplicate rows and conflicting learning/outcome counts are rejected.
- Missing results remain unavailable, not zero-success policies. Partial rows preserve completed, expected, and missing counts. `expected_evaluations` may declare a matrix; otherwise missing final cells are added only for supplied 1M models and supplied FINAL datasets. Missing model identities are never invented.
- Real hashes are mandatory; null hashes are allowed only in explicitly labeled fixtures. Nonfinite values, Boolean counts, negative counts, and invalid transition denominators are rejected. Optional units and exported metrics remain attributable to their exporter; the reader does not invent inference.

The normalized study returned by the adapter uses the neutral `results` key for transport metadata. Its `records` retain individual model seeds, budgets, roles, and missingness. Synthetic fixtures remain explicitly labeled.

## Selected native-recorder replay

Native-recorder selected traces must reference FINAL-role 1M models. Matched real traces require primary slot 0. Matched and representative categories stay distinct; the reader does not select successful examples or rerun selection.

Available entries should contain `trajectory_artifact_id`, `snapshot_artifact_id`, their file-byte SHA256s, exact `episode_id` and `episode_summary`, and exporter verification. `captured_original` attributes capture to the exporter. `replay_exact` additionally requires `summary_match:true` and `events_match:true`; the reader never claims a fresh simulation.

The reader validates native trajectory-recorder v1 and scenario-serialization v2 identity, checksum, map, configuration, model/reward/role/scenario/episode joins, frame order and bounds, and terminal summary consistency. It displays saved positions, HP, rewards, and events. It does not validate physics, load policies, infer enemy behavior, or execute environments.

## Recorded case bindings

`load_recorded_binding(bundle, binding)` reads the separate indexed case/trace format through `replay_bindings.json`. Its inputs name the case, trace, world, derived configuration, checkpoint index, and collection manifest by artifact ID. It preserves original producer job/checkpoint/seed/scenario identities and validates the derived configuration against the original job digest, not against a recomputed digest of relocated fields.

This reader is restricted to the recorded TRAIN/DEV roles. Its payload contains a reset frame from `initial_state`, then one frame after each saved action. WAIT and invalid actions remain distinct recorded events. Outcome priority comes from reconciled recorded success/death/timeout flags, with fatal outcomes taking precedence over a coincident goal event. It reports stored-data join checks only, not physics verification or a final-study claim.

Synthetic snapshot and embedded-world trajectory bytes are shipped only in the optional synthetic release. Their metadata remains in Git. Missing release assets make replay unavailable while results metadata remains readable.
