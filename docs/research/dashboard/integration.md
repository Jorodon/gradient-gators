# Integration, evidence and downloads

This reference contains measured development evidence and labelled synthetic examples. Integration uses saved records without running training or evaluations. Streamlit integration and browser acceptance remain team work.

## One download folder

Obtain the relevant GitHub release links from **@jorodons**. Download the recorded results for full numerical acceptance, measured worlds for spatial replay, and synthetic examples for developer/player checks. The internal bundle IDs below are not GitHub release titles, tags or published asset names.

Create `docs/research/dashboard/downloaded_releases/` at the repository root, then extract the downloads into it. Preserve each archive's bundled top-level folder and all manifest IDs; do not flatten or rename the bundle contents. The supplied `.gitignore` excludes this directory. An external directory also works when explicitly supplied to the readers.

The planned Streamlit integration reuses its existing artifact-directory input: select the common extraction directory, then choose the detected result bundle if needed. Selecting the bundle directory directly is also supported by the planned adapter. Existing CSV input remains valid. A single selection determines results and sibling world locations; users do not need to configure the two developer-test environment variables below. Until DASH-01/02/04 are implemented, the current upstream dashboard only understands its existing CSV inputs.

## Repository and release boundaries

The repository keeps reusable code, schemas, metadata examples, a small authentic case/trace test fixture and these two guides. It contains **no playable maps, terrain snapshots or trajectories embedding a synthetic world**. Dataset membership lists and scenario hashes are metadata, not maps.

The full recorded corpus and worlds are distributed separately from Git:

| Internal bundle ID | Contents | Use |
|---|---|---|
| `dashboard_recorded_results_v1` | Five-reward reference summaries, curves, 3,501 cases, 36 traces, configurations, dataset manifests and answer key | Full numerical/identity acceptance evidence |
| `research_worlds_v1` | 400 unique TRAIN/DEV world snapshots | Exact geometry for recorded replay; WAIT reuses an existing DEV world |
| `dashboard_synthetic_examples_v1` | Labelled synthetic snapshot and embedded-world trajectory | Player/parser checks, never scientific evidence |

Each bundle folder has a `release_manifest.json` and checksum. Keep the manifest-defined folder IDs when extracting; readers use them to resolve dependencies. Reuse existing world downloads only when exact identities and hashes match. No model weights, full training folders, private path receipts or final holdouts are included.

An example local extraction layout is:

```text
docs/research/dashboard/downloaded_releases/    # Git-ignored
  dashboard_recorded_results_v1/
    bundle_manifest.json
    expected_results.json
    experiments/...
    datasets/...
    ...release and binding metadata...
  research_worlds_v1/
    release_manifest.json
    worlds/...
  dashboard_synthetic_examples_v1/
    release_manifest.json
    payload/...
```

`BundleIndex(bundle_directory, releases_root=...)` receives both roots explicitly. An entry with legacy `storage: repository` resolves relative to **the selected bundle directory**, even when that bundle was extracted from a release. This field does not assert that its file must be committed to Git. A `storage: release` entry resolves from `releases_root/release_id/relative_path`; `release_id` is the manifest's internal bundle ID. Indexed reads check length and SHA-256 and reject escaping/ambiguous paths. Readers use explicit local roots and do not download or search for missing files automatically.

Without the full results download, the small fixture supports limited regression checks only; it cannot reproduce the five-reward totals. With results but no worlds, numerical summaries, curves and case/event records remain readable; affected spatial replay reports unavailable. Never substitute a synthetic world for missing measured terrain.

## Authentic reference cohort

`entropy_0_01_experiment` contains the five historical `fixed_e001_*` jobs from SCREEN-01: model seed **85235071**, schedule seed **1852481053**, original rewards, fixed collection, entropy **0.01**, budget **500,000 training steps**. These are five reward conditions from a larger screen, not five seeds of one reward and not the complete experiment archive.

| Reward | DEV successes at 0 / 100k / 250k / 500k (each /100) | TRAIN endpoint /300 | DEV normalized AULC |
|---|---|---:|---:|
| Sparse | 0 / 17 / 26 / 39 | 193 | 24.40% |
| Efficiency | 0 / 4 / 12 / 26 | 151 | 12.30% |
| Safety | 0 / 0 / 9 / 21 | 85 | 8.85% |
| Balanced | 0 / 0 / 7 / 19 | 107 | 7.55% |
| Progress | 0 / 5 / 10 / 23 | 118 | 11.00% |

The cohort has **3,500 cases, 25 summary panels, 20 DEV curve points and 35 endpoint traces**. AULC is trapezoidal area of success fraction against actual checkpoint steps, divided by the 500k horizon (multiply by 100 for percentages above). This four-point descriptive curve is separate from the supported final research protocols.

The separately labelled `recurrent_ppo_experiment/wait_example` adds **one case and one trace**, excluded from baseline totals. Source job `anchor_confirmation_s2_recurrent_progress`, model seed **3380075943**, 1M checkpoint, scenario `scn_0c28e5e21850fded08c4fa74388c211d42ccd4818cf2b13ff5dd233395e2dcde`: WAIT at step 3, success after 13 steps, final HP 90.

The small fixture under `research_support/dashboard_reference/tests/fixtures/recorded_examples/` retains selected genuine evidence and required join metadata. Its selection metadata defines its limited scope. Full-cohort summaries belong to the release, not to that subset's empirical coverage.

## What the files mean

| File/record | Meaning and required interpretation |
|---|---|
| `bundle_manifest.json` | Packing list: stable artifact IDs, relative locations, lengths, byte hashes and source references; excludes itself |
| `expected_results.json` | Full-corpus importer answer key with denominators and scope; neither a new experiment nor a policy target |
| `publication_verification.json` | Evidence-preparation audit; not proof of publication or Streamlit acceptance |
| `replay_bindings.json` | Exact case/trace/configuration/checkpoint/dataset/world joins; filenames alone are insufficient |
| `configuration.json` | Derived portable configuration; original machine directories replaced by dataset IDs |
| Checkpoint metadata | Nominal, consumed and optimized steps plus recorded model hashes; absent model ZIPs cannot be reverified |
| Case | `identity`, `metrics`, `sha256`, `status`; join reward/configuration/checkpoint/collection/scenario/evaluation mode |
| Trace | Initial state, ordered post-action frames and internal content seal |
| Result | Scoped counts/sums/means and case-hash inventory; existence alone does not prove full coverage |
| Dataset manifest | Expected membership and collection/scenario/checksum identity |
| World snapshot | Terrain and initial world, distributed separately |

All original raw cases and traces retain their bytes and internal seals. In the six portable configurations, `source_job_sha256` identifies the original canonical job **content**, `source_file_sha256` fingerprints original job **file bytes**, and the bundle entry fingerprints the **adapted file**. They are different hash domains. Do not recalculate historical case identities from an adapted configuration. Similarly, a trace's internal seal excludes its own seal field and differs from its whole-file hash. Historical `trace_file` values remain producer-relative provenance, not instructions to search another machine.

Known metric mappings: `length → episode_length`, `damage → damage_taken`, `illegal_moves → invalid_moves`, `wait → wait_actions`. Modern damage is HP; older invalid-move/wait fields may be 0–100 percentages. Preserve units and leave unsupported fields unavailable. The baseline aggregates do not provide separate all-case fall/hazard/enemy damage totals. Training and evaluation seeds have different roles.

## Replay semantics and gaps

The 35 baseline traces use the same five DEV and two TRAIN scenarios across rewards; 12 include invalid moves and none includes WAIT. This selected set does not estimate event prevalence. The separate WAIT example fills that rendering case without changing baseline summaries.

A bounded audit of 1,898 saved traces found no authentic death-on-goal example. A labelled synthetic fixture can check rendering precedence; it does not fill the measured-evidence gap.

Frame 0 is reset state. Frame N is state after action N, including enemy movement and damage. WAIT/invalid moves can leave position unchanged while time, enemies and HP change. Success requires reaching the goal alive; death on the goal is failure. Missing action probabilities, critic values, LSTM state or reward components cannot be reconstructed as measured evidence.

## Other supported input families

| Family | Handling |
|---|---|
| Repository CSV | Explicit episode/evaluation/checkpoint scope and manifest; parse booleans and join full identities; no one-summary-file precedence |
| Structured export | Historical `research-results-v1`/`m4-results-v1`; recurrent `research-results-v2` with explicit protocol; DEV remains DEV |
| Manual studies | Contract/cells/checkpoint/evaluation CSVs and optional gzip traces; preserve seed roles, units, failed-attempt lineage and coverage |
| Older entropy/terrain | JSONL reset/step traces with matching summary/configuration/checkpoint/mode; distinguish deterministic and sampled evaluation |
| Historical aggregates | Preserve limitations; unavailable curves/replays remain unavailable |

Readers are format-specific. The initial implementation supports the supplied full recorded corpus, the existing native CSV path and the already supplied structured/normalized readers. Other historical formats are incremental follow-up work when an actual input requires them; unsupported input must be identified clearly. The bundle resolver and recorded-replay helper do not themselves normalize all screen summaries or complete Streamlit integration. Consult the [schema contracts](../../../research_support/dashboard_reference/schemas/results_contract.md) and [normalized-record contract](../../../research_support/dashboard_reference/schemas/normalized_studies_contract.md).

## Supported final protocols

The historical protocol requires genuine 0/250k/500k/750k/1M observations. The recurrent protocol requires 0/100k/250k/500k/1M/1.5M/2M, using actual consumed steps for learning area and threshold coordinates. Each panel contains 100 learning-validation cases. Sustained 80% is the second of two qualifying positive scheduled checkpoints. A complete horizon without attainment is right-censored at its declared budget; missing measurements are incomplete. Do not apply either contract to the four-point pilot curve. See [recurrent protocol](recurrent_protocol.md) for the versioned reader, exporter and native statistic names.

Final roles are ID, hazard, obstacle, pursuit and combined OOD, with 200 scenarios each. [replay_selection.json](replay_selection.json) documents the historical 1M selection. The recurrent exporter uses the completed producer's `selected_traces.json` at 2M: two outcome-independent hash-selected scenarios per condition, primary slot0, all five rewards (50 matched selections), plus separately labeled representative selections. The original hash namespace and collection identities are retained. Reference code contains no measured final outcomes or final worlds.

## Conclusions and selected-result downloads

The Streamlit evidence section reads `docs/research/studies/<study_id>/conclusions.json`.
This is researcher-authored interpretation, separate from the measured results. Its
`binding` must match the loaded `study_id`, `protocol_id`, verified `release_tag`
and `export_sha256`. Publication details are attached only when the published
manifest matches the loaded result-file digest. Moving the downloads directory
does not change that evidence identity.

The conclusions file uses `schema_version: 1`, a `status` of `draft`, `approved`
or `superseded`, `author`, `reviewer`, ISO `reviewed_at` and `evidence_cutoff`
dates, and a `claims` array. Each claim has a stable `id`, an explicit `scope`,
the `claim` text, a `limitations` array and an `evidence` array referencing
`record_id`, `model_id`, `comparison_id` or `configuration_id` values supplied
by that study. The evidence cutoff is the researcher's declared evidence boundary;
it is not inferred from the publication preparation date. The export hash pins the
exact evidence bytes. Inspect the source panel for available identities.

Each conclusions file records its review status explicitly. Review the
interpretation, limitations and supporting identities against the published
results, and set `status` to `approved` only after review.
Missing, draft, superseded, malformed or mismatched files leave conclusions
unavailable without preventing valid numerical results from loading. Refresh or
rerun the page after editing. A newer result export requires its own reviewed
binding; an older conclusion must not be silently transferred to it.

Conclusions are labelled study context and retain their authored scope when
filters change. The source panel separates that study context from the current
selection. CSV downloads contain the displayed portable table; JSON downloads
include the same rows, filter values, displayed summary metrics and available
provenance. Missing values remain missing. Aggregate rows describe evaluation
panels, not individual episodes. Private absolute paths and raw internal metadata
are excluded; relative artifact references and original/published hashes remain
distinct. An unavailable source identity is not guessed for older inputs.
An empty filter list retains the current dashboard convention of no restriction.
The downloaded JSON is a selection summary, not a replacement release bundle or
an input to the strict research-results importer.

Other views can reuse `selected_table` and `build_export` in
`src/evaluation/dashboard_evidence.py`. Pass the exact displayed table and its
declared scope. The helpers serialize values; they do not calculate scientific
metrics or filter a published statistical comparison into a new test. Published
paired comparisons and learning metrics remain the responsibility of their
respective views. Do not attach unfiltered study statistics to a filtered subset
as though those statistics were calculated for that subset.

Results-only downloads support this feature; worlds and model weights are not
required. Extract releases under the common directory described above and keep
downloaded contents out of Git. The focused checks are:

```powershell
python -B -m pytest -q tests/test_dashboard_evidence.py tests/test_dashboard_app.py tests/test_dashboard_data.py
```

## Offline validation

From the repository root, basic checks need no RL libraries, worlds or full dataset:

```powershell
python -B -m unittest discover -s research_support/dashboard_reference/tests -p 'test_*.py'
```

For full reference and replay checks after extracting the separate downloads:

```powershell
$env:DASHBOARD_RELEASES_ROOT = (Resolve-Path 'docs/research/dashboard/downloaded_releases').Path
$env:DASHBOARD_RESULTS_BUNDLE = Join-Path $env:DASHBOARD_RELEASES_ROOT 'dashboard_recorded_results_v1'
python -B -m unittest discover -s research_support/dashboard_reference/tests -p 'test_*.py'
node research_support/dashboard_reference/viewer/test_grid_player.js
```

The small fixture is checked in both modes. `DASHBOARD_RESULTS_BUNDLE` adds full-corpus numerical/coverage checks; full replay checks also require `DASHBOARD_RELEASES_ROOT`. Release-dependent tests explicitly skip without their assets; a skip is not a passed replay check. The Node check uses an in-memory DOM and external synthetic release, not a real browser. [test_results.md](../../../research_support/dashboard_reference/test_results.md) records performed checks and their limits.

Acceptance must preserve hashes, expected/completed counts, metric definitions, values and unsupported capabilities. Export/reimport must preserve identities and values. Browser checks must separately establish filter/card/table/chart/replay agreement. These checks do not rerun simulator physics or certify final research readiness.
