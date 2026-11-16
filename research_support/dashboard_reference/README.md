# Dashboard reference code

Reusable readers, metrics, replay rendering and offline checks for **Streamlit**, which remains the final UI. These parts still require integration into the team's dashboard. Nothing here trains a policy, loads model weights or steps an environment.

Read the [architecture overview](../../docs/research/dashboard/README.md) and [integration/evidence guide](../../docs/research/dashboard/integration.md) for scientific rules, reference results, release layout and validation commands. Schema contracts are in `schemas/`; older input support is in `readers/legacy_csv.py`.

## Folders and files

Readers load and validate evidence, schemas define its structure, metrics calculate agreed measurements, and the player displays saved episodes in Streamlit.

| Folder | Purpose |
|---|---|
| `readers/` | Translate known formats and check identity; filenames alone cannot reliably join models, checkpoints and scenarios |
| `metrics/` | Shared scientific calculations so charts and tables use the same definitions |
| `schemas/` | Machine-readable JSON shapes and written field/coverage contracts |
| `viewer/` | Reusable grid player, not a second dashboard application |
| `examples/` | Tiny labelled synthetic metadata examples, not measured results; world references require a separate download |
| `tests/` | Offline checks plus a small authentic fixture under `fixtures/recorded_examples/` |

All Python/JavaScript files are listed below. “Reusable” means suitable for integration, not already connected to Streamlit.

| File | What it does | Role |
|---|---|---|
| `readers/bundle_paths.py` | Resolve indexed artifact IDs within explicit local bundle/release roots; verify lengths/hashes and reject ambiguous/escaping paths | Reusable file access |
| `readers/results_adapter.py` | Validate structured research exports, datasets/models/metrics and final-protocol restrictions; retain legacy schema compatibility | Reusable reader for this format, not a universal importer |
| `protocols.py` | Define the supported historical and recurrent scientific protocols, grids, endpoints and training-counter checks | Shared protocol selection |
| `readers/recurrent_export.py` | Convert a completed recurrent producer report into a portable results bundle, optionally with separate saved replay/world assets | Offline export; preserves producer statistics |
| `readers/recurrent_replay.py` | Verify recurrent case/trace/world joins and normalize recorded frames for the existing player | Saved-data replay; no inference |
| `readers/replay.py` | Check exact case/world/configuration/checkpoint/trace joins and prepare saved display frames | Reusable replay adapter |
| `readers/legacy_csv.py` | Read supported older CSV/manual-study records from an explicitly chosen folder | Reusable format-specific reader |
| `readers/normalized_model.py` | Validate common normalized study records and numeric fields | Reusable validation |
| `metrics/research_metrics.py` | Aggregates, learning area, sustained-success time, paired comparisons, Holm correction, robustness and fixed replay selection | Reusable calculations; frozen final rules are not automatically appropriate for older studies |
| `viewer/grid_player.js` | Draw supplied geometry and saved frames with step/play controls and cleanup | Reusable rendering; no file fetching, actions or environment execution |
| `viewer/test_grid_player.js` | Check player behavior, independent mounts and cleanup with the external synthetic release | Developer test; in-memory DOM, not Streamlit/browser acceptance |
| `tests/test_bundle_paths.py` | Check portable lookup, byte verification, invalid paths and absent releases | Developer test |
| `tests/test_results_reference.py` | Check structured-result counts and joins, including inconsistent/rejected evidence | Developer test |
| `tests/test_research_metrics.py` | Check calculations and frozen coverage/selection rules | Developer test |
| `tests/test_portable_outputs.py` | Check relative/logical output references and legacy/final-role restrictions | Developer test |
| `tests/test_recorded_bindings.py` | Check small authentic fixture; optionally add full-corpus numeric coverage and released-world replay verification | Developer test; fixture coverage is not full-cohort coverage |

`viewer/grid_player.css` styles the player. `schemas/*.schema.json` specifies machine-checkable structure; neighboring contract Markdown explains meaning. `examples/research_results.json` and `normalized_studies.json` are synthetic metadata examples. `examples/bundle_manifest.json` links them to separately released synthetic world/trajectory assets.

`test_results.md` records performed checks. `source_provenance.json` fingerprints reusable code/docs and identifies the source revision; evidence bundles have separate manifests. The small fixture's selection metadata explains its limited scope; it is not an alternative five-reward answer key. The code-root `.gitattributes` preserves indexed bytes against line-ending conversion. The documentation `.gitignore` excludes downloaded release assets. The `test_recurrent_*.py` modules cover the versioned protocol, metrics, native export and saved replay using labeled synthetic inputs.

These folders use Python namespace packages. Run checks/imports with the repository root on Python's import path.

## Portable API

Obtain the relevant release links from **@jorodons**. Create `docs/research/dashboard/downloaded_releases/` at the repository root and extract downloads there, preserving bundled top-level folders and manifest IDs. The example uses the internal bundle ID `dashboard_recorded_results_v1`; it does not specify a GitHub release title, tag or published asset name.

```python
from pathlib import Path
from research_support.dashboard_reference.readers.bundle_paths import BundleIndex

downloads = Path('docs/research/dashboard/downloaded_releases')
bundle = BundleIndex(
    downloads / 'dashboard_recorded_results_v1',
    releases_root=downloads,
)
# Verify bundle-local evidence; world downloads are not required for this check.
bundle.verify_all(include_releases=False)
# artifact_id comes from bundle_manifest.json or replay_bindings.json:
# record = bundle.read_json(artifact_id)
```

The caller supplies explicit local roots. `read_json` accepts an artifact ID. `artifact_for_source` resolves logical producer ID + relative artifact path + optional source hash. Legacy `storage: repository` means relative to the selected bundle directory, including a bundle extracted from a release; it does not demand Git storage. Missing worlds raise an explicit unavailable-asset error. Recorded hashes of absent model archives are provenance, not newly verified model bytes.

Historical structured exports use `research-results-v1`; immutable `m4-results-v1` exports remain accepted. The recurrent protocol explicitly uses `research-results-v2` with `recurrent-complete-900-2m-v1`. Direct readers require an explicit `allowed_root`. Bundle-aware exports use `load_bundle_export`; structured replay uses `load_selected(..., bundle=bundle)` for either supported recorder; authentic reference replay uses `load_recorded_binding(bundle, binding)`. Screen-summary normalization and Streamlit integration remain issue work. [Recurrent protocol/export instructions](../../docs/research/dashboard/recurrent_protocol.md) describe complete-campaign conversion and metric field mappings.

```javascript
const player = window.GatorRecordedGrid.mount(containerElement, validatedReplayPayload);
// On component replacement/unmount:
player.destroy();
```

Python validates the payload before rendering. Shared filters, matched synchronization and browser acceptance are not supplied by this player alone.

## Checks and downloads

From repository root:

```powershell
python -B -m unittest discover -s research_support/dashboard_reference/tests -p 'test_*.py'
```

This always checks the small authentic fixture and synthetic metadata. Full numerical acceptance requires the results download and `DASHBOARD_RESULTS_BUNDLE`; these checks supplement fixture checks. Spatial replay requires measured worlds. Developer/player checks use synthetic examples. Set `DASHBOARD_RELEASES_ROOT` to the common download directory for world-backed checks. See [integration.md](../../docs/research/dashboard/integration.md#offline-validation) for exact commands. No RL libraries or environment steps are required. A skipped download-dependent check is not a pass.
