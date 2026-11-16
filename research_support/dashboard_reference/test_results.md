# Offline verification

Verification date: 2026-11-15. These checks cover reference code and saved-data compatibility; they do not certify a completed Streamlit application or completed primary research.

| Check | Result |
|---|---|
| Python with all reference releases configured | 78 tests passed, zero skipped |
| Default fixture-only suite |Download-dependent checks are explicitly skipped; they are not evidence passes |
| New recurrent coverage | 38 tests covering protocol separation, counter/grid validation, incomplete inventories, metrics, export and saved replay |
| Export integration |Results-only and results-plus-replay roundtrips; relocation, missing worlds, unchanged source hashes, Unicode job metadata and rejected incomplete campaigns |
| Historical numerical reference |3,500 baseline cases, 25 summary panels and 20 DEV learning points reconciled against the released answer key |
| Historical recorded replay | All 36 released case/trace/configuration/checkpoint/dataset/world joins checked |
| JavaScript player |Passed saved-frame transitions, independent mounts and cleanup in the in-memory DOM test; no real-browser claim |
| Repository boundary |No measured world files or full results corpus added; native replay edge cases are labeled synthetic test inputs |

From the repository root:

```powershell
python -B -m unittest discover -s research_support/dashboard_reference/tests -p 'test_*.py' -v

# After extracting the required results, worlds and synthetic-example assets:
$env:DASHBOARD_RELEASES_ROOT = (Resolve-Path 'docs/research/dashboard/downloaded_releases').Path
$env:DASHBOARD_RESULTS_BUNDLE = Join-Path $env:DASHBOARD_RELEASES_ROOT 'dashboard_recorded_results_v1'
python -B -m unittest discover -s research_support/dashboard_reference/tests -p 'test_*.py' -v
node research_support/dashboard_reference/viewer/test_grid_player.js
```

The bundle folder above is an internal manifest ID, not a prescribed GitHub release title. Obtain the current asset links and checksums from the release documentation. Full downloads stay outside Git. A results-only installation can validate numerical data; unavailable maps disable affected replay.

The recurrent tests use invented inputs, including a 65,000-entry synthetic case-index inventory. They do not use primary policy outcomes. Saved replay checks validate stored states, events, hashes and joins without simulator or model execution. The exporter forwards authoritative producer statistics; transport acceptance is not an independent raw-episode statistical recomputation.

Still required after integration: Streamlit browser/selection/filter/export agreement, real completed-primary release reconciliation, and final research interpretation. Existing missing-world/error messages must remain visible. `source_provenance.json` fingerprints the verified code/documentation tree and excludes itself.
