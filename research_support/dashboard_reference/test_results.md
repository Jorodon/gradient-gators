# Offline verification

Verification dated 2026-11-15 covers saved evidence, reference code and package integrity. It does not certify an integrated Streamlit application.

| Check | Observed result |
|---|---|
| Python without downloads | 40 tests: 34 passed, 6 explicitly skipped because optional release evidence was absent |
| Python with all three local releases configured | 40 tests passed, zero skipped |
| JavaScript grid player | Passed saved-frame transitions, independent mounts and cleanup using the external synthetic fixture |
| Issue publisher | 37 mocked tests passed; both ten-draft wording versions rendered locally, with no GitHub calls |
| Small recorded fixture | Three authentic cases/traces with unchanged raw bytes: success/invalid move, timeout/invalid moves, and supplemental WAIT; no aggregate conclusions permitted |
| Full numerical reference | Recomputed 3,500 baseline cases into 25 summary panels and 20 DEV learning points; compared the full answer key, including learning area |
| Full recorded replay | All 36 case/trace/configuration/checkpoint/dataset/world joins passed; supplemental WAIT remains separate |
| Evidence preservation | All 3,501 raw cases and 36 traces retain their original bytes and internal seals |
| Repository boundary | No playable maps in the repository payload |

Obtain the relevant release links from **@jorodons**. Create `docs/research/dashboard/downloaded_releases/` at the repository root and extract the downloads there, preserving bundled top-level folders and manifest IDs. Results are required for full numerical acceptance, measured worlds for spatial replay, and synthetic examples for developer/player checks. The folder name `dashboard_recorded_results_v1` below is an internal bundle ID, not a GitHub release title, tag or published asset name.

Run from the repository root:

```powershell
# Default: lightweight fixture + metadata/math/path checks.
python -B -m unittest discover -s research_support/dashboard_reference/tests -p 'test_*.py' -v

# Full evidence and spatial replay after extracting the three separate releases.
$env:DASHBOARD_RELEASES_ROOT = (Resolve-Path 'docs/research/dashboard/downloaded_releases').Path
$env:DASHBOARD_RESULTS_BUNDLE = Join-Path $env:DASHBOARD_RELEASES_ROOT 'dashboard_recorded_results_v1'
python -B -m unittest discover -s research_support/dashboard_reference/tests -p 'test_*.py' -v
node research_support/dashboard_reference/viewer/test_grid_player.js
```

The optional full-results tests add coverage; they do not replace the small-fixture tests. Results without world downloads still support numerical validation. Missing assets produce explicit unavailable/skip behavior, never an invented replay or a zero-success row. A skip is not a passed evidence check.

`source_provenance.json` excludes itself from its file hashes. The results bundle's `publication_verification.json` records evidence-preparation checks; it does not establish application integration or publication.

Remaining boundaries: Streamlit selection/filter/summary/curve/replay/export agreement and real-browser accessibility/lifecycle still require team integration. These checks do not rerun simulator physics, verify absent model weights, or demonstrate final-policy robustness. No authentic death-on-goal example is supplied. No training, policy inference or environment execution is part of these checks. Publisher tests made no live GitHub calls.
