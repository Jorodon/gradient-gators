# Research dashboard reference

**Streamlit remains the application.** This package supplies readers, metric calculations, a saved-episode player and offline checks for the team's existing dashboard. It does not replace that dashboard or complete its integration.

The integration reference is `develop` commit `e1a13b32c0149376a9cd8f07e044237241a937d1`. At that revision, the existing loader has summary-file precedence, incomplete trace identities, mixed-checkpoint defaults and undisplayed learning metrics.

## How the parts fit

```text
Saved experiment outputs + explicitly downloaded releases
  -> readers: verify identities, hashes and coverage
  -> common records + exported scientific metrics
  -> Streamlit: filters, charts, tables, conclusions and exports
       -> validated saved frames -> grid player
```

Python validates evidence; metric exporters calculate scientific results; Streamlit owns selection state; JavaScript displays saved frames. No dashboard action trains a policy, loads model weights, steps an environment or changes evidence. The player neither fetches arbitrary files nor validates simulator physics. Use a component API supported by the project's installed Streamlit version; see the [official component documentation](https://docs.streamlit.io/develop/concepts/custom-components).

The repository contains code, schemas, synthetic metadata examples and a small authentic test fixture. Obtain the relevant release links from **@jorodons**, create `docs/research/dashboard/downloaded_releases/` at the repository root and extract the downloads while preserving their bundled top-level folders and manifest IDs. The results bundle is required for full numerical acceptance, measured worlds for spatial replay, and synthetic examples for developer checks. A missing world disables affected spatial replay; available numerical evidence remains readable. [Integration and evidence](integration.md) explains the release layout, exact dataset scope and checks. [Code guide](../../../research_support/dashboard_reference/README.md) explains each reusable file.

## Keep loading simple

Extend the existing artifact-directory input and optional `GATOR_DASHBOARD_DATA_DIR` setting. The user selects the common download directory in Streamlit. No database, registration service, second server, frontend build system or automatic downloader is required. One in-memory result structure and the existing Streamlit cache/session state are enough.

Accept either that common download directory or a specific bundle directory. If the selected directory contains `bundle_manifest.json`, read that bundle and resolve sibling world releases from its parent. Otherwise inspect only immediate children for supported manifests and let the user choose a study; do not crawl unrelated directories or automatically pool studies. Preserve support for the existing selected CSV directory. Unsupported formats get a specific message; additional historical adapters can be added only when needed.

The interface should distinguish **Mock examples** from **Saved results**. Preserve existing PILOT/REAL loader compatibility; identify development/final phase from the records, not an assumption that REAL means final. Results tables and curves require the results download; map playback additionally requires the world download. The synthetic world release is for developer tests, not required to view measured results.

This is the agreed integration behavior for the issues. Copying the reference package alone does **not** make the current Streamlit app understand JSON bundles. Keep the existing app and views; the ten issues are work items, not ten new pages.

## Presentation rules

- Declare checkpoint/budget, dataset identity, evaluation mode, reward definition and seed scope. Keep endpoint and learning-history views distinct.
- Keep measured/synthetic origin separate from development/final phase. Repeated DEV evaluation is not final ID evidence.
- Show successes/completed and completed/expected. Missing is not zero. Training seeds are independent policies; repeated scenarios or action draws are not.
- Compare task/safety measures across rewards. Compare raw return only within a specified reward definition and scale.
- Preserve units and denominators: HP damage, pooled invalid actions/transitions, mean episode invalid fraction, safe successes/all episodes and success-conditional metrics answer different questions.
- Display versioned exporter statistics. Browser code must not silently recalculate confidence intervals, filtered Holm families, learning area or thresholds.
- Preserve strict final rules: TRAIN300, DEV100, learning-validation100 and final200; learning grid 0/250k/500k/750k/1M; matched final replay at 1M and primary slot0. Development adapters must explicitly represent their own grids and TRAIN100/300/900 rather than weaken final validators.
- Keep final robustness unavailable until accepted final outputs exist. Match scenarios across rewards within a condition, not across unrelated ID/OOD collections.
- Preserve author, date, evidence cutoff, supporting records, limitations and superseded status for conclusions. New data does not silently rewrite interpretations.

## Views and ownership

| View | Purpose | Issue boundary |
|---|---|---|
| Shared evidence/selection | Identity, coverage, capabilities and common filters | DASH-01 and DASH-04 |
| Results | Comparable success/safety/routing metrics, seed points and exact counts | DASH-02 producer summaries; DASH-05 UI |
| Learning | Checkpoint curves, AULC, threshold attainment, censoring and completeness | DASH-02 producer curves; DASH-06 UI |
| Robustness | ID/four OOD conditions, per-seed coverage and exported qualification | DASH-07 |
| Replay | Reset/post-action frames, geometry, HP/enemies/events; synchronized comparison optional | DASH-03 joins; DASH-09 player integration |
| Evidence/conclusions | Configuration, provenance, authored findings and exports | DASH-08 |
| Acceptance | Data checks, browser consistency and onboarding | DASH-10 |

Obtain the foundation commit and relevant release links from @jorodons. Checkout-only UI/contracts can be developed using small fixtures, while full numerical and spatial acceptance require the appropriate downloads. Both technical and beginner-friendly drafts describe the same ten issues.

## Names and identities

Use lowercase `snake_case` for new files, folders and stable experiment IDs, including acronyms (`ppo`, `lstm`). Do not name experiments after a milestone. Keep friendly labels separately.

| Purpose | Example |
|---|---|
| Experiment ID | `entropy_0_01_experiment` |
| Recurrence family | `recurrent_ppo_experiment` |
| Distinct combined study | `recurrent_ppo_maps900_complete_episode_experiment` |
| Display label | `Efficiency with entropy 0.01` |
| Condition path | `conditions/efficiency/` |
| Internal bundle ID | `dashboard_recorded_results_v1` |

Bundle IDs identify folders and manifest references; they do not determine GitHub release titles, tags or published asset names. Encode decimal values as `0_01` in IDs and `0.01` in labels. Keep reward, entropy, collector, map count, seeds, budget, checkpoint and evaluation mode in explicit metadata; do not infer them from names.

Historical job/collection IDs, canonical-hash domains, protocol IDs and the legacy `m4-results-v1` alias remain unchanged inside sealed evidence. Renaming them could break joins or the frozen matched-scenario rule. GitHub's `Milestone 4` remains organizational metadata. A `source_ref` contains a logical producer ID, relative artifact path and SHA-256; the caller supplies its own local root at runtime.
