# Recurrent research protocol and dashboard export

Streamlit consumes validated saved records. The research campaign, export tool and dashboard are separate stages: training produces evidence, the offline exporter translates it, and the UI displays it. No dashboard action runs a policy or modifies evidence.

## Supported scientific contracts

| Field | Historical common PPO | Recurrent complete-episode PPO |
|---|---|---|
| Protocol | `m4-common-ppo-v1` | `recurrent-complete-900-2m-v1` |
| Wire format | `research-results-v1` or `m4-results-v1` | `research-results-v2` |
| TRAIN collection | 300 | 900 |
| Final endpoint | 1M | 2M consumed interactions |
| Nominal learning checkpoints | 0,250k,500k,750k,1M | 0,100k,250k,500k,1M,1.5M,2M |
| Learning/final scenarios per panel | 100/200 | 100/200 |

The recurrent common method uses normalized geometry observations, recurrent PPO, entropy 0.01, gamma 0.99, GAE lambda 0.95, learning rate 0.0003, ten epochs, batch target 125, clip 0.2 and reward scale 0.01. It collects at least 1,000 transitions and finishes the current episode before updating; the collector capacity is 1,399. An incomplete final episode prefix at the exact 2M cap is excluded from optimization and reported. The five original reward definitions remain unchanged.

Five independent training seeds are paired across the five rewards. Their approved slot inventory belongs to the scientific protocol; selectors display the identities from records rather than hardcoding names. New experiments can declare other study/release identities, but a different scientific design needs a supported protocol definition rather than silently bypassing validation.

The complete recurrent inventory contains 25 final policies and 175 saved checkpoint-model identities,175 learning panels, 25 TRAIN panels and 125 final panels:65,000 unique evaluation cases. Repeated recovery work and selected replay-verification attempts are separate. Inventory completeness does not itself establish statistical correctness, experiment quality or task success.

## Export a completed campaign

The exporter reads a native completed report, matrix/contract hashes, collection manifests, case index, selected trace list and passed replay receipt. It refuses a running/incomplete controller and never imports an RL library. Choose a new output directory outside the source campaign. Existing outputs are not overwritten; failed exports retain a labeled partial directory and cannot appear as a completed bundle.

From the repository root, after the campaign completes:

```powershell
$CampaignRoot = Read-Host 'Completed campaign directory'
$ExportRoot = Read-Host 'New export directory outside the campaign'
python -B -m research_support.dashboard_reference.readers.recurrent_export --source "$CampaignRoot" --output "$ExportRoot" --study-id recurrent_ppo_complete_episode_experiment --include-replay
```

The example study ID is a label, not a fixed importer requirement. Omit `--include-replay` for numerical results only. The tool creates a results bundle and, when requested, a distinct selected-worlds bundle. It does not package model weights or the entire training directory. Metadata-only collection manifests list the expected scenarios; they are not playable world files. Each bundle has a release manifest and file hashes.

Upload the two bundle folders as separate release assets when reviewed. Do not commit their full contents. Consumers extract the assets with their top-level folder names intact under `docs/research/dashboard/downloaded_releases/`. Actual release names/tags are supplied with the publication; no fixed GitHub release name is required. Missing worlds disable affected replay, not numerical results.

The exporter verifies structural coverage, configuration identity and selected replay joins. It forwards the producer's scientific statistics and does not independently recompute 65,000 raw episodes or reverify absent model archives. Keep the original run and its validation receipts for audit.

The report and native case recorder use different canonical Unicode encodings for job hashes. The export retains `producer_report_job_sha256` and the native `source_job_sha256` separately; these identify the same source job under their documented hash domains and must not be substituted for each other.

## Integration API and metric fields

Use `BundleIndex(bundle_directory, releases_root=download_directory)`, then `load_bundle_export(bundle, 'research_results')`. Standard records expose nominal `checkpoint`, `actual_training_steps` and `optimized_training_steps`. The `results.inventory` object records required/observed coverage; unavailable data must not become zero or a robustness pass.

The exporter preserves native scientific values in named fields:

| Export field | Meaning |
|---|---|
| `learning` |175 per-checkpoint evaluation panels with nominal and actual coordinates |
| `producer_learning_metrics` |25 complete producer curves, `normalized_aulc`, `time_to_sustained_80`, `threshold_status`, `censoring_steps` |
| `comparisons` |Four prespecified shaped-minus-sparse combined-OOD contrasts; native `ci95_nominal`, `p_two_sided`, `p_holm` and paired-seed differences |
| `producer_robustness_models` / `producer_robustness_rewards` |Exported per-model checks and reward-family qualification |
| Outcome `source_metrics` |Original HP damage, source damage, action fractions, raw return and conditional route statistics with their original names |
| `provenance` / `compute_accounting` |Producer hashes, evidence scope, logical interaction and unique-case totals |

The reader retains these values under `results.extensions` and per-record `metadata.exported`. Do not treat an average illegal-move count as an integer pooled total. Raw return is not a common scale for ranking different reward formulas.

Recurrent AULC uses trapezoidal success area against actual consumed checkpoint steps divided by 2M. Sustained 80% requires two adjacent positive scheduled checkpoints, with attainment at the second point's actual coordinate. Missing observations make a curve incomplete; a fully observed curve that never qualifies is right-censored at 2M. The pure metric helper uses `time_to_80_successes` and `censoring_timestep` names; these correspond to the producer names above. Do not recompute either representation in browser code.

Final ID and hazard/obstacle/pursuit/combined OOD each have 200 scenarios per model. The original broad-robustness thresholds and four-test Holm family remain unchanged. Five seeds, not hundreds of repeated episodes, are the independent training replicates. A valid weak policy remains valid evidence.

## Saved replay and acceptance

Use `load_selected(study, selected_index, bundle=bundle)`. A recurrent selection binds the exact case, trace, world, collection, job digest, checkpoint hash, evaluation seed and condition. The loader checks original content seals and stored event/summary totals, then returns the same grid-player frame interface. It does not simulate missing states or reevaluate a model.

Matched selections use two outcome-independent hash-selected scenarios per condition, primary slot 0 and all five rewards; representative selections remain separate. Read the producer selection list. Do not use the historical 1M reference selection file as a 2M trace index.

Synthetic checks cover protocol separation, checkpoint overshoot, missing/censored curves, complete-vs-partial inventories, export relocation, source immutability and saved replay joins. Existing historical tests remain applicable. Real final release reconciliation and the Streamlit browser smoke test remain required when that evidence is available; synthetic passing tests are not primary research results.
