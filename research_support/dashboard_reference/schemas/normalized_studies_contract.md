# Normalized results v1

The interchange format is independent of the application UI. `normalized_studies.schema.json` describes the structure; `readers/normalized_model.py` is the dependency-free validator. `examples/normalized_studies.json` is a labeled synthetic example. Keep synthetic examples separate from observed results.

Top level: `{"schema_version":1,"studies":[...]}`. A study requires `id`, `title`, `status`, `protocol`, `sources`, `caveats`, `records`. Optional `group` defaults to Imported. `metadata` is an object for method/version, reward, entropy, lambda, collector, projection width, maps and budget conventions. `sources` lists original provenance, not executable instructions. `status` is one of completed/running/partial/failed/historical/unknown. An evidence-only study has `records:[]`, with its recorded outcome in `protocol` and caveats.

Each record is an aggregate for exactly one comparison group:

| Field | Meaning |
|---|---|
| id | Unique within the study |
| condition | Arm identifier; include evaluation set when sets overlap |
| split | TRAIN, DEV (repeatedly consulted development), or DIAGNOSTIC; never final ID/OOD |
| evaluation_mode | deterministic, sampled, or the precise historical diagnostic mode |
| orientation | original, reflected, or the recorded orientation |
| seed | Actual training seed, a clearly labeled historical seed slot, or null |
| sampling_seed | Optional action-sampling replicate; never treated as another trained seed |
| checkpoint | Nonnegative fixed interaction/target budget, or null |
| successes / episodes | Success numerator and positive episode denominator, or null |
| deaths / timeouts | Optional episode counts, sharing `episodes` |
| mean_damage | Optional nonnegative finite per-episode mean |
| invalid_actions / transitions | Optional transition numerator/denominator, not an episode rate |
| source / evidence / data_status | Required provenance path/text, evidence type, and completeness label |
| metadata | Optional object for method and actual policy/update timing |

Required strings must be nonempty. Study IDs use 1–160 ASCII letters/digits/underscore/hyphen/dot. Duplicate study IDs, record IDs and comparison dimensions are rejected. Counts must be integers, not booleans or rates. Known success/death/timeout counts cannot sum beyond the episode denominator. Invalid actions need a positive transition denominator. Missing optional values may be omitted or null; they never become zero. NaN/Infinity and final splits are rejected. For distinct diagnostic case sets, use distinct conditions or studies and describe selection in the protocol. Do not aggregate independent training seeds into one record: keep every seed visible.

Legacy readers require an explicit caller-configured `allowed_root`. Network paths, traversal outside that root, hidden directories and paths naming final/held-out/OOD data are denied. JSON files are capped at 48 MiB. Validation does not follow any path in `sources`; those strings are provenance. Newline-terminated CSV is required; an unterminated final physical line is conservatively ignored until a later explicit read. No personal host root or built-in study catalog is discovered.

## Extend an adapter

Use normalized JSON for a compatible new study. For a new raw layout, add a bounded reader with explicit roots and tests; application integration remains separate. Read only explicitly configured filenames. Emit the schema above, preserve portable source identities and hashes, and test the reader with small fixtures. Keep scientific selection (split, mode, orientation, training seed, action seed, case set and target budget) in the grouping key.

The existing raw episode adapter supports the known 300 TRAIN / 100 development full-parent protocol and the entropy discovery's 20-parent sampled diagnostic per base sampling seed. It does not infer arbitrary sample sizes from growing files. Other counts/layouts should use normalized JSON or a new adapter with an explicit expected-count rule. Duplicate/missing parents withhold the entire affected group. No final claim is inferred from CSV existence; manual run status comes only from saved `state/execution_status.json`, with the original status and timestamp retained.

This reusable package has no historical study catalog or refresh service. Use a distinct study/version identity when correcting externally supplied evidence.
