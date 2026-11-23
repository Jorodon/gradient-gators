"""Portable dashboard evidence and human-authored, release-bound conclusions.

These helpers preserve recorded statistics and human-authored claims. Display
and export callers use the same selection and portable table conversion.
"""

from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import re

import pandas as pd


TABLE_COLUMNS = frozenset("""
record_id study_id protocol_id model_id configuration_id comparison_id run_id
attempt_id experiment_id reward_type reward condition model_seed training_seed
seed model_seed_slot seed_slot slot checkpoint_timestep checkpoint checkpoint_step
dataset_role role split distribution distribution_label evaluation_mode orientation
episode_id scenario_id trace_id environment_seed episode scenario success terminated
truncated episode_count episodes success_count successes death_count deaths
timeout_count timeouts n_expected n_completed n_missing transitions invalid_actions
episode_return episode_length damage_taken fall_damage hazard_contacts enemy_contacts
invalid_moves mean_damage mean_length mean_length_success mean_length_failure
nominal_training_steps consumed_training_steps actual_training_steps
optimized_training_steps final_unoptimized_steps source_reference data_source
data_origin study_status record_type evidence data_status trace_available
success_rate success_fraction mean_episode_return mean_episode_length
damage_unit length_unit return_unit training_steps_unit metric_unit unit
""".split())
_PRIVATE = {
    "metadata", "source_path", "source_id", "cache_key", "cache_id",
    "reported_metrics", "scientific_results",
}
_BINDING = ("study_id", "protocol_id", "release_tag", "export_sha256")
_IDENTITIES = ("model_id", "record_id", "configuration_id", "comparison_id")
_ABSOLUTE = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]|\\\\|(?:^|[\s\"'(])/(?!/)[A-Za-z_.]|(?:^|[/\\])\.\.(?:[/\\]|$)")

# --- HELPER FUNCTIONS --- #


def _scalar(value):
    """Convert safe scalar values and preserve missing measurements as nulls."""
    if value is None or isinstance(value, (dict, list, tuple, set, Path)) or getattr(value, "ndim", 0) != 0:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if pd.isna(value):
        return None
    if isinstance(value, str):
        if _ABSOLUTE.search(value) or value.startswith(("//", "\\", "~/", "bundle:", "normalized:", "native:")):
            return None
        return value
    if isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return None


def _portable(value):
    """Remove private fields and paths from nested export metadata."""
    if isinstance(value, dict):
        return {
            key: _portable(item) for key, item in value.items()
            if isinstance(key, str) and not key.startswith("_")
            and key not in _PRIVATE and _scalar(key) is not None
        }
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    return _scalar(value)


def _pick(document, fields):
    """Select portable scalar fields from optional metadata."""
    if not isinstance(document, dict):
        return {}
    return {key: _scalar(document[key]) for key in fields if key in document}


def _reference(value):
    """Keep artifact identifiers and relative paths as source references."""
    value = _scalar(value)
    if not isinstance(value, str) or not value or "\\" in value or "://" in value:
        return None
    if value.startswith("artifact:"):
        return value if re.fullmatch(r"artifact:[a-z][a-z0-9_]*", value) else None
    return value if ":" not in value and all(part and not part.startswith(".") for part in value.split("/")) else None


def _text(value):
    """Check that required text is nonempty and contains no private path."""
    return isinstance(value, str) and bool(value.strip()) and _scalar(value) == value


# --- EVIDENCE CONTEXT --- #


def build_evidence_context(study: dict, bundle_manifest: dict, publication: dict | None = None) -> dict:
    """Collect source provenance and attach publication metadata on exact matches."""
    results = study.get("results", {})
    metadata = study.get("metadata", {})
    provenance = metadata.get("provenance", {})
    context = dict(
        study_id=_scalar(study.get("id")), protocol_id=_scalar(study.get("protocol")),
        bundle_id=_scalar(bundle_manifest.get("bundle_id")),
        export_sha256=_scalar(results.get("export_sha256")),
        export_reference=_reference(results.get("export_path")),
        fixture=study.get("fixture") is True, publication_status="unknown",
        release_tag=None, publication=None,
    )
    publication = publication if isinstance(publication, dict) else {}
    source = publication.get("source", {})
    source = source if isinstance(source, dict) else {}
    matches = (
        all(context[key] and publication.get(key) == context[key] for key in ("study_id", "protocol_id"))
        and context["export_sha256"] and source.get("dashboard_export_sha256") == context["export_sha256"]
    )
    if matches:
        context["publication"] = _pick(
            publication.get("publication", {}),
            ("status", "release_tag", "release_url", "release_commit", "human_reviewer"),
        )
        context["release_tag"] = context["publication"].get("release_tag")
        context["publication_status"] = context["publication"].get("status") or "unknown"
    context["source_hashes"] = dict(
        published_export_sha256=context["export_sha256"],
        **_pick(provenance, ("source_report_sha256", "source_matrix_sha256", "source_contract_sha256", "source_seal_sha256")),
        **(_pick(source, ("producer_source_seal_sha256", "producer_report_sha256")) if matches else {}),
    )
    artifacts = []
    for entry in bundle_manifest.get("files", []):
        original = entry.get("source_ref", {})
        artifacts.append(dict(
            **_pick(entry, ("artifact_id", "category", "release_id", "source_byte_identical")),
            relative_path=_reference(entry.get("relative_path")),
            published_sha256=_scalar(entry.get("sha256")),
            original_artifact_path=_reference(original.get("artifact_path")),
            original_sha256=_scalar(original.get("sha256")),
        ))
    context["artifacts"] = artifacts
    context["datasets"] = [
        dict(
            **_pick(dataset, ("role", "scenario_count", "manifest_sha256")),
            manifest_path=_reference(dataset.get("manifest_path")),
        )
        for dataset in metadata.get("datasets", [])
    ]
    models = {}
    for record in study.get("records", []):
        model = record.get("metadata", {}).get("model", {})
        if model.get("model_id"):
            models[model["model_id"]] = _pick(model, (
                "model_id", "source_model_id", "reward", "seed", "slot", "seed_slot",
                "checkpoint_step", "actual_training_steps", "optimized_training_steps", "final_unoptimized_steps",
                "checkpoint_sha256", "source_job_sha256", "producer_report_job_sha256", "recovery_status",
            ))
    context["models"] = list(models.values())
    context["record_ids"] = [_scalar(record.get("id")) for record in study.get("records", [])]
    context["comparison_ids"] = [
        _scalar(value) for row in results.get("comparisons", [])
        if (value := row.get("comparison_id", row.get("id")))
    ]
    context["configuration_ids"] = [value for key in ("configuration_id", "run_id") if (value := _scalar(provenance.get(key)))]
    context["recovery_qualifiers"] = _pick(provenance, ("recovery_status", "recovery_note", "recovery_method", "validation_scope"))
    context["compute_accounting"] = _pick(
        results.get("extensions", {}).get("compute_accounting", {}),
        ("logical_training_interactions", "unique_evaluation_cases", "note"),
    )
    context["limitations"] = [value for item in study.get("caveats", []) if (value := _scalar(item)) is not None]
    context["units"] = {
        key: _scalar(value) for key, value in results.get("units", {}).items()
        if _text(key) and key not in _PRIVATE and not key.startswith("_")
    }
    return context


# --- REVIEWED CONCLUSIONS --- #


def _unavailable(status, message):
    """Return a conclusion state without approved claims."""
    return dict(status=status, message=message, claims=[])


def load_conclusions(path: Path, evidence: dict) -> dict:
    """Load reviewed claims only when their study, release and evidence match."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return _unavailable("missing", "No approved conclusions available for this evidence; no reviewed sidecar was found.")
    except (OSError, UnicodeError, ValueError):
        return _unavailable("invalid", "Conclusions could not be read as valid JSON.")
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        return _unavailable("invalid", "Conclusions require sidecar schema version 1.")
    status = document.get("status")
    if status in ("draft", "superseded"):
        return _unavailable(status, f"Conclusions are {status}; approved claims are unavailable.")
    if status != "approved":
        return _unavailable("invalid", "Conclusions require an explicit approved review status.")
    binding = document.get("binding")
    if (not isinstance(binding, dict) or set(binding) != set(_BINDING)
            or any(not _text(binding.get(key)) or binding[key] != evidence.get(key) for key in _BINDING)):
        return _unavailable("mismatch", "Conclusions do not match this study, protocol, release and export hash.")
    try:
        for field in ("evidence_cutoff", "reviewed_at"):
            datetime.fromisoformat(document[field].replace("Z", "+00:00"))
        if not all(_text(document.get(key)) for key in ("author", "reviewer")):
            raise ValueError
        claims = document["claims"]
        if not isinstance(claims, list) or not claims:
            raise ValueError
        known = {
            "model_id": {model.get("model_id") for model in evidence.get("models", [])},
            **{key: set(evidence.get(key[:-3] + "_ids", [])) for key in _IDENTITIES[1:]},
        }
        seen = set()
        approved = []
        for claim in claims:
            if (not isinstance(claim, dict) or not all(_text(claim.get(key)) for key in ("id", "scope", "claim"))
                    or claim["id"] in seen or not isinstance(claim.get("limitations"), list)
                    or not all(_text(item) for item in claim["limitations"])
                    or not isinstance(claim.get("evidence"), list) or not claim["evidence"]):
                raise ValueError
            seen.add(claim["id"])
            refs = []
            for ref in claim["evidence"]:
                if not isinstance(ref, dict):
                    raise ValueError
                identities = [key for key in _IDENTITIES if key in ref]
                if not identities or any(not _text(ref[key]) or ref[key] not in known[key] for key in identities):
                    raise ValueError
                refs.append(_pick(ref, (*_IDENTITIES, "dataset_role", "evaluation_mode", "checkpoint_timestep", "reward", "seed")))
            approved.append(dict(**_pick(claim, ("id", "scope", "claim")), limitations=claim["limitations"], evidence=refs))
    except (KeyError, TypeError, ValueError, AttributeError):
        return _unavailable("invalid", "Conclusions require valid review dates, authorship and claims with known evidence identities.")
    return dict(
        status="available", message="Approved human-reviewed conclusions for the bound study and release.",
        binding=dict(binding), claims=approved,
        **_pick(document, ("evidence_cutoff", "author", "reviewer", "reviewed_at")),
    )


# --- DISPLAY AND EXPORT --- #


def selected_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Prepare the displayed and exported values with their original row order."""
    columns = [column for column in dataframe.columns if column in TABLE_COLUMNS]
    return pd.DataFrame({
        column: pd.Series(
            [(_reference(value) if column == "source_reference" else _scalar(value)) for value in dataframe[column]],
            index=dataframe.index, dtype=object,
        ) for column in columns
    }, index=dataframe.index)


def build_export(
    table: pd.DataFrame, evidence: dict, filters: dict, view_scope: str,
    conclusions: dict, displayed_metrics=None,
) -> dict:
    """Prepare selected rows for export and preserve the supplied metrics."""
    table = selected_table(table)
    kinds = set(table.get("record_type", pd.Series(dtype=object)).dropna())
    if kinds == {"aggregate"}:
        row_scope = "aggregate panels; rows are not individual episodes"
    elif "aggregate" in kinds:
        row_scope = "mixed aggregate panels and episode records"
    elif "episode_id" in table or "episode" in table:
        row_scope = "individual episodes"
    else:
        row_scope = "selected records; row granularity is not declared"

    payload = _portable(dict(
        schema_version=1, evidence=evidence, filters=filters,
        view_scope=view_scope, row_scope=row_scope,
        displayed_metrics=displayed_metrics, conclusions=conclusions,
    ))
    # The selected table already contains portable scalar values.
    payload["table"] = dict(columns=list(table.columns), rows=table.to_dict("records"))
    return payload
