"""Synthetic evidence tests; no saved scientific conclusions are generated."""

import json

import numpy as np
import pandas as pd

from src.evaluation.dashboard_evidence import (
    build_evidence_context,
    build_export,
    load_conclusions,
    selected_table,
)


def evidence_inputs():
    study = {
        "id": "synthetic_study", "protocol": "synthetic-protocol", "fixture": True,
        "caveats": ["Synthetic fixture; not research evidence."],
        "metadata": {
            "provenance": {"source_report_sha256": "b" * 64, "recovery_status": "resumed"},
            "datasets": [{"role": "FINAL_ID", "manifest_path": "datasets/final/manifest.json"}],
        },
        "records": [{"id": "model-1:FINAL_ID", "metadata": {"model": {
            "model_id": "model-1", "actual_training_steps": 101,
            "checkpoint_path": "C:/producer/models/model.zip",
        }}}],
        "results": {
            "export_sha256": "a" * 64, "export_path": "artifact:research_results",
            "extensions": {"compute_accounting": {"logical_training_interactions": 100}},
            "units": {"damage": "HP"},
        },
    }
    bundle = {
        "bundle_id": "synthetic_results", "local_path": "C:/releases/original",
        "files": [{"artifact_id": "research_results", "relative_path": "research_results.json",
                   "sha256": "a" * 64, "source_ref": {"sha256": "e" * 64}}],
    }
    publication = {
        "study_id": "synthetic_study", "protocol_id": "synthetic-protocol",
        "source": {"dashboard_export_sha256": "a" * 64, "producer_report_sha256": "b" * 64},
        "publication": {"status": "published", "release_tag": "synthetic-release",
                        "release_url": "https://example.org/release", "private_path": "C:/secret"},
    }
    return study, bundle, publication


def approved(evidence):
    return {
        "schema_version": 1, "status": "approved",
        "binding": {key: evidence[key] for key in ("study_id", "protocol_id", "release_tag", "export_sha256")},
        "evidence_cutoff": "2026-11-20", "reviewed_at": "2026-11-21T12:00:00Z",
        "author": "Synthetic reviewer", "reviewer": "Synthetic reviewer",
        "claims": [{"id": "synthetic-claim", "scope": "study-wide", "claim": "Fixture assertion only.",
                    "limitations": ["Synthetic example, not a scientific claim."],
                    "evidence": [{"record_id": "model-1:FINAL_ID", "model_id": "model-1"}]}],
    }


def write_sidecar(tmp_path, document):
    path = tmp_path / "conclusions.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


# Ensures portable metadata preserves published and original evidence hashes
def test_context_keeps_portable_evidence_metadata():
    context = build_evidence_context(*evidence_inputs())
    assert context["fixture"] is True
    assert context["publication_status"] == "published"
    assert context["source_hashes"]["source_report_sha256"] == "b" * 64
    assert context["artifacts"][0]["published_sha256"] == "a" * 64
    assert context["artifacts"][0]["original_sha256"] == "e" * 64
    assert context["datasets"][0]["manifest_path"] == "datasets/final/manifest.json"
    assert context["models"][0]["actual_training_steps"] == 101
    assert context["recovery_qualifiers"] == {"recovery_status": "resumed"}
    assert context["compute_accounting"] == {"logical_training_interactions": 100}
    assert context["units"] == {"damage": "HP"}
    assert "C:/" not in json.dumps(context, allow_nan=False)
    assert "private_path" not in context["publication"]


# Ensures publication metadata requires the loaded study, protocol and export hash
def test_publication_requires_matching_evidence():
    study, bundle, publication = evidence_inputs()
    for changed in (
        {**publication, "study_id": "other"},
        {**publication, "protocol_id": "other"},
        {**publication, "source": {"dashboard_export_sha256": "f" * 64}},
    ):
        context = build_evidence_context(study, bundle, changed)
        assert context["publication"] is None
        assert context["release_tag"] is None
        assert context["publication_status"] == "unknown"
        assert "producer_report_sha256" not in context["source_hashes"]


# Ensures approved claims bind to evidence rather than its download folder
def test_approved_review_requires_all_four_bindings(tmp_path):
    study, bundle, publication = evidence_inputs()
    evidence = build_evidence_context(study, bundle, publication)
    document = approved(evidence)
    path = write_sidecar(tmp_path, document)
    bundle["local_path"] = "D:/renamed/downloaded/release"
    moved = build_evidence_context(study, bundle, publication)
    assert moved == evidence
    result = load_conclusions(path, moved)
    assert result["status"] == "available"
    assert result["claims"] == document["claims"]

    for field in ("study_id", "protocol_id", "release_tag", "export_sha256"):
        result = load_conclusions(path, {**evidence, field: "other"})
        assert result["status"] == "mismatch"
        assert result["claims"] == []


# Ensures missing, draft and malformed sidecars leave claims unavailable
def test_unavailable_reviews_are_nonblocking(tmp_path):
    evidence = build_evidence_context(*evidence_inputs())
    missing = load_conclusions(tmp_path / "missing.json", evidence)
    document = approved(evidence)
    document["status"] = "draft"
    path = write_sidecar(tmp_path, document)
    draft = load_conclusions(path, evidence)
    path.write_text("{broken json", encoding="utf-8")
    malformed = load_conclusions(path, evidence)

    assert [missing["status"], draft["status"], malformed["status"]] == ["missing", "draft", "invalid"]
    assert missing["claims"] == draft["claims"] == malformed["claims"] == []


# Ensures a reviewed claim cannot cite evidence absent from the loaded study
def test_unknown_evidence_leaves_claims_unavailable(tmp_path):
    evidence = build_evidence_context(*evidence_inputs())
    document = approved(evidence)
    document["claims"][0]["evidence"] = [{"model_id": "missing-model"}]
    result = load_conclusions(write_sidecar(tmp_path, document), evidence)
    assert result["status"] == "invalid"
    assert result["claims"] == []


# Ensures exports match selected rows and preserve nulls without private paths
def test_export_matches_selected_table():
    raw = pd.DataFrame({
        "record_id": ["a", "b", "c"], "record_type": ["aggregate"] * 3,
        "reward_type": ["sparse", "safety", "sparse"], "success_count": [0, None, 2],
        "mean_damage": [np.nan, 2.5, float("inf")],
        "source_reference": ["artifact:research_results", "research_results.json", "C:/private/results.json"],
        "source_id": ["bundle:private:cache"] * 3, "reported_metrics": [{"path": "C:/private"}] * 3,
    })
    filtered = raw.iloc[[2, 0]]
    table = selected_table(filtered)
    assert list(table.index) == [2, 0]
    assert list(table.record_id) == ["c", "a"]
    assert list(table.mean_damage) == [None, None]
    assert list(table.source_reference) == [None, "artifact:research_results"]
    assert "source_id" not in table and "reported_metrics" not in table

    export = build_export(filtered, build_evidence_context(*evidence_inputs()),
        {"reward_types": ["sparse"], "model_seeds": [np.int64(7)]}, "Filtered summary",
        {"status": "missing", "claims": []}, {"success_rate": np.float64(0.333), "unavailable": pd.NA},
        displayed_views={"learning": {"scope": "selected_learning_model", "model_id": "policy-a",
            "points": pd.DataFrame({"actual_steps": [100123], "success_rate": [np.nan]})}})
    decoded = json.loads(json.dumps(export, allow_nan=False))
    assert decoded["table"] == {"columns": list(table.columns), "rows": table.to_dict("records")}
    assert decoded["filters"] == {"reward_types": ["sparse"], "model_seeds": [7]}
    assert decoded["displayed_metrics"] == {"success_rate": 0.333, "unavailable": None}
    assert decoded["displayed_views"]["learning"] == {
        "scope": "selected_learning_model", "model_id": "policy-a",
        "points": {"columns": ["actual_steps", "success_rate"], "rows": [{"actual_steps": 100123, "success_rate": None}]},
    }
    assert "not individual episodes" in decoded["row_scope"]
    assert "private" not in json.dumps(decoded)
    assert table.to_csv(index=False).splitlines()[1].startswith("c,aggregate,sparse,2.0,,")
