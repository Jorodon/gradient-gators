"""Robustness view: what it shows, and more importantly what it refuses to show.

The risk this view carries is not a wrong number on screen - it is a
qualification printed from evidence that does not support one. So most of these
tests assert an absence: no verdict from synthetic data, no pass from an
incomplete cohort, no final claim from a checkpoint the protocol did not
prescribe. A blank cell is the correct output for a condition nobody evaluated,
and a test that only checked the happy path would never notice if it became a
zero instead.
"""

import pandas as pd
import pytest

from research_support.dashboard_reference.protocols import (
    LEGACY_PROTOCOL_ID,
    PRIMARY_SEEDS,
    RECURRENT_PROTOCOL_ID,
)
from src.evaluation.dashboard_app import _render_robustness

FINAL_ROLES = (
    "FINAL_ID",
    "FINAL_OOD_HAZARD",
    "FINAL_OOD_OBSTACLE",
    "FINAL_OOD_PURSUIT",
    "FINAL_OOD_COMBINED",
)


class Recorder:
    """Minimal stand-in for the Streamlit module, capturing what was rendered."""

    def __init__(self):
        self.messages = {"info": [], "warning": [], "error": [], "caption": [],
                         "markdown": [], "subheader": []}
        self.frames = []

    def _record(self, kind):
        def capture(text, *args, **kwargs):
            self.messages[kind].append(str(text))
        return capture

    def __getattr__(self, name):
        if name in ("info", "warning", "error", "caption", "markdown", "subheader"):
            return self._record(name)
        raise AttributeError(name)

    def dataframe(self, frame, *args, **kwargs):
        self.frames.append(frame)

    def text(self, kind):
        return " ".join(self.messages[kind])

    def all_text(self):
        return " ".join(sum(self.messages.values(), []))

    def frame_with(self, column):
        for frame in self.frames:
            if column in getattr(frame, "columns", ()):
                return frame
        return None


class Artifacts:
    def __init__(self, scientific_results):
        self.scientific_results = scientific_results


def build_results(
    *,
    protocol_id=RECURRENT_PROTOCOL_ID,
    checkpoint_step=2_000_000,
    fixture=False,
    complete=True,
    rates=None,
    missing_role=None,
):
    """An export shaped like research_metrics.build_export output."""
    rates = rates if rates is not None else {role: 0.9 for role in FINAL_ROLES}

    models, outcomes, assessments = [], [], []

    for seed in PRIMARY_SEEDS:
        model_id = f"sparse_{seed}"
        models.append({
            "model_id": model_id, "reward": "sparse", "seed": seed,
            "checkpoint_step": checkpoint_step,
        })

        for role in FINAL_ROLES:
            fraction = None if role == missing_role else rates[role]
            outcomes.append({
                "model_id": model_id, "role": role,
                "n_expected": 200,
                "n_completed": 0 if fraction is None else 200,
                "success_fraction": fraction,
                "death_count": None if fraction is None else 4,
                "timeout_count": None if fraction is None else 2,
            })

        base = rates["FINAL_ID"]
        checks = {
            "id": base >= 0.70,
            "combined": rates["FINAL_OOD_COMBINED"] >= 0.35,
            "combined_drop": base - rates["FINAL_OOD_COMBINED"] <= 0.40,
            **{
                role: rates[role] >= 0.50 and base - rates[role] <= 0.25
                for role in FINAL_ROLES[1:-1]
            },
        }
        assessed = missing_role is None
        assessments.append({
            "model_id": model_id, "reward": "sparse", "seed": seed,
            "fully_assessed": assessed,
            "passes": all(checks.values()) if assessed else None,
            "checks": checks if assessed else None,
            "reason": None if assessed else "missing_conditions",
        })

    n_pass = sum(1 for item in assessments if item["passes"] is True)
    n_assessed = sum(1 for item in assessments if item["fully_assessed"])

    return {
        "protocol_id": protocol_id,
        "fixture": fixture,
        "models": models,
        "outcomes": outcomes,
        "robustness_models": assessments,
        "robustness_families": [{
            "reward": "sparse", "n_expected": 5,
            "n_fully_assessed": n_assessed, "n_pass": n_pass,
            "passes": n_pass >= 4 if n_assessed == 5 else None,
            "reason": None if n_assessed == 5 else "five_fully_assessed_models_required",
        }],
        "comparisons": [{
            "reward": "efficiency", "reference": "sparse",
            "role": "FINAL_OOD_COMBINED", "n_pairs": 5,
            "mean_difference": 0.08, "ci95": [0.01, 0.15],
            "p_two_sided": 0.03, "p_holm": 0.09,
            "reject_holm_05": False, "reason": None,
        }],
        "coverage": {
            "complete": complete,
            "n_expected_episodes": 5000,
            "n_completed_episodes": 5000 if complete else 4200,
            "missing_conditions": [] if complete else ["FINAL_OOD_PURSUIT"],
            "missing_reward_seed_models": [],
        },
        "limitations": ["Five trained seeds are the independent replicates."],
    }


def render(results):
    recorder = Recorder()
    _render_robustness(recorder, Artifacts(results) if results is not None else None)
    return recorder


# --------------------------------------------------------------------------- #
# Absent evidence
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("artifacts_results", [None, {}])
def test_absent_results_render_pending_and_no_table(artifacts_results):
    """Pending is a state, not an error, and it must not render a table."""
    recorder = render(artifacts_results)

    assert "Pending" in recorder.text("info")
    assert not recorder.frames


def test_pending_message_refuses_development_data_as_a_substitute():
    """The specific confusion this view exists to prevent."""
    recorder = render(None)

    assert "development results cannot stand in" in recorder.text("info")


# --------------------------------------------------------------------------- #
# Synthetic evidence
# --------------------------------------------------------------------------- #


def test_fixture_data_is_labelled_and_yields_no_verdict():
    """A synthetic example may be drawn, but it cannot qualify anything."""
    recorder = render(build_results(fixture=True))

    assert "Synthetic example" in recorder.text("warning")

    models = recorder.frame_with("passes")
    families = recorder.frame_with("family passes")

    assert models["passes"].isna().all()
    assert families["family passes"].isna().all()


def test_fixture_data_does_not_also_claim_incompleteness():
    """Labelling it synthetic is the point; an incomplete warning would bury that."""
    recorder = render(build_results(fixture=True, complete=False))

    assert "Incomplete evaluation" not in recorder.text("warning")


# --------------------------------------------------------------------------- #
# Incomplete evaluation
# --------------------------------------------------------------------------- #


def test_incomplete_cohort_shows_coverage_but_withholds_the_pass():
    recorder = render(build_results(complete=False))
    text = recorder.text("warning")

    assert "Incomplete evaluation" in text
    assert "4200" in text and "5000" in text
    assert "FINAL_OOD_PURSUIT" in text
    assert recorder.frame_with("passes")["passes"].isna().all()


def test_unevaluated_condition_is_blank_not_zero():
    """Missing is not zero - a blank cell and a 0% score mean opposite things."""
    recorder = render(build_results(missing_role="FINAL_OOD_PURSUIT", complete=False))
    rates = recorder.frame_with("success_rate")

    pursuit = rates[rates["condition"] == "Pursuit"]
    hazard = rates[rates["condition"] == "Hazard"]

    assert pursuit["success_rate"].isna().all()
    assert not pursuit["success_rate"].eq(0).any()
    assert hazard["success_rate"].notna().all()


def test_unavailable_safety_counts_stay_unavailable():
    recorder = render(build_results(missing_role="FINAL_OOD_PURSUIT", complete=False))
    rates = recorder.frame_with("success_rate")
    pursuit = rates[rates["condition"] == "Pursuit"]

    assert pursuit["deaths"].isna().all()
    assert pursuit["timeouts"].isna().all()


# --------------------------------------------------------------------------- #
# Protocol endpoint gating
# --------------------------------------------------------------------------- #


def test_recurrent_protocol_rejects_a_one_million_checkpoint():
    """An earlier checkpoint is a different measurement, not a weaker final one."""
    recorder = render(
        build_results(protocol_id=RECURRENT_PROTOCOL_ID, checkpoint_step=1_000_000)
    )
    text = recorder.text("error")

    assert "Endpoint mismatch" in text
    assert "2,000,000" in text
    assert recorder.frame_with("passes")["passes"].isna().all()


def test_recurrent_protocol_accepts_its_own_endpoint():
    recorder = render(build_results(protocol_id=RECURRENT_PROTOCOL_ID))

    assert not recorder.messages["error"]
    assert recorder.frame_with("passes")["passes"].eq(True).all()


def test_legacy_protocol_keeps_its_own_endpoint_requirement():
    """Widening the recurrent rule must not weaken the historical validator."""
    recorder = render(
        build_results(protocol_id=LEGACY_PROTOCOL_ID, checkpoint_step=1_000_000)
    )

    assert not recorder.messages["error"]
    assert recorder.frame_with("passes")["passes"].eq(True).all()


def test_legacy_protocol_rejects_the_recurrent_endpoint():
    recorder = render(
        build_results(protocol_id=LEGACY_PROTOCOL_ID, checkpoint_step=2_000_000)
    )

    assert "Endpoint mismatch" in recorder.text("error")


def test_unsupported_protocol_claim_shows_no_assessment():
    recorder = render(build_results(protocol_id="invented-protocol-v9"))

    assert "Unsupported" in recorder.text("warning")
    assert not recorder.frames


# --------------------------------------------------------------------------- #
# Complete cohort
# --------------------------------------------------------------------------- #


def test_complete_cohort_reports_a_verdict_and_names_failed_criteria():
    failing = {"FINAL_ID": 0.9, "FINAL_OOD_HAZARD": 0.4, "FINAL_OOD_OBSTACLE": 0.8,
               "FINAL_OOD_PURSUIT": 0.8, "FINAL_OOD_COMBINED": 0.6}
    recorder = render(build_results(rates=failing))
    models = recorder.frame_with("passes")

    assert models["passes"].eq(False).all()
    assert models["failed criteria"].str.contains("hazard").all()


def test_criteria_and_protocol_scope_are_stated_on_screen():
    recorder = render(build_results())
    text = recorder.text("caption")

    assert "70%" in text and "50%" in text and "35%" in text
    assert "25 percentage points" in text and "40 percentage points" in text
    assert "2,000,000" in text
    assert "200 cases" in text


def test_contrasts_are_displayed_without_being_recomputed():
    """The UI shows the exported inference; it must not invent its own."""
    recorder = render(build_results())
    contrasts = recorder.frame_with("p (Holm)")

    assert contrasts["p (Holm)"].iloc[0] == 0.09
    assert bool(contrasts["rejects at 0.05"].iloc[0]) is False
    assert contrasts["95% CI"].iloc[0] == [0.01, 0.15]
    assert "not recomputed here" in recorder.text("caption")


def test_seed_is_the_replicate_unit_in_the_caption():
    recorder = render(build_results())

    assert "episodes and repeated scenarios are not" in recorder.text("caption")


def test_export_limitations_are_surfaced():
    recorder = render(build_results())

    assert "independent replicates" in recorder.text("caption")
