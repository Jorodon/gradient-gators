"""Robustness view: what it shows, and more importantly what it refuses to show.

The risk this view carries is not a wrong number on screen - it is a
qualification printed from evidence that does not support one, or a complete
cohort reported as incomplete. So most of these tests assert an absence: no
verdict from synthetic data, no pass from an incomplete cohort, no final claim
from a checkpoint the protocol did not prescribe, and a blank cell rather than a
zero for a condition nobody finished evaluating.

The first version of this suite was built entirely from fixtures shaped by hand,
and every test passed while the view was broken against a real released bundle -
the fixtures encoded the same wrong assumption as the code. So the completeness
and verdict paths are now driven by the actual downloaded release where it is
available, and hand-built records are used only for the states a correct release
cannot exhibit.
"""

import pandas as pd
import pytest

from research_support.dashboard_reference.protocols import (
    LEGACY_PROTOCOL_ID,
    PRIMARY_SEEDS,
    RECURRENT_PROTOCOL_ID,
)
from src.evaluation.dashboard_app import _render_robustness

FINAL_CONDITIONS = (
    "FINAL_ID",
    "FINAL_OOD_HAZARD",
    "FINAL_OOD_OBSTACLE",
    "FINAL_OOD_PURSUIT",
    "FINAL_OOD_COMBINED",
)

RELEASE_DIR = "docs/research/dashboard/downloaded_releases"


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

    def frame_with(self, column):
        for frame in self.frames:
            if column in getattr(frame, "columns", ()):
                return frame
        return None


class Artifacts:
    def __init__(self, scientific_results, episodes=None):
        self.scientific_results = scientific_results
        self.episodes = pd.DataFrame() if episodes is None else episodes


def render(artifacts):
    recorder = Recorder()
    _render_robustness(recorder, artifacts)
    return recorder


# --------------------------------------------------------------------------- #
# Hand-built records, for states a correct release cannot exhibit
# --------------------------------------------------------------------------- #


def episode_rows(*, checkpoint=2_000_000, counts=None, successes=None,
                 origin="measured", rewards=("sparse",), seeds=PRIMARY_SEEDS):
    """Normalized episode records of the shape the dashboard loader produces."""
    counts = counts or {role: 200 for role in FINAL_CONDITIONS}
    successes = successes or {role: 180 for role in FINAL_CONDITIONS}
    rows = []

    for reward in rewards:
        for seed in seeds:
            for role in FINAL_CONDITIONS:
                rows.append({
                    "reward_type": reward, "training_seed": seed,
                    "dataset_role": role, "episode_count": counts[role],
                    "success_count": successes[role], "death_count": 4,
                    "timeout_count": 2, "mean_damage": 12.5,
                    "checkpoint_timestep": checkpoint, "data_origin": origin,
                })

    return pd.DataFrame(rows)


def results(*, protocol_id=RECURRENT_PROTOCOL_ID, complete=True, budget=2_000_000,
            missing_roles=(), fixture=None):
    return {
        "protocol_id": protocol_id,
        "training_budget": budget,
        "fixture": fixture,
        "missing_final_roles": list(missing_roles),
        "inventory": {
            "complete": complete, "expected_models": 175,
            "observed_models": 175 if complete else 150,
            "missing_models": [] if complete else ["sparse_s4"],
            "missing_dataset_roles": [],
        },
        "comparisons": [{
            "reward": "efficiency", "reference": "sparse",
            "role": "FINAL_OOD_COMBINED", "n_pairs": 5,
            "mean_difference": 0.08, "ci95": [0.01, 0.15],
            "p_two_sided": 0.03, "p_holm": 0.09,
            "reject_holm_05": False, "reason": None,
        }],
    }


# --------------------------------------------------------------------------- #
# Absent evidence
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("payload", [None, {}])
def test_absent_results_render_pending_and_no_table(payload):
    """Pending is a state, not an error, and it must not render a table."""
    recorder = render(Artifacts(payload) if payload is not None else None)

    assert "Pending" in recorder.text("info")
    assert not recorder.frames


def test_pending_message_refuses_development_data_as_a_substitute():
    recorder = render(None)

    assert "development results cannot stand in" in recorder.text("info")


def test_results_without_final_records_do_not_render_a_verdict():
    """A bundle can be loaded and still carry no final conditions."""
    recorder = render(Artifacts(results(), pd.DataFrame()))

    assert "No final-condition records" in recorder.text("info")
    assert not recorder.frames


# --------------------------------------------------------------------------- #
# Synthetic evidence
# --------------------------------------------------------------------------- #


def test_synthetic_origin_is_labelled_and_yields_no_verdict():
    recorder = render(Artifacts(results(), episode_rows(origin="synthetic")))

    assert "Synthetic example" in recorder.text("warning")
    assert recorder.frame_with("passes")["passes"].isna().all()
    assert recorder.frame_with("family passes")["family passes"].isna().all()


def test_declared_fixture_flag_also_suppresses_the_verdict():
    recorder = render(Artifacts(results(fixture=True), episode_rows()))

    assert "Synthetic example" in recorder.text("warning")
    assert recorder.frame_with("passes")["passes"].isna().all()


def test_synthetic_data_does_not_also_claim_incompleteness():
    recorder = render(
        Artifacts(results(complete=False), episode_rows(origin="synthetic"))
    )

    assert "Incomplete evaluation" not in recorder.text("warning")


# --------------------------------------------------------------------------- #
# Incomplete evaluation
# --------------------------------------------------------------------------- #


def test_incomplete_inventory_shows_coverage_but_withholds_the_pass():
    recorder = render(Artifacts(results(complete=False), episode_rows()))
    text = recorder.text("warning")

    assert "Incomplete evaluation" in text
    assert "150 of 175" in text
    assert recorder.frame_with("passes")["passes"].isna().all()


def test_missing_final_condition_is_reported_as_incomplete():
    recorder = render(
        Artifacts(results(missing_roles=["FINAL_OOD_PURSUIT"]), episode_rows())
    )

    assert "FINAL_OOD_PURSUIT" in recorder.text("warning")


def test_partially_evaluated_condition_is_blank_not_zero():
    """Missing is not zero - a blank cell and a 0% score mean opposite things."""
    counts = {role: 200 for role in FINAL_CONDITIONS}
    counts["FINAL_OOD_PURSUIT"] = 120
    recorder = render(Artifacts(results(complete=False), episode_rows(counts=counts)))

    rates = recorder.frame_with("success_rate")
    pursuit = rates[rates["condition"] == "Pursuit"]
    hazard = rates[rates["condition"] == "Hazard"]

    assert pursuit["success_rate"].isna().all()
    assert not pursuit["success_rate"].eq(0).any()
    assert hazard["success_rate"].notna().all()


def test_partial_condition_still_reports_its_counts():
    """Blanking the rate must not hide how much was actually evaluated."""
    counts = {role: 200 for role in FINAL_CONDITIONS}
    counts["FINAL_OOD_PURSUIT"] = 120
    recorder = render(Artifacts(results(complete=False), episode_rows(counts=counts)))

    rates = recorder.frame_with("success_rate")
    pursuit = rates[rates["condition"] == "Pursuit"]

    assert pursuit["completed"].eq(120).all()
    assert pursuit["expected"].eq(200).all()


# --------------------------------------------------------------------------- #
# Protocol endpoint gating
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("checkpoint", [1_000_000, None])
def test_recurrent_protocol_rejects_an_unverified_final_checkpoint(checkpoint):
    """An earlier checkpoint is a different measurement, not a weaker final one."""
    recorder = render(
        Artifacts(results(), episode_rows(checkpoint=checkpoint))
    )
    text = recorder.text("error")

    assert "Endpoint mismatch" in text
    assert "2,000,000" in text
    assert recorder.frame_with("passes")["passes"].isna().all()


def test_recurrent_protocol_accepts_its_own_endpoint():
    recorder = render(Artifacts(results(), episode_rows()))

    assert not recorder.messages["error"]
    assert recorder.frame_with("passes")["passes"].notna().all()


def test_legacy_protocol_keeps_its_own_endpoint_requirement():
    """Widening the recurrent rule must not weaken the historical validator."""
    recorder = render(
        Artifacts(
            results(protocol_id=LEGACY_PROTOCOL_ID, budget=1_000_000),
            episode_rows(checkpoint=1_000_000),
        )
    )

    assert not recorder.messages["error"]


def test_legacy_protocol_rejects_the_recurrent_endpoint():
    recorder = render(
        Artifacts(
            results(protocol_id=LEGACY_PROTOCOL_ID, budget=1_000_000),
            episode_rows(checkpoint=2_000_000),
        )
    )

    assert "Endpoint mismatch" in recorder.text("error")


def test_unsupported_protocol_claim_shows_no_assessment():
    recorder = render(Artifacts(results(protocol_id="invented-v9"), episode_rows()))

    assert "Unsupported" in recorder.text("warning")
    assert not recorder.frames


# --------------------------------------------------------------------------- #
# Presentation rules
# --------------------------------------------------------------------------- #


def test_criteria_and_protocol_scope_are_stated_on_screen():
    recorder = render(Artifacts(results(), episode_rows()))
    text = recorder.text("caption")

    assert "70%" in text and "50%" in text and "35%" in text
    assert "25 percentage points" in text and "40 percentage points" in text
    assert "2,000,000" in text and "200 cases" in text


def test_damage_is_not_offered_as_evidence_of_safe_navigation():
    recorder = render(Artifacts(results(), episode_rows()))

    assert "low damage alone is not" in recorder.text("caption")


def test_contrasts_are_displayed_without_being_recomputed():
    """The UI shows the exported inference; it must not invent its own."""
    recorder = render(Artifacts(results(), episode_rows()))
    contrasts = recorder.frame_with("p (Holm)")

    assert contrasts["p (Holm)"].iloc[0] == 0.09
    assert bool(contrasts["rejects at 0.05"].iloc[0]) is False
    assert contrasts["nominal 95% CI (pp)"].iloc[0] == [1.0, 15.0]
    assert "not recomputed here" in recorder.text("caption")


def test_seed_is_the_replicate_unit_in_the_caption():
    recorder = render(Artifacts(results(), episode_rows()))

    assert "episodes and repeated scenarios are not" in recorder.text("caption")


# --------------------------------------------------------------------------- #
# The downloaded release
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def release_artifacts():
    """The published frozen study, when it has been downloaded.

    Skipped rather than failed when absent: the release is deliberately not
    committed, so a checkout without it is a normal state. These are the tests
    the hand-built records above cannot replace, because the defect they exist
    to catch was a mismatch between assumed and actual bundle shape.
    """
    from pathlib import Path

    from src.evaluation.dashboard_data import load_dashboard_artifacts

    root = Path(__file__).resolve().parents[1] / RELEASE_DIR / "recurrent_ppo_frozen_protocol_v1_results"

    if not root.is_dir() or not any(root.iterdir()):
        pytest.skip(f"frozen study release not downloaded to {RELEASE_DIR}")

    artifacts = load_dashboard_artifacts(root)

    if not artifacts.scientific_results:
        pytest.skip("downloaded release carries no scientific results")

    return artifacts


def test_released_study_is_not_reported_as_incomplete(release_artifacts):
    """The defect that shipped: a complete cohort rendered as incomplete."""
    recorder = render(release_artifacts)

    assert "Incomplete evaluation" not in recorder.text("warning")
    assert not recorder.messages["error"]


def test_released_study_populates_every_column(release_artifacts):
    """Reward, seed and success rate were all blank against the real bundle."""
    rates = render(release_artifacts).frame_with("success_rate")

    assert rates is not None and len(rates) == 125
    assert rates["reward"].notna().all()
    assert rates["seed"].notna().all()
    assert rates["success_rate"].notna().all()
    assert rates["expected"].eq(200).all()


def test_released_study_produces_per_model_and_family_verdicts(release_artifacts):
    recorder = render(release_artifacts)
    models = recorder.frame_with("failed criteria")
    families = recorder.frame_with("family passes")

    assert len(models) == 25
    assert models["fully assessed"].all()
    assert models["passes"].notna().all()
    assert len(families) == 5
    assert families["fully assessed"].eq("5 of 5").all()


def test_released_study_verdicts_match_the_recorded_rates(release_artifacts):
    """Spot-check one model end to end, so a wrong join would be visible."""
    recorder = render(release_artifacts)
    rates = recorder.frame_with("success_rate")
    models = recorder.frame_with("failed criteria")

    sparse = rates[(rates["reward"] == "sparse") & (rates["seed"] == PRIMARY_SEEDS[0])]
    identity = sparse[sparse["condition"] == "ID"]["success_rate"].iloc[0]

    assert identity == 69.5

    verdict = models[(models["reward"] == "sparse") & (models["seed"] == PRIMARY_SEEDS[0])]

    assert bool(verdict["passes"].iloc[0]) is False
    assert "ID at least 70%" in verdict["failed criteria"].iloc[0]


def test_released_study_contrasts_carry_holm_adjusted_values(release_artifacts):
    recorder = Recorder()
    view = _render_robustness(recorder, release_artifacts)
    contrasts = recorder.frame_with("p (Holm)")

    assert len(contrasts) == 4
    assert contrasts["p (Holm)"].notna().all()
    assert contrasts["paired seeds"].eq(5).all()
    assert len(view["tables"]["paired_seed_differences"]) == 20
    assert "Full frozen cohort" in recorder.text("caption")
    for saved in release_artifacts.scientific_results["comparisons"]:
        row = contrasts[contrasts["reward"] == saved["reward"]].iloc[0]
        assert row["nominal 95% CI (pp)"] == pytest.approx([100 * value for value in saved["ci95_nominal"]])
        pairs = view["tables"]["paired_seed_differences"]
        paired = pairs[pairs["reward"] == saved["reward"]]
        assert paired["difference (pp)"].tolist() == pytest.approx([100 * pair["difference"] for pair in saved["pairs"]])
        assert paired["seed"].tolist() == [pair["seed"] for pair in saved["pairs"]]


def test_historical_final_coverage_is_checked_when_inventory_is_not_supplied():
    '''
    Uses the historical protocol's required final models instead of assuming a missing inventory is complete.
    '''
    from research_support.dashboard_reference.protocols import REWARDS

    payload = results(protocol_id=LEGACY_PROTOCOL_ID, budget=1_000_000)
    payload["inventory"] = None
    frame = episode_rows(checkpoint=1_000_000, rewards=REWARDS)
    recorder = render(Artifacts(payload, frame))
    assert "Incomplete evaluation" not in recorder.text("warning")
    assert recorder.frame_with("passes")["passes"].notna().all()

    partial = render(Artifacts(payload, frame.iloc[1:]))
    assert "Incomplete evaluation" in partial.text("warning")
    assert partial.frame_with("passes")["passes"].isna().all()
