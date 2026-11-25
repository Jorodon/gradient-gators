"""Results view: coverage, seed variation and denominators.

The failure this suite is built to catch is not an exception. It is a number
that renders confidently and means something other than what the reader will
take it to mean: a success rate computed over a cohort nobody finished, a zero
standing in for a missing measurement, training and final conditions added
together into one percentage, or two different invalid-move denominators shown
under one label.

The reconciliation tests read the published release's own JSON and compare it
with what the view rendered. That direction matters. An earlier dashboard view
in this repository passed a full suite of hand-built fixtures while being
broken against the real bundle, because the fixtures were written from the same
mistaken assumption as the code. Fixtures here cover only the states a correct
release cannot exhibit - absent fields, a single seed, a half-finished
condition - and every claim a real release can check is checked against it.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.dashboard_app import (
    _primary_role,
    _render_results,
    _render_safety_panel,
    _render_success_panel,
    _result_records,
)

RELEASE_ROOT = Path("docs/research/dashboard/downloaded_releases")
RELEASE_RESULTS = RELEASE_ROOT / "recurrent_ppo_frozen_protocol_v1_results"


class Recorder:
    """Minimal stand-in for the Streamlit module, capturing what was rendered."""

    TEXT_KINDS = ("info", "warning", "error", "caption", "markdown", "subheader")

    def __init__(self, choice=None):
        self.messages = {kind: [] for kind in self.TEXT_KINDS}
        self.frames = []
        self.charts = []
        self.choice = choice
        self.offered = []

    def _record(self, kind):
        def capture(text, *args, **kwargs):
            self.messages[kind].append(str(text))
        return capture

    def __getattr__(self, name):
        if name in self.TEXT_KINDS:
            return self._record(name)
        raise AttributeError(name)

    def selectbox(self, label, options, *args, **kwargs):
        options = list(options)
        self.offered.append((label, options))
        if self.choice is not None and self.choice in options:
            return self.choice
        return options[0] if options else None

    def dataframe(self, frame, *args, **kwargs):
        self.frames.append(frame)

    def bar_chart(self, frame, *args, **kwargs):
        self.charts.append(frame)

    def text(self, kind):
        return " ".join(self.messages[kind])

    def all_text(self):
        return " ".join(self.text(kind) for kind in self.TEXT_KINDS)

    def frame_with(self, column):
        for frame in self.frames:
            if column in getattr(frame, "columns", ()):
                return frame
        return None

    def row_for(self, column, reward):
        """Returns the row of the first frame carrying `column` for a reward."""
        frame = self.frame_with(column)
        assert frame is not None, f"no rendered table has a {column!r} column"
        matches = frame[frame["Reward"] == reward]
        assert not matches.empty, f"no row for reward {reward!r}"
        return matches.iloc[0]


class Artifacts:
    def __init__(self, scientific_results, source_id="test-source"):
        self.scientific_results = scientific_results
        self.source_id = source_id
        self.episodes = pd.DataFrame()


def outcome(
    reward,
    seed,
    role="FINAL_ID",
    expected=200,
    completed=200,
    successes=120,
    checkpoint=2_000_000,
    metrics=None,
    **overrides,
):
    """Builds one exported outcome record."""
    source_metrics = {
        "reward": reward,
        "seed": seed,
        "death_count": 10,
        "timeout_count": 20,
        "fall_damage_mean": 4.0,
        "hazard_damage_mean": 3.0,
        "enemy_damage_mean": 2.0,
        "final_hp_mean": 80.0,
        "illegal_moves_fraction_pooled": 0.5,
        "illegal_moves_fraction_episode_mean": 0.2,
        "zero_damage_success_count": 40,
        "raw_return_mean": 12.5,
    }
    if metrics is not None:
        source_metrics.update(metrics)
        for key, value in list(source_metrics.items()):
            if value is _ABSENT:
                del source_metrics[key]

    record = {
        "model_id": f"{reward}_s{seed}",
        "role": role,
        "checkpoint_step": checkpoint,
        "n_expected": expected,
        "n_completed": completed,
        "success_count": successes,
        "death_count": 10,
        "timeout_count": 20,
        "mean_damage": 9.0,
        "source_metrics": source_metrics,
    }
    record.update(overrides)
    return record


class _Absent:
    """Marker for a field a release does not publish at all."""


_ABSENT = _Absent()


def results(outcomes, protocol_id="recurrent-complete-900-2m-v1", **extra):
    payload = {"protocol_id": protocol_id, "outcomes": list(outcomes)}
    payload.update(extra)
    return payload


def render(scientific_results, choice=None):
    recorder = Recorder(choice=choice)
    _render_results(recorder, Artifacts(scientific_results))
    return recorder


# --- states a correct release cannot exhibit -------------------------------


def test_incomplete_condition_withholds_the_rate_instead_of_reporting_zero():
    """A cohort nobody finished has no success rate - and it is not zero."""
    recorder = render(results([
        outcome("sparse", seed, completed=120, successes=60)
        for seed in range(5)
    ]))

    row = recorder.row_for("Success rate", "sparse")
    assert row["Success rate"] == "Unavailable"
    assert "0.0%" not in row["Success rate"]
    assert row["Completed / expected"] == "600 / 1,000"


def test_incomplete_condition_still_reports_its_coverage():
    """Withholding the rate must not also hide how much was evaluated."""
    recorder = render(results([
        outcome("sparse", seed, completed=120, successes=60)
        for seed in range(5)
    ]))

    row = recorder.row_for("Successes / completed", "sparse")
    assert row["Successes / completed"] == "300 / 600"


def test_absent_safety_measure_reads_unavailable_not_zero():
    """A field the release never published is missing, not a measurement of 0."""
    recorder = render(results([
        outcome("sparse", seed, metrics={"fall_damage_mean": _ABSENT})
        for seed in range(5)
    ]))

    row = recorder.row_for("Fall damage", "sparse")
    assert row["Fall damage"] == "Unavailable"
    assert row["Hazard damage"] == "3.00 HP"


def test_partially_reported_measure_is_not_aggregated_from_the_models_that_have_it():
    """Three of five seeds reporting a field does not make a cohort figure."""
    partial = [outcome("sparse", seed) for seed in range(3)]
    partial += [
        outcome("sparse", seed, metrics={"final_hp_mean": _ABSENT})
        for seed in (3, 4)
    ]

    row = render(results(partial)).row_for("Final HP", "sparse")
    assert row["Final HP"] == "Unavailable"


def test_single_seed_claims_no_spread_across_training_seeds():
    """One policy cannot produce a spread over independent training seeds."""
    recorder = render(results([outcome("sparse", 1)]))

    assert "Single independent seed for: sparse" in recorder.text("caption")
    assert recorder.row_for("Seeds", "sparse")["Seeds"] == 1


def test_multiple_seeds_do_not_trigger_the_single_seed_caption():
    """The caption must track the records, not appear unconditionally."""
    recorder = render(results([outcome("sparse", seed) for seed in range(5)]))

    assert "Single independent seed" not in recorder.text("caption")


def test_repeated_episodes_do_not_count_as_additional_training_seeds():
    """Seed count follows distinct seeds, not the number of evaluated episodes."""
    repeated = [
        outcome("sparse", 7, expected=200, completed=200),
        outcome("sparse", 7, expected=200, completed=200),
        outcome("sparse", 7, expected=200, completed=200),
    ]

    row = render(results(repeated)).row_for("Seeds", "sparse")
    assert row["Seeds"] == 1


def test_conditions_are_never_pooled_into_one_success_figure():
    """TRAIN and FINAL_ID are different populations and stay separate."""
    mixed = [outcome("sparse", seed, role="FINAL_ID", successes=100)
             for seed in range(5)]
    mixed += [outcome("sparse", seed, role="TRAIN", successes=180)
              for seed in range(5)]

    final = render(results(mixed), choice="FINAL_ID")
    train = render(results(mixed), choice="TRAIN")

    assert final.row_for("Successes / completed", "sparse")[
        "Successes / completed"] == "500 / 1,000"
    assert train.row_for("Successes / completed", "sparse")[
        "Successes / completed"] == "900 / 1,000"
    # Pooling the two would give 1,400 / 2,000; neither view may show it.
    for recorder in (final, train):
        rendered = recorder.frame_with("Successes / completed")
        assert "1,400 / 2,000" not in rendered.to_string()


def test_every_condition_in_the_records_is_offered():
    """The reader can reach each condition; none is silently dropped."""
    mixed = [outcome("sparse", 1, role=role)
             for role in ("FINAL_ID", "TRAIN", "FINAL_OOD_PURSUIT")]

    _, options = render(results(mixed)).offered[0]
    assert set(options) == {"FINAL_ID", "TRAIN", "FINAL_OOD_PURSUIT"}


def test_counts_are_shown_against_their_denominator():
    """Deaths without a denominator is a number nobody can interpret."""
    row = render(results([
        outcome("sparse", seed) for seed in range(5)
    ])).row_for("Deaths", "sparse")

    assert row["Deaths"] == "50 / 1,000"


def test_both_invalid_move_denominators_are_reported_separately():
    """Pooled and per-episode fractions answer different questions."""
    recorder = render(results([outcome("sparse", seed) for seed in range(5)]))
    frame = recorder.frame_with("Invalid moves (pooled over transitions)")

    assert frame is not None
    assert "Invalid moves (mean per-episode fraction)" in frame.columns
    row = frame[frame["Reward"] == "sparse"].iloc[0]
    assert row["Invalid moves (pooled over transitions)"] == "0.5000"
    assert row["Invalid moves (mean per-episode fraction)"] == "0.2000"


def test_safe_success_is_presented_with_completion():
    """Low damage alone is not evidence of safe navigation."""
    recorder = render(results([outcome("sparse", seed) for seed in range(5)]))
    row = recorder.row_for("Safe success rate", "sparse")

    assert row["Safe success rate"] == "20.0%"
    assert row["Success rate"] == "60.0%"


def test_raw_return_is_marked_as_incomparable_across_rewards():
    """Return scales differ by reward definition; the view must say so."""
    recorder = render(results([
        outcome(reward, seed)
        for reward in ("sparse", "safety")
        for seed in range(5)
    ]))

    caption = recorder.text("caption")
    assert "comparable only within" in caption
    assert "never as a ranking across rewards" in caption


def test_mismatched_training_budgets_are_labelled_not_silently_compared():
    """Two endpoints in one table is a comparison caveat, not a detail."""
    mixed = [outcome("sparse", seed, checkpoint=2_000_000) for seed in range(3)]
    mixed += [outcome("safety", seed, checkpoint=1_000_000) for seed in range(3)]

    warning = render(results(mixed)).text("warning")
    assert "mixed training budgets" in warning


def test_unequal_seed_counts_across_rewards_are_labelled():
    """Five seeds against two is not a like-for-like comparison."""
    mixed = [outcome("sparse", seed) for seed in range(5)]
    mixed += [outcome("safety", seed) for seed in range(2)]

    assert "unequal independent-seed counts" in render(results(mixed)).text("warning")


def test_a_second_cohort_drives_its_own_counts_and_labels():
    """Nothing about the frozen study may be baked into the table."""
    legacy = [
        outcome("sparse", seed, expected=100, completed=100, successes=55,
                checkpoint=1_000_000)
        for seed in range(3)
    ]

    recorder = render(results(legacy, protocol_id="m4-common-ppo-v1"))
    row = recorder.row_for("Completed / expected", "sparse")

    assert row["Completed / expected"] == "300 / 300"
    assert row["Seeds"] == 3
    assert row["Success rate"] == "55.0%"
    caption = recorder.text("caption")
    assert "m4-common-ppo-v1" in caption
    assert "1,000,000 consumed interactions" in caption
    assert "Independent training seeds: 3 per reward" in caption


def test_results_without_exported_outcomes_say_so():
    """An artifact set without summaries renders a message, not an empty table."""
    recorder = render({"protocol_id": "recurrent-complete-900-2m-v1"})

    assert "not available" in recorder.text("info")
    assert recorder.frames == []


def render_panels(scientific_results):
    """Renders the two headline panels the dashboard shows above the table."""
    records = _result_records(Artifacts(scientific_results))
    role = _primary_role(records)
    success, safety = Recorder(), Recorder()
    _render_success_panel(success, records, role)
    _render_safety_panel(safety, records, role)
    return success, safety


def test_headline_chart_omits_rewards_whose_cohort_is_incomplete():
    """A bar drawn from a partial denominator is a claim nobody can support."""
    mixed = [outcome("sparse", seed) for seed in range(5)]
    mixed += [outcome("safety", seed, completed=100) for seed in range(5)]

    success, _ = render_panels(results(mixed))
    charted = success.charts[0] if success.charts else None

    assert charted is not None
    assert "sparse" in charted.index
    assert "safety" not in charted.index
    assert "Withheld as incomplete: safety" in success.text("caption")


def test_headline_chart_is_withheld_when_no_cohort_is_complete():
    """With nothing fully evaluated there is no bar to draw, and no zero."""
    success, _ = render_panels(results([
        outcome("sparse", seed, completed=100) for seed in range(5)
    ]))

    assert success.charts == []
    assert "No reward has a fully evaluated cohort" in success.text("info")


def test_headline_panels_describe_the_condition_they_show():
    """A bar chart without its condition invites the pooled reading."""
    success, safety = render_panels(results([
        outcome("sparse", seed) for seed in range(5)
    ]))

    assert "Condition: FINAL_ID" in success.text("caption")
    assert "Condition: FINAL_ID" in safety.text("caption")
    assert "not evidence of safe navigation" in safety.text("caption")


def test_safety_panel_reports_nothing_when_damage_is_absent():
    """An absent measure leaves the panel empty rather than drawing zeros."""
    _, safety = render_panels(results([
        outcome("sparse", seed, mean_damage=None) for seed in range(5)
    ]))

    assert safety.charts == []
    assert "not reported" in safety.text("info")


# --- reconciliation against the published release --------------------------


@pytest.fixture(scope="module")
def release_results():
    """Loads the published frozen study through the real dashboard loader."""
    if not RELEASE_RESULTS.exists():
        pytest.skip("Published recurrent results bundle not downloaded")
    from src.evaluation.dashboard_data import load_dashboard_artifacts

    return load_dashboard_artifacts(RELEASE_ROOT)


@pytest.fixture(scope="module")
def published_outcomes():
    """Reads the release's own JSON, independently of the dashboard loader."""
    if not RELEASE_RESULTS.exists():
        pytest.skip("Published recurrent results bundle not downloaded")
    payload = json.loads(
        (RELEASE_RESULTS / "research_results.json").read_text(encoding="utf-8")
    )
    return payload["outcomes"]


def render_release(artifacts, choice=None):
    recorder = Recorder(choice=choice)
    _render_results(recorder, artifacts)
    return recorder


def test_released_study_reconciles_with_its_own_published_counts(
    release_results, published_outcomes
):
    """Displayed successes and completions equal the release's own numbers."""
    recorder = render_release(release_results, choice="FINAL_ID")

    for reward in ("sparse", "safety", "balanced", "efficiency", "progress"):
        published = [
            record for record in published_outcomes
            if record["role"] == "FINAL_ID"
            and record["source_metrics"]["reward"] == reward
        ]
        successes = sum(record["success_count"] for record in published)
        completed = sum(record["n_completed"] for record in published)

        row = recorder.row_for("Successes / completed", reward)
        assert row["Successes / completed"] == f"{successes:,} / {completed:,}"


def test_released_study_reports_five_independent_seeds_per_reward(release_results):
    """The frozen protocol's independent sample size is five, not 1,000."""
    recorder = render_release(release_results, choice="FINAL_ID")
    frame = recorder.frame_with("Seeds")

    assert set(frame["Seeds"]) == {5}
    assert "Independent training seeds: 5 per reward" in recorder.text("caption")


def test_released_study_declares_its_protocol_and_endpoint(release_results):
    """No rate is readable without the scope that produced it."""
    caption = render_release(release_results, choice="FINAL_ID").text("caption")

    assert "recurrent-complete-900-2m-v1" in caption
    assert "2,000,000 consumed interactions" in caption
    assert "Condition: FINAL_ID" in caption


def test_released_study_is_a_comparable_cohort(release_results):
    """The frozen study is internally consistent; no caveat should appear."""
    recorder = render_release(release_results, choice="FINAL_ID")

    assert recorder.text("warning") == ""


def test_released_study_populates_every_results_column(release_results):
    """A shape mismatch shows up as blanks; this is the test that catches it."""
    recorder = render_release(release_results, choice="FINAL_ID")
    frame = recorder.frame_with("Success rate")

    assert len(frame) == 5
    for column in ("Successes / completed", "Completed / expected",
                   "Success rate", "Safe success rate"):
        values = frame[column].tolist()
        assert all(value not in ("", None) for value in values)
        assert "Unavailable" not in values


def test_released_study_reports_every_safety_measure(release_results):
    """Every safety field the release publishes must reach the table."""
    recorder = render_release(release_results, choice="FINAL_ID")
    frame = recorder.frame_with("Fall damage")

    assert frame is not None
    for column in ("Deaths", "Timeouts", "Mean damage", "Fall damage",
                   "Hazard damage", "Enemy damage", "Final HP",
                   "Invalid moves (pooled over transitions)",
                   "Invalid moves (mean per-episode fraction)"):
        assert "Unavailable" not in frame[column].tolist(), column


def test_released_study_reports_one_seed_row_per_model(release_results):
    """Twenty-five policies produce twenty-five rows, not one per episode."""
    recorder = render_release(release_results, choice="FINAL_ID")
    frame = recorder.frame_with("Seed")

    assert len(frame) == 25
    assert frame["Seed"].nunique() == 5


def test_released_final_and_train_conditions_differ(release_results):
    """If these matched, the view would be pooling conditions somewhere."""
    final = render_release(release_results, choice="FINAL_ID")
    train = render_release(release_results, choice="TRAIN")

    final_row = final.row_for("Completed / expected", "sparse")
    train_row = train.row_for("Completed / expected", "sparse")
    assert final_row["Completed / expected"] != train_row["Completed / expected"]
