"""Tests for frozen checkpoint evaluation metrics / validation contracts."""

import pytest

from src.evaluation.checkpoint_evaluation import (
    calculate_normalized_aulc,
    compute_learning_metrics,
    time_to_sustained_threshold,
    validate_checkpoint_validation_manifest,
)


def _manifest(count: int = 100, split: str = "CHECKPOINT_VALIDATION") -> dict:
    """Builds minimal manifest for validation contract tests."""
    entries = [
        {
            "scenario_id": f"scenario-{index}",
            "map_id": f"map-{index}",
            "checksum": f"checksum-{index}",
            "area_band": "small",
        }
        for index in range(count)
    ]
    return {
        "collection_id": "validation",
        "split": split,
        "entries": entries,
        "checksum": "manifest-checksum",
    }


def _checkpoint_rows(*success_counts: int) -> list[dict[str, float | int]]:
    """Builds aggregate rows w/ the production checkpoint schema."""
    return [
        {
            "checkpoint_timestep": index * 25_000,
            "scenario_count": 100,
            "success_count": count,
            "success_rate": count / 100,
        }
        for index, count in enumerate(success_counts)
    ]


def test_validation_manifest_requires_100_unique_held_out_scenarios():
    """Requires exactly 100 checkpoint validation scenarios."""
    assert len(validate_checkpoint_validation_manifest(_manifest())) == 100

    with pytest.raises(ValueError, match="exactly 100"):
        validate_checkpoint_validation_manifest(_manifest(99))

    with pytest.raises(ValueError, match="CHECKPOINT_VALIDATION"):
        validate_checkpoint_validation_manifest(_manifest(split="TRAIN"))


def test_validation_manifest_rejects_train_overlap():
    """Rejects scenario or map IDs shared w/ the training collection."""
    train = _manifest()
    validation = _manifest()
    train["entries"] = [train["entries"][0]]

    with pytest.raises(ValueError, match="scenario IDs overlap"):
        validate_checkpoint_validation_manifest(validation, train_manifest=train)


def test_aulc_uses_observed_points_without_imputing_missing_checkpoints():
    """Calculates normalized trapezoids using only observed checkpoints."""
    points = [(0, 0.0), (50_000, 0.4), (100_000, 0.8)]

    assert calculate_normalized_aulc(points, training_budget=100_000) == pytest.approx(0.4)


def test_threshold_requires_two_scheduled_checkpoints():
    """Returns second qualifying checkpoint (only when checkpoints are adjacent)."""
    rows = _checkpoint_rows(0, 80, 79, 80, 80)

    assert time_to_sustained_threshold(rows, scenario_count=100) == 100_000

    missing_checkpoint = [rows[0], rows[1], rows[3]]
    assert time_to_sustained_threshold(missing_checkpoint, scenario_count=100) is None


def test_learning_metrics_mark_nonattainment_as_censored():
    """Records censoring when  threshold is never reached twice."""
    metrics = compute_learning_metrics(
        _checkpoint_rows(0, 70, 79, 79),
        scenario_count=100,
        training_budget=75_000,
    )

    assert metrics["time_to_80_successes"] is None
    assert metrics["time_to_80_status"] == "censored"
    assert metrics["censoring_timestep"] == 75_000

@pytest.mark.parametrize("omitted", [0, 1, 2])
def test_incomplete_curve_is_not_reported_as_full_budget_aulc(omitted):
    rows = _checkpoint_rows(0, 80, 90)
    del rows[omitted]
    metrics = compute_learning_metrics(rows, scenario_count=100, training_budget=50_000)
    assert metrics["curve_complete"] is False
    assert metrics["missing_checkpoint_timesteps"] == [omitted * 25_000]
    assert metrics["normalized_aulc"] is None
    assert metrics["time_to_80_status"] == "incomplete"