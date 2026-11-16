"""Protocol metric regressions using fabricated saved-record data only."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research_support.dashboard_reference.metrics import research_metrics as mm
from research_support.dashboard_reference.protocols import LEGACY_PROTOCOL_ID, RECURRENT_PROTOCOL_ID, get_protocol
from research_support.dashboard_reference.tests.test_research_metrics import episode

PROTOCOL = RECURRENT_PROTOCOL_ID
GRID = get_protocol(PROTOCOL)["grid"]


def points(counts=(0, 80, 80, 60, 90, 40, 100), actual=None):
    actual = list(GRID) if actual is None else actual
    return [{"checkpoint_step": step, "actual_training_steps": consumed,
             "optimized_training_steps": max(0, consumed - 10),
             "n_expected": 100, "n_completed": 100, "success_count": count}
            for step, consumed, count in zip(GRID, actual, counts)]


def model(step=2_000_000, *, seed=mm.PRIMARY_SEEDS[0], reward="sparse", model_id=None):
    return {"model_id": model_id or f"{reward}_{seed}_{step}", "reward": reward,
            "seed": seed, "checkpoint_step": step, "actual_training_steps": step,
            "optimized_training_steps": max(0, step - 10)}


def dataset(role, count=1):
    return {"role": role, "scenario_count": count,
            "scenario_ids": [f"scenario_{i}" for i in range(count)],
            "manifest_path": "synthetic/manifest.json", "manifest_sha256": None}


class RecurrentMetricsTests(unittest.TestCase):
    def test_legacy_default_remains_exact(self):
        legacy = [{"checkpoint_step": t, "n_expected": 100, "n_completed": 100,
                   "success_count": count} for t, count in zip(mm.GRID, (0, 80, 80, 60, 90))]
        result = mm.learning_metrics(legacy)
        self.assertEqual(result, mm.learning_metrics(legacy, protocol_id=LEGACY_PROTOCOL_ID))
        self.assertEqual(result["normalized_aulc"], .6625)
        self.assertEqual(result["time_to_80_successes"], 500_000)
        self.assertNotIn("coordinate", result)
        with self.assertRaises(ValueError):
            mm.learning_metrics(points())

    def test_uneven_actual_coordinates_control_area_and_attainment(self):
        # This triangle has width 250,000 under the nominal schedule but 250,100
        # in consumed interactions. Equal checkpoint weighting is also wrong.
        actual = [0, 100_500, 250_100, 500_900, 1_000_000, 1_500_800, 2_000_000]
        triangle = mm.learning_metrics(points((0, 100, 0, 0, 0, 0, 0), actual), protocol_id=PROTOCOL)
        self.assertAlmostEqual(triangle["normalized_aulc"], .062525)
        result = mm.learning_metrics(points(actual=actual), protocol_id=PROTOCOL)
        self.assertEqual(result["time_to_80_successes"], 250_100)
        self.assertEqual(result["coordinate"], "actual_consumed_interactions")
        self.assertEqual(result["checkpoint_coordinates"][2]["checkpoint_step"], 250_000)
        self.assertIsNone(result["censoring_timestep"])

    def test_threshold_requires_adjacent_positive_scheduled_checkpoints(self):
        for counts in ((100, 100, 0, 0, 0, 0, 0), (0, 80, 79, 80, 79, 80, 79)):
            result = mm.learning_metrics(points(counts), protocol_id=PROTOCOL)
            self.assertEqual(result["threshold_status"], "censored")
            self.assertIsNone(result["time_to_80_successes"])
            self.assertEqual(result["censoring_timestep"], 2_000_000)
        result = mm.learning_metrics(points((0, 0, 0, 0, 0, 80, 80)), protocol_id=PROTOCOL)
        self.assertEqual(result["time_to_80_successes"], 2_000_000)

    def test_missing_or_partial_is_incomplete_and_never_interpolated(self):
        for index in range(len(GRID)):
            missing = points()
            del missing[index]
            partial = points()
            partial[index].update(n_completed=80, success_count=80)
            for rows in (missing, partial):
                result = mm.learning_metrics(rows, protocol_id=PROTOCOL)
                self.assertEqual(result["missing_or_partial_checkpoints"], [GRID[index]])
                self.assertEqual(result["threshold_status"], "incomplete")
                self.assertIsNone(result["normalized_aulc"])
                self.assertIsNone(result["time_to_80_successes"])
                self.assertIsNone(result["censoring_timestep"])

    def test_checkpoint_windows_and_counter_accounting(self):
        rows = points()
        rows[1]["actual_training_steps"] += 1398
        self.assertIsNotNone(mm.learning_metrics(rows, protocol_id=PROTOCOL)["normalized_aulc"])
        invalid = ((0, "actual_training_steps", 1), (6, "actual_training_steps", 2_000_001),
                   (1, "actual_training_steps", 99_999), (1, "actual_training_steps", 101_399),
                   (1, "optimized_training_steps", 100_001), (1, "optimized_training_steps", -1),
                   (1, "actual_training_steps", 100_000.0), (1, "checkpoint_step", 750_000),
                   (1, "n_completed", 100.0), (1, "n_expected", 99),
                   (1, "final_unoptimized_steps", 11))
        for index, field, value in invalid:
            with self.subTest(field=field, value=value):
                bad = points()
                bad[index][field] = value
                with self.assertRaises(ValueError):
                    mm.learning_metrics(bad, protocol_id=PROTOCOL)
        for field in ("actual_training_steps", "optimized_training_steps"):
            bad = points()
            del bad[1][field]
            with self.assertRaises(ValueError):
                mm.learning_metrics(bad, protocol_id=PROTOCOL)
        with self.assertRaises(ValueError):
            mm.learning_metrics(points() + [points()[0]], protocol_id=PROTOCOL)

    def test_protocol_selection_is_explicit_and_unknown_ids_fail(self):
        calls = (
            lambda: mm.learning_metrics([], protocol_id="unknown"),
            lambda: mm.paired_contrasts([], [], protocol_id="unknown"),
            lambda: mm.family_robustness({}, protocol_id="unknown"),
            lambda: mm.coverage([], [], [], protocol_id="unknown"),
            lambda: mm.matched_scenarios([], protocol_id="unknown"),
            lambda: mm.select_traces([], [], [], protocol_id="unknown"),
            lambda: mm.build_export(study_id="synthetic", provenance={}, datasets=[],
                                    models=[], rows=[], fixture=True, protocol_id="unknown"))
        for call in calls:
            with self.assertRaises(ValueError):
                call()

    def test_contrasts_use_2m_five_seed_four_test_family(self):
        models, outcomes = [], []
        for step in (1_000_000, 2_000_000):
            for reward in mm.REWARDS:
                for index, seed in enumerate(mm.PRIMARY_SEEDS):
                    m = model(step, seed=seed, reward=reward)
                    models.append(m)
                    outcomes.append({"model_id": m["model_id"], "role": "FINAL_OOD_COMBINED",
                                     "complete": True, "success_fraction": (
                                         .9 if step == 1_000_000 else .2 if reward == "sparse" else .2 + .05 * index)})
        results = mm.paired_contrasts(outcomes, models, protocol_id=PROTOCOL)
        self.assertEqual(len(results), 4)
        for result in results:
            self.assertEqual((result["n_pairs"], result["df"]), (5, 4))
            self.assertAlmostEqual(result["t"], 2.82842712474619)
            self.assertAlmostEqual(result["p_two_sided"], .0474206555843196)
            self.assertAlmostEqual(result["ci95"][0], .00183784192612198)
            self.assertAlmostEqual(result["p_holm"], .1896826223372784)
        first_shaped = next(m["model_id"] for m in models if m["reward"] == "efficiency" and m["checkpoint_step"] == 2_000_000)
        missing = mm.paired_contrasts([o for o in outcomes if o["model_id"] != first_shaped], models, protocol_id=PROTOCOL)
        self.assertEqual(missing[0]["n_pairs"], 4)
        self.assertIsNone(missing[0]["p_holm"])
        self.assertAlmostEqual(missing[1]["p_holm"], .1896826223372784)
        self.assertEqual(mm.paired_contrasts(outcomes, models)[0]["reason"], "zero_variance_t_inference_undefined")

    def test_trace_endpoint_changes_but_outcome_blind_hash_does_not(self):
        datasets = [dataset("FINAL_ID", 5)]
        selected_ids = mm.matched_scenarios(datasets)
        self.assertEqual(selected_ids, mm.matched_scenarios(datasets, protocol_id=PROTOCOL))
        models = [model(1_000_000), model(2_000_000)]
        rows = [episode(sid, model=m["model_id"], role="FINAL_ID")
                for m in models for sid in selected_ids["FINAL_ID"]]
        selected, _ = mm.select_traces(rows, models, datasets, protocol_id=PROTOCOL)
        self.assertTrue(selected)
        self.assertEqual({r["model_id"] for r in selected}, {models[1]["model_id"]})
        self.assertEqual({r["model_id"] for r in mm.select_traces(rows, models, datasets)[0]}, {models[0]["model_id"]})

    def test_build_export_respects_learning_and_final_panel_schedule(self):
        from research_support.dashboard_reference.readers.results_adapter import adapt
        models = [model(step) for step in GRID]
        datasets = [dataset(role) for role in ("LEARNING_VALIDATION", "TRAIN", *mm.FINAL_ROLES)]
        rows = [episode("scenario_0", model=m["model_id"], role=d["role"])
                for m in models for d in datasets
                if d["role"] == "LEARNING_VALIDATION" or m["checkpoint_step"] == 2_000_000]
        result = mm.build_export(study_id="synthetic", provenance={}, datasets=datasets, models=models,
                                 rows=rows, fixture=True, protocol_id=PROTOCOL)
        self.assertEqual(result["schema_version"], "research-results-v2")
        self.assertEqual(result["protocol_id"], PROTOCOL)
        self.assertEqual(result["coverage"]["n_expected"], 13)
        self.assertTrue(result["coverage"]["complete"])
        self.assertEqual(len(result["learning"]), 7)
        self.assertEqual(len(result["robustness_models"]), 1)
        self.assertEqual(result["robustness_models"][0]["model_id"], models[-1]["model_id"])
        self.assertTrue(result["robustness_models"][0]["passes"])
        self.assertEqual(result["planned_final_coverage"]["n_completed_episodes"], 5)
        plan = result["planned_campaign_coverage"]
        self.assertEqual((plan["n_expected_model_checkpoints"], plan["n_expected_evaluation_panels"], plan["n_expected_episodes"]), (175, 325, 65_000))
        self.assertFalse(plan["complete"])
        study = adapt(result, "synthetic/results.json", "a" * 64)
        self.assertEqual(study["results"]["final_model_inventory"], 1)
        self.assertEqual(study["results"]["training_budget"], 2_000_000)
        self.assertFalse(study["results"]["inventory"]["complete"])
        wrong = episode("scenario_0", model=models[-2]["model_id"], role="FINAL_ID")
        with self.assertRaises(ValueError):
            mm.coverage(rows + [wrong], models, datasets, protocol_id=PROTOCOL)

    def test_production_train_denominator_900_requires_explicit_protocol(self):
        seal = "a" * 64
        m = model()
        m.update({key: seal for key in ("checkpoint_sha256", "config_sha256", "source_sha256", "runtime_sha256")})
        d = dataset("TRAIN", 900)
        d["manifest_sha256"] = seal
        rows = [episode(sid, model=m["model_id"], seed=m["seed"], reward=m["reward"],
                        checkpoint_sha256=seal, snapshot_sha256=seal, events_sha256=seal, record_sha256=seal)
                for sid in d["scenario_ids"]]
        kwargs = dict(study_id="synthetic", provenance={"identities": {k: {"sha256": seal} for k in ("source", "runtime")}},
                      datasets=[d], models=[m], rows=rows)
        result = mm.build_export(**kwargs, protocol_id=PROTOCOL)
        self.assertEqual(result["outcomes"][0]["n_expected"], 900)
        self.assertFalse(result["planned_campaign_coverage"]["complete"])
        with self.assertRaises(ValueError):
            mm.build_export(**kwargs)
        wrong = copy.deepcopy(kwargs)
        wrong["datasets"][0].update(scenario_count=300, scenario_ids=d["scenario_ids"][:300])
        wrong["rows"] = wrong["rows"][:300]
        with self.assertRaises(ValueError):
            mm.build_export(**wrong, protocol_id=PROTOCOL)

    def test_duplicate_checkpoint_and_incorrect_seed_slot_rejected(self):
        for models in ([model(), model(model_id="duplicate")], [dict(model(), slot=1)],
                       [dict(model(), seed_slot=False)]):
            with self.assertRaises(ValueError):
                mm.build_export(study_id="synthetic", provenance={}, datasets=[], models=models,
                                rows=[], fixture=True, protocol_id=PROTOCOL)

    def test_robustness_cutoffs_and_four_of_five_rule_remain(self):
        passing = mm.robustness(dict(zip(mm.FINAL_ROLES, (.7, .5, .5, .5, .35))))
        self.assertTrue(passing["passes"])
        self.assertFalse(mm.robustness(dict(zip(mm.FINAL_ROLES, (.7, .5, .5, .5, .349))))["passes"])
        assessments = {seed: dict(passing, passes=i < 4) for i, seed in enumerate(mm.PRIMARY_SEEDS)}
        self.assertTrue(mm.family_robustness(assessments, protocol_id=PROTOCOL)["passes"])
        del assessments[mm.PRIMARY_SEEDS[-1]]
        self.assertIsNone(mm.family_robustness(assessments, protocol_id=PROTOCOL)["passes"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
