"""Synthetic data only. Run directly with Python -B; no learning dependencies."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research_support.dashboard_reference.metrics import research_metrics as mm


def episode(sid="a", *, model="m", role="TRAIN", success=True, length=10, **extra):
    return {"model_id": model, "role": role, "scenario_id": sid, "status": "completed",
            "success": success, "terminated": success, "truncated": not success,
            "end_reason": "goal" if success else "timestep_limit", "final_hp": 100,
            "episode_return": 100 if success else 0, "episode_length": length,
            "damage_taken": 0, "fall_damage": 0, "hazard_damage": 0, "enemy_damage": 0,
            "hazard_contacts": 0, "enemy_contacts": 0, "wait_actions": 0, "invalid_moves": 0, **extra}


def points(counts=(0, 80, 80, 60, 90)):
    return [{"checkpoint_step": t, "n_expected": 100, "n_completed": 100, "success_count": s}
            for t, s in zip(mm.GRID, counts)]


class MetricsTests(unittest.TestCase):
    def test_complete_grid_and_second_poststart(self):
        result = mm.learning_metrics(points())
        self.assertAlmostEqual(result["normalized_aulc"], .6625)
        self.assertEqual(result["time_to_80_successes"], 500000)
        self.assertEqual(mm.learning_metrics(points((100, 100, 0, 0, 0)))["threshold_status"], "censored")
        self.assertEqual(mm.learning_metrics(points((100, 100, 0, 0, 0)))["censoring_timestep"], 1000000)

    def test_missing_and_partial_grid_not_censored(self):
        for rows in (points()[1:], points()[:-1]):
            result = mm.learning_metrics(rows)
            self.assertIsNone(result["normalized_aulc"])
            self.assertEqual(result["threshold_status"], "incomplete")
        rows = points()
        rows[1]["n_completed"] = 80
        self.assertIsNone(mm.learning_metrics(rows)["normalized_aulc"])

    def test_duplicate_offgrid_rejected(self):
        with self.assertRaises(ValueError):
            mm.learning_metrics(points() + [points()[0]])
        with self.assertRaises(ValueError):
            mm.learning_metrics([dict(points()[0], checkpoint_step=25000)])

    def test_raw_units_and_distinct_invalid_denominators(self):
        rows = [episode(length=10, invalid_moves=5, wait_actions=2, episode_return=-20,
                        damage_taken=9, fall_damage=2, hazard_damage=3, enemy_damage=4),
                episode("b", success=False, length=30, invalid_moves=3, wait_actions=10)]
        result = mm.aggregate(rows, model_id="m", role="TRAIN", scenario_ids=["a", "b"])
        self.assertEqual(result["total_damage_taken"], 9)
        self.assertEqual(result["total_episode_return"], -20)
        self.assertEqual(result["mean_length_success"], 10)
        self.assertEqual(result["mean_length_failure"], 30)
        self.assertAlmostEqual(result["invalid_fraction_episode_mean"], .3)
        self.assertAlmostEqual(result["invalid_fraction_pooled"], .2)
        self.assertEqual(result["zero_damage_success_count"], 0)

    def test_missing_error_not_policy_zeros(self):
        rows = [episode(), {"model_id": "m", "role": "TRAIN", "scenario_id": "b", "status": "error"}]
        result = mm.aggregate(rows, model_id="m", role="TRAIN", scenario_ids=["a", "b", "c"])
        self.assertEqual(result["n_completed"], 1)
        self.assertIsNone(result["success_fraction"])
        self.assertEqual(result["success_fraction_observed"], 1)
        self.assertEqual(result["missing_scenario_ids"], ["c"])
        self.assertEqual(result["error_scenario_ids"], ["b"])

    def test_coverage_duplicate_unexpected(self):
        datasets = [{"role": r, "scenario_ids": ["a", "b"]} for r in mm.FINAL_ROLES]
        models = [{"model_id": m} for m in ("m", "n")]
        rows = [episode(s, model=m["model_id"], role=d["role"]) for m in models for d in datasets for s in d["scenario_ids"]]
        self.assertEqual(mm.coverage(rows, models, datasets)["n_expected"], 20)
        self.assertTrue(mm.coverage(rows, models, datasets)["complete"])
        self.assertFalse(mm.coverage(rows[:-1], models, datasets)["complete"])
        with self.assertRaises(ValueError):
            mm.coverage(rows + [rows[0]], models, datasets)
        with self.assertRaises(ValueError):
            mm.coverage([episode("unknown")], models, datasets)

    def test_living_goal_death_timeout_validation(self):
        for row in (episode(final_hp=0), episode(damage_taken=1), episode(success="False"), episode(wait_actions=9, invalid_moves=2)):
            with self.assertRaises(ValueError):
                mm.validate_episode(row)
        death = episode(success=False, truncated=False, terminated=True, end_reason="death", final_hp=0)
        self.assertFalse(mm.validate_episode(death)["success"])

    def test_holm_order_and_missing_family_slots(self):
        self.assertEqual(mm.holm([.03, .001, .02, .4]), [.06, .004, .06, .4])
        self.assertEqual(mm.holm([None, .01, None, .04]), [None, .04, None, .12])

    def comparison_input(self):
        models, outcomes = [], []
        for reward in mm.REWARDS:
            for i, seed in enumerate(mm.PRIMARY_SEEDS):
                mid = f"{reward}_{seed}"
                models.append({"model_id": mid, "reward": reward, "seed": seed, "checkpoint_step": 1000000})
                outcomes.append({"model_id": mid, "role": "FINAL_OOD_COMBINED", "complete": True,
                                 "success_fraction": .2 if reward == "sparse" else .2 + .05*i})
        return outcomes, models

    def test_paired_df4_reference(self):
        outcomes, models = self.comparison_input()
        result = mm.paired_contrasts(outcomes, models)[0]
        self.assertEqual(result["n_pairs"], 5)
        self.assertEqual(result["df"], 4)
        self.assertAlmostEqual(result["t"], 2.82842712474619)
        self.assertAlmostEqual(result["p_two_sided"], .0474206555843196)
        self.assertAlmostEqual(result["ci95"][0], .00183784192612198)

    def test_missing_pairs_and_zero_variance(self):
        outcomes, models = self.comparison_input()
        result = mm.paired_contrasts(outcomes[1:], models)[0]
        self.assertEqual(result["n_pairs"], 4)
        self.assertIsNone(result["p_two_sided"])
        for o in outcomes:
            o["success_fraction"] = .2
        result = mm.paired_contrasts(outcomes, models)[0]
        self.assertEqual(result["reason"], "zero_variance_t_inference_undefined")
        self.assertIsNone(result["ci95"])

    def test_robustness_all_conditions_zero_id_family(self):
        rates = dict(zip(mm.FINAL_ROLES, [.7, .5, .5, .5, .35]))
        self.assertTrue(mm.robustness(rates)["passes"])
        rates["FINAL_OOD_HAZARD"] = .49
        self.assertFalse(mm.robustness(rates)["passes"])
        self.assertIsNone(mm.robustness({"FINAL_ID": 1})["passes"])
        self.assertIsNone(mm.robustness(dict.fromkeys(mm.FINAL_ROLES, 0))["retention"]["FINAL_OOD_HAZARD"])
        assessments = {s: {"fully_assessed": True, "passes": i < 4} for i, s in enumerate(mm.PRIMARY_SEEDS)}
        self.assertTrue(mm.family_robustness(assessments)["passes"])
        del assessments[mm.PRIMARY_SEEDS[-1]]
        self.assertIsNone(mm.family_robustness(assessments)["passes"])

    def test_selection_uses_full_pool_and_ties(self):
        models = [{"model_id": str(s), "reward": "sparse", "seed": s, "checkpoint_step": 1000000} for s in mm.PRIMARY_SEEDS[:2]]
        datasets = [{"role": "FINAL_ID", "scenario_ids": ["a", "b", "c", "d", "e"]}]
        chosen_ids = mm.matched_scenarios(datasets)["FINAL_ID"]
        outside = next(s for s in datasets[0]["scenario_ids"] if s not in chosen_ids)
        rows = [episode(s, model=models[0]["model_id"], role="FINAL_ID", length=1) for s in chosen_ids]
        rows += [episode(outside, model=m["model_id"], role="FINAL_ID", length=20) for m in models]
        rows += [episode("e", model=models[1]["model_id"], role="FINAL_ID", length=50)]
        selected, absent = mm.select_traces(rows, models, datasets)
        representative = [r for r in selected if r["selection_category"] == "representative"][0]
        self.assertEqual(representative["scenario_id"], outside)
        self.assertEqual(representative["model_id"], str(min(mm.PRIMARY_SEEDS[:2])))
        self.assertTrue(absent)

    def test_export_partial_fixture_json_safe(self):
        d = {"role": "FINAL_ID", "scenario_ids": ["a", "b"], "scenario_count": 2,
             "manifest_path": "synthetic/manifest.json", "manifest_sha256": None}
        m = {"model_id": "m", "reward": "sparse", "seed": mm.PRIMARY_SEEDS[0], "checkpoint_step": 1000000}
        result = mm.build_export(study_id="fixture", provenance={}, datasets=[d], models=[m],
                                rows=[episode(role="FINAL_ID")], fixture=True)
        self.assertIsNone(result["outcomes"][0]["success_fraction"])
        self.assertFalse(result["coverage"]["complete"])
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
