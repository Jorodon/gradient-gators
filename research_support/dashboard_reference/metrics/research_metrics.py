"""Research pure saved-record metrics. No simulator, model or training imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median, stdev

if __package__:
    from ..protocols import (FINAL_ROLES, LEGACY_PROTOCOL_ID, PRIMARY_SEEDS,
                             REWARDS, get_protocol, validate_training_steps)
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from protocols import (FINAL_ROLES, LEGACY_PROTOCOL_ID, PRIMARY_SEEDS,
                           REWARDS, get_protocol, validate_training_steps)

# Public constants remain the historical defaults for existing callers.
GRID = get_protocol()["grid"]
ROLE_COUNTS = get_protocol()["role_counts"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def number(value, label, minimum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} below minimum")
    
    return value


def validate_episode(row):
    """Reject ambiguous/partial outcomes; counts and raw units are authoritative."""
    if row.get("status") != "completed":
        raise ValueError("Only completed episodes enter outcome metrics")
    
    if any(type(row.get(k)) is not bool for k in ("success", "terminated", "truncated")):
        raise ValueError("Outcome flags must be booleans, never CSV strings")
    
    reason = row.get("end_reason")
    if reason not in ("goal", "death", "timestep_limit"):
        raise ValueError("Unrecognized outcome")
    
    if (row["success"] != (reason == "goal") or
            row["truncated"] != (reason == "timestep_limit") or
            row["terminated"] != (reason != "timestep_limit")):
        raise ValueError("Inconsistent termination / living goal")
    
    if number(row["final_hp"], "final_hp", 0) <= 0 and reason != "death":
        raise ValueError("Non-living outcome must be death")
    
    if reason == "death" and row["final_hp"] != 0:
        raise ValueError("Death requires exhausted HP")
    
    steps = row["episode_length"]
    if type(steps) is not int or steps <= 0:
        raise ValueError("episode_length must be positive integer")
    
    for key in ("wait_actions", "invalid_moves", "hazard_contacts", "enemy_contacts"):
        if type(row[key]) is not int or not 0 <= row[key] <= steps:
            raise ValueError(f"Invalid {key}")
        
    if row["wait_actions"] + row["invalid_moves"] > steps:
        raise ValueError("WAIT and invalid actions overlap")
    
    number(row["episode_return"], "episode_return")

    for key in ("damage_taken", "fall_damage", "hazard_damage", "enemy_damage"):
        number(row[key], key, 0)

    if not math.isclose(row["damage_taken"], sum(row[k] for k in
                        ("fall_damage", "hazard_damage", "enemy_damage")), abs_tol=1e-8):
        raise ValueError("Damage sources do not reconcile")
    
    return row


def _scheduled_pairs(models, datasets, protocol):
    """The recurrent schedule evaluates acquisition/final panels only at budget."""
    return [(m, d) for m in models for d in datasets
            if protocol["id"] == LEGACY_PROTOCOL_ID
            or d["role"] in ("DEV", "LEARNING_VALIDATION")
            or m["checkpoint_step"] == protocol["budget"]]


def coverage(rows, models, datasets, *, protocol_id=None):
    """Exact scheduled coverage; unexpected and duplicate rows are errors."""
    protocol = get_protocol(protocol_id)
    expected = {(m["model_id"], d["role"], s)
                for m, d in _scheduled_pairs(models, datasets, protocol)
                for s in d["scenario_ids"]}
    seen = set()
    errors = []

    for row in rows:
        key = (row["model_id"], row["role"], row["scenario_id"])
        if key not in expected or key in seen:
            raise ValueError(f"Unexpected or duplicate record: {key}")
        
        seen.add(key)

        if row.get("status") != "completed":
            errors.append(key)

        else:
            validate_episode(row)

    missing = sorted(expected - seen)

    return {"n_expected": len(expected), 
            "n_completed": len(seen) - len(errors),
            "missing": missing, "errors": errors,
            "complete": not missing and not errors}


def aggregate(rows, *, model_id, role, scenario_ids):
    expected = set(scenario_ids)

    if len(expected) != len(scenario_ids) or not expected:
        raise ValueError("Expected scenarios must be unique and nonempty")
    
    seen = set()
    good = []
    errors = []

    for row in rows:
        if row["model_id"] != model_id or row["role"] != role:
            raise ValueError("Mixed model / role aggregate")
        
        sid = row["scenario_id"]
        if sid not in expected or sid in seen:
            raise ValueError("Duplicate or unexpected scenario")
        
        seen.add(sid)
        if row.get("status") == "completed":
            good.append(validate_episode(row))

        else:
            errors.append(sid)

    n, denom = len(good), len(expected)
    successes = [r for r in good if r["success"]]
    failures = [r for r in good if not r["success"]]
    steps = sum(r["episode_length"] for r in good)

    result = {"model_id": model_id, "role": role, "n_expected": denom,
              "n_completed": n, "complete": n == denom,
              "missing_scenario_ids": sorted(expected - seen), "error_scenario_ids": errors,
              "success_count": len(successes),
              "death_count": sum(r["end_reason"] == "death" for r in good),
              "timeout_count": sum(r["end_reason"] == "timestep_limit" for r in good),
              "actual_evaluation_steps": steps,
              "success_fraction": len(successes) / denom if n == denom else None,
              "success_fraction_observed": len(successes) / n if n else None,
              "reason_incomplete": None if n == denom else "missing_or_error_records",
              "zero_damage_success_count": sum(r["damage_taken"] == 0 for r in successes)}
    
    result["zero_damage_success_fraction"] = (
        result["zero_damage_success_count"] / denom if n == denom else None)
    
    for outcome in ("death", "timeout"):
        result[f"{outcome}_fraction"] = result[f"{outcome}_count"] / denom if n == denom else None

    result["zero_damage_given_success_fraction"] = (
        result["zero_damage_success_count"] / len(successes) if successes else None)
    
    result["zero_damage_given_success_reason"] = None if successes else "no_successes"

    for label, selected in (("all", good), ("success", successes), ("failure", failures)):
        result[f"mean_length_{label}"] = mean(r["episode_length"] for r in selected) if selected else None
        result[f"mean_length_{label}_reason"] = None if selected else f"no_{label}_episodes"

    for key in ("episode_return", "damage_taken", "fall_damage", "hazard_damage", "enemy_damage",
                "hazard_contacts", "enemy_contacts", "wait_actions", "invalid_moves"):
        result[f"total_{key}"] = sum(r[key] for r in good)
        result[f"mean_{key}"] = mean(r[key] for r in good) if good else None

    for name, key in (("wait", "wait_actions"), ("invalid", "invalid_moves")):
        result[f"{name}_fraction_episode_mean"] = mean(r[key] / r["episode_length"] for r in good) if good else None
        result[f"{name}_fraction_pooled"] = sum(r[key] for r in good) / steps if steps else None

    result["empty_metrics_reason"] = None if good else "no_completed_episodes"
    return result


def learning_metrics(points, *, protocol_id=None):
    """Require the complete protocol grid; never fill missing evaluations.

    Recurrent areas and threshold times use consumed interactions at completed
    episodes. The checkpoint grid identifies scheduled panels, not coordinates.
    """
    protocol = get_protocol(protocol_id)
    grid, budget = protocol["grid"], protocol["budget"]
    recurrent = protocol["id"] != LEGACY_PROTOCOL_ID
    denominator = protocol["role_counts"]["LEARNING_VALIDATION"]
    by_step = {}

    for p in points:
        t = p["checkpoint_step"]
        if t not in grid or t in by_step:
            raise ValueError("Duplicate or off-grid checkpoint")

        if recurrent:
            validate_training_steps(p, protocol["id"])
            if any(type(p[k]) is not int for k in ("n_expected", "n_completed", "success_count")):
                raise ValueError("Learning counts must be integers")
        
        if (type(p["success_count"]) is not int or not
                0 <= p["success_count"] <= p["n_completed"] <= p["n_expected"] or
                p["n_expected"] != denominator):
            raise ValueError("Learning validation requires exact 100-scenario counts")
        
        by_step[t] = p

    missing = [t for t in grid if t not in by_step or by_step[t]["n_completed"] != denominator]

    result = {"observed_checkpoints": sorted(by_step), "missing_or_partial_checkpoints": missing,
              "normalized_aulc": None, "time_to_80_successes": None,
              "threshold_status": "incomplete", "censoring_timestep": None,
              "reason": "incomplete_grid" if missing else None}

    if recurrent:
        result.update(coordinate="actual_consumed_interactions",
                      threshold_rule="two_adjacent_positive_scheduled_checkpoints_at_least80_of100",
                      checkpoint_coordinates=[{k: by_step[t][k] for k in
                          ("checkpoint_step", "actual_training_steps", "optimized_training_steps")}
                          for t in sorted(by_step)])
    
    if missing:
        return result
    
    rates = [by_step[t]["success_count"] / denominator for t in grid]
    actual = [by_step[t]["actual_training_steps"] if recurrent else t for t in grid]
    if any(a >= b for a, b in zip(actual, actual[1:])):
        raise ValueError("Consumed interaction coordinates must strictly increase")
    if recurrent:
        result["normalized_aulc"] = sum((actual[i] - actual[i-1]) * (rates[i] + rates[i-1]) / 2
                                         for i in range(1, len(grid))) / budget
    else:
        result["normalized_aulc"] = sum((a + b) * .125 for a, b in zip(rates, rates[1:]))
    attained = next((actual[i] for i in range(2, len(grid))
                     if rates[i-1] >= .8 and rates[i] >= .8), None)
    result.update(time_to_80_successes=attained,
                  threshold_status="attained" if attained is not None else "censored",
                  censoring_timestep=None if attained is not None else budget)
    
    return result


def holm(p_values):
    """Four-test family; unknown p values retain a slot and remain undefined."""
    if len(p_values) != 4:
        raise ValueError("The frozen protocol has exactly four primary tests")
    
    for p in p_values:
        if p is not None and not 0 <= number(p, "p") <= 1:
            raise ValueError("Invalid p value")
        
    order = sorted(range(4), key=lambda i: (p_values[i] is None, p_values[i] or 0, i))
    adjusted, previous = [None] * 4, 0.0

    for rank, i in enumerate(order):
        if p_values[i] is not None:
            previous = max(previous, min(1.0, (4 - rank) * p_values[i]))
            adjusted[i] = previous

    return adjusted


def paired_contrasts(outcomes, models, *, protocol_id=None):
    """Fixed combined-OOD endpoint: n=5, df=4, nominal CI, four-test Holm.

    - Missing pairs retain descriptive differences
    - The df4 two-sided tail is integrated analytically; zero variance is undefined
    """
    protocol = get_protocol(protocol_id)
    seeds = protocol["primary_seeds"]
    rewards = protocol["rewards"]
    inventory = {m["model_id"]: m for m in models}
    values = {}

    for o in outcomes:
        m = inventory[o["model_id"]]
        if o["role"] != "FINAL_OOD_COMBINED" or m["checkpoint_step"] != protocol["budget"]:
            continue
        
        validate_training_steps(m, protocol["id"])
        if m["seed"] not in seeds or m["reward"] not in rewards:
            raise ValueError("Primary comparison contains an unregistered seed or reward")
        
        key = (m["reward"], m["seed"])
        if key in values:
            raise ValueError("Duplicate primary seed / reward")
        
        values[key] = o["success_fraction"] if o["complete"] else None

    results = []
    for reward in rewards[1:]:
        pairs, missing = [], []

        for seed in seeds:
            a, b = values.get((reward, seed)), values.get(("sparse", seed))

            if a is None or b is None:
                missing.append(seed)

            else:
                if not all(0 <= number(v, "success fraction") <= 1 for v in (a, b)):
                    raise ValueError("Invalid success fraction")
                
                pairs.append({"seed": seed, "shaped": a, "sparse": b, "difference": a - b})

        ds = [p["difference"] for p in pairs]
        result = {"reward": reward, "reference": "sparse", "role": "FINAL_OOD_COMBINED",
                  "inference": "primary", "unit": "trained_model_seed", "n_pairs": len(ds),
                  "pairs": pairs, "missing_seeds": missing, "mean_difference": mean(ds) if ds else None,
                  "ci95": None, "df": None, "t": None, "p_two_sided": None,
                  "reason": "missing_seed_pairs" if missing else None,
                  "assumptions": "Five independent matched seeds; approximately normal seed differences; small n; nominal unadjusted CI."}
        
        if len(ds) == 5:
            sd = stdev(ds)
            result["df"] = 4

            if sd <= 1e-15:
                result["reason"] = "zero_variance_t_inference_undefined"

            else:
                se = sd / math.sqrt(5)
                t = mean(ds) / se
                u = abs(t) / math.sqrt(t*t + 4)
                p = max(0.0, min(1.0, 1 - 1.5*u + .5*u**3))
                result.update(t=t, p_two_sided=p,
                              ci95=[mean(ds) - 2.7764451051977987*se,
                                    mean(ds) + 2.7764451051977987*se])
        results.append(result)

    for result, p in zip(results, holm([r["p_two_sided"] for r in results])):
        result.update(p_holm=p, reject_holm_05=p <= .05 if p is not None else None)

    return results


def robustness(rates):
    if set(rates) - set(FINAL_ROLES):
        raise ValueError("Unknown final condition")
    
    if any(rates.get(r) is None for r in FINAL_ROLES):
        return {"fully_assessed": False, "passes": None, "reason": "missing_conditions", "retention": None}
    
    if any(not 0 <= number(rates[r], r) <= 1 for r in FINAL_ROLES):
        raise ValueError("Invalid fraction")
    
    base = rates["FINAL_ID"]
    checks = {"id": base >= .70, "combined": rates[FINAL_ROLES[-1]] >= .35,
              "combined_drop": base - rates[FINAL_ROLES[-1]] <= .40 + 1e-12}
    
    for role in FINAL_ROLES[1:-1]:
        checks[role] = rates[role] >= .50 and base - rates[role] <= .25 + 1e-12

    return {"fully_assessed": True, "passes": all(checks.values()), "checks": checks,
            "reason": None, "retention": {r: rates[r]/base if base else None for r in FINAL_ROLES[1:]},
            "retention_reason": None if base else "zero_ID_denominator"}


def family_robustness(assessments, *, protocol_id=None):
    """Input maps the five prespecified seed identifiers to model assessments."""
    protocol = get_protocol(protocol_id)
    if set(assessments) - set(protocol["primary_seeds"]):
        raise ValueError("Unexpected primary seed")
    
    n = sum(a["fully_assessed"] for a in assessments.values())
    passed = sum(a["passes"] is True for a in assessments.values())
    
    return {"n_expected": 5, "n_fully_assessed": n, "n_pass": passed,
            "passes": passed >= 4 if n == 5 else None,
            "reason": None if n == 5 else "five_fully_assessed_models_required"}


def matched_scenarios(datasets, *, protocol_id=None):
    """Preselect solely from IDs, before outcomes: hash then ID tie-break."""
    get_protocol(protocol_id)
    # Both registered protocols prescribe this same outcome-blind selection salt.
    return {d["role"]: sorted(d["scenario_ids"], key=lambda s:
            (digest([LEGACY_PROTOCOL_ID, d["role"], s]), s))[:2]
            for d in datasets if d["role"] in FINAL_ROLES}


def select_traces(rows, models, datasets, *, protocol_id=None):
    protocol = get_protocol(protocol_id)
    seeds = protocol["primary_seeds"]
    inventory = {m["model_id"]: m for m in models}
    selected, absent = [], []
    matched = matched_scenarios(datasets, protocol_id=protocol["id"])
    eligible = [r for r in rows if r.get("status") == "completed" and r["role"] in FINAL_ROLES
                and inventory[r["model_id"]]["checkpoint_step"] == protocol["budget"]
                and inventory[r["model_id"]]["seed"] in seeds]
    for r in eligible:
        m = inventory[r["model_id"]]
        validate_training_steps(m, protocol["id"])
        if m["seed"] == seeds[0] and r["scenario_id"] in matched[r["role"]]:
            selected.append({**r, "selection_category": "matched"})
    for role in matched:
        for reward in protocol["rewards"]:
            for success in (True, False):
                pool = [r for r in eligible if r["role"] == role and r["success"] == success
                        and inventory[r["model_id"]]["reward"] == reward]
                if not pool:
                    absent.append({"role": role, "reward": reward, "success": success,
                                   "reason": "no_eligible_outcome_class"})
                    continue
                mid = median(r["episode_length"] for r in pool)
                chosen = min(pool, key=lambda r: (abs(r["episode_length"] - mid), r["scenario_id"],
                                                   inventory[r["model_id"]]["seed"]))
                selected.append({**chosen, "selection_category": "representative"})
    return selected, absent


def build_export(*, study_id, provenance, datasets, models, rows, fixture=False, traces=(),
                 protocol_id=None):
    protocol = get_protocol(protocol_id)
    recurrent = protocol["id"] != LEGACY_PROTOCOL_ID
    seeds, rewards, budget = protocol["primary_seeds"], protocol["rewards"], protocol["budget"]
    matrix = coverage(rows, models, datasets, protocol_id=protocol["id"])

    if len({m["model_id"] for m in models}) != len(models):
        raise ValueError("Duplicate model ID")
    
    if len({d["role"] for d in datasets}) != len(datasets):
        raise ValueError("Duplicate dataset role")

    if recurrent:
        dimensions = set()
        for m in models:
            validate_training_steps(m, protocol["id"])
            if m["seed"] not in seeds or m["reward"] not in rewards:
                raise ValueError("Unregistered recurrent seed or reward")
            dimension = (m["reward"], m["seed"], m["checkpoint_step"])
            if dimension in dimensions:
                raise ValueError("Duplicate reward/seed/checkpoint model")
            dimensions.add(dimension)
            for key in ("slot", "seed_slot"):
                if key in m and (type(m[key]) is not int or m[key] != seeds.index(m["seed"])):
                    raise ValueError("Primary seed slot mismatch")
    
    for d in datasets:
        if recurrent and d["role"] not in protocol["role_counts"]:
            raise ValueError("Unknown recurrent dataset role")
        if d["scenario_count"] != len(d["scenario_ids"]) or len(set(d["scenario_ids"])) != len(d["scenario_ids"]):
            raise ValueError("Dataset count mismatch")
        if not fixture and (d["scenario_count"] != protocol["role_counts"][d["role"]] or not d["manifest_sha256"]):
            raise ValueError("Production datasets require exact counts and hashes")
        
    if not fixture:
        import re
        def hash_required(value):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("Production export requires exact identity hashes")
            
        identities = provenance["identities"]
        for name in ("source", "runtime"):
            hash_required(identities[name]["sha256"])

        for d in datasets:
            hash_required(d["manifest_sha256"])

        model_map = {m["model_id"]: m for m in models}
        for m in models:
            for name in ("checkpoint_sha256", "config_sha256", "source_sha256", "runtime_sha256"):
                hash_required(m[name])
            if m["source_sha256"] != identities["source"]["sha256"] or m["runtime_sha256"] != identities["runtime"]["sha256"]:
                raise ValueError("Model/source/runtime export join mismatch")
            
        for r in rows:
            m = model_map[r["model_id"]]
            if r["seed"] != m["seed"] or r["reward"] != m["reward"] or r["checkpoint_sha256"] != m["checkpoint_sha256"]:
                raise ValueError("Episode/model export join mismatch")
            for name in ("snapshot_sha256", "events_sha256", "record_sha256"):
                hash_required(r[name])

    outcomes = [aggregate([r for r in rows if r["model_id"] == m["model_id"] and r["role"] == d["role"]],
                          model_id=m["model_id"], role=d["role"], scenario_ids=d["scenario_ids"])
                for m, d in _scheduled_pairs(models, datasets, protocol)]
    
    inventory = {m["model_id"]: m for m in models}
    learning = [{**o, "checkpoint_step": inventory[o["model_id"]]["checkpoint_step"]}
                for o in outcomes if o["role"] in ("DEV", "LEARNING_VALIDATION")]
    if recurrent:
        for point in learning:
            model = inventory[point["model_id"]]
            point.update({k: model[k] for k in ("actual_training_steps", "optimized_training_steps")})
    
    curves = []
    groups = sorted({(inventory[o["model_id"]]["reward"], inventory[o["model_id"]]["seed"], o["role"])
                     for o in learning})
    for reward, seed, role in groups:
        points = [o for o in learning if o["role"] == role and
                  (inventory[o["model_id"]]["reward"], inventory[o["model_id"]]["seed"]) == (reward, seed)]
        # Tiny viewer fixtures are explicitly outside the 100-scenario protocol.
        metric = learning_metrics(points, protocol_id=protocol["id"]) if all(o["n_expected"] == 100 for o in points) else {
            "normalized_aulc": None, "reason": "fixture_denominator"}
        curves.append({"reward": reward, "seed": seed, "role": role, **metric})

    assessments = []
    for m in models:
        if m["checkpoint_step"] == budget:
            rates = {o["role"]: o["success_fraction"] for o in outcomes
                     if o["model_id"] == m["model_id"] and o["role"] in FINAL_ROLES}
            assessments.append({"model_id": m["model_id"], "reward": m["reward"], "seed": m["seed"],
                                **robustness(rates)})
            
    families = [{"reward": reward, **family_robustness({a["seed"]: a for a in assessments
                 if a["reward"] == reward and a["seed"] in seeds}, protocol_id=protocol["id"])} for reward in rewards]
    
    result = {"schema_version": protocol["schema_versions"][0], "study_id": study_id, "fixture": fixture,
              "provenance": provenance, "datasets": datasets, "models": models,
              "outcomes": outcomes, "learning": learning, "learning_curves": curves,
              "comparisons": paired_contrasts(outcomes, models, protocol_id=protocol["id"]), "traces": list(traces),
              "coverage": matrix, "robustness_models": assessments, "robustness_families": families,
              "limitations": ["Five trained seeds are the independent replicates; scenarios are not seeds.",
                              "TRAIN acquisition, adaptive DEV and fresh ID transfer have distinct meanings.",
                              "No ordinary t-test is applied to censored threshold times."]}
    
    if recurrent:
        result["protocol_id"] = protocol["id"]

    final_models = [(m["reward"], m["seed"]) for m in models if m["checkpoint_step"] == budget]
    if len(final_models) != len(set(final_models)):
        raise ValueError("Ambiguous final reward/seed model inventory")
    
    expected_models = {(r, s) for r in rewards for s in seeds}
    missing_models = sorted(expected_models - set(final_models))
    missing_roles = sorted(set(FINAL_ROLES) - {d["role"] for d in datasets})
    final_rows = [r for r in rows if r["role"] in FINAL_ROLES and r.get("status") == "completed"
                  and (not recurrent or inventory[r["model_id"]]["checkpoint_step"] == budget)]
    expected_final_episodes = len(expected_models) * sum(protocol["role_counts"][r] for r in FINAL_ROLES)

    result["planned_final_coverage"] = {
        "n_expected_models": len(expected_models), "n_expected_conditions": len(FINAL_ROLES),
        "n_expected_episodes": expected_final_episodes,
        "n_completed_episodes": len(final_rows), "missing_reward_seed_models": missing_models,
        "missing_conditions": missing_roles,
        "complete": not fixture and not missing_models and not missing_roles and len(final_rows) == expected_final_episodes and matrix["complete"],
        "requested_coverage_scope": "coverage reports only the explicitly supplied model/dataset inventory"}

    if recurrent:
        required_roles = protocol["required_roles"]
        expected_slots = {(reward, seed, step) for reward, seed in expected_models for step in protocol["grid"]}
        supplied_slots = {(m["reward"], m["seed"], m["checkpoint_step"]) for m in models}
        missing_slots = sorted(expected_slots - supplied_slots)
        missing_campaign_roles = sorted(set(required_roles) - {d["role"] for d in datasets})
        expected_campaign_episodes = len(expected_models) * (
            protocol["role_counts"]["LEARNING_VALIDATION"] * len(protocol["grid"])
            + sum(protocol["role_counts"][r] for r in required_roles if r != "LEARNING_VALIDATION"))
        completed_campaign = sum(r["role"] in required_roles and r.get("status") == "completed" for r in rows)
        result["planned_campaign_coverage"] = {
            "n_expected_model_checkpoints": len(expected_slots),
            "n_expected_evaluation_panels": len(expected_models) * (len(protocol["grid"]) + len(required_roles) - 1),
            "n_expected_episodes": expected_campaign_episodes,
            "n_completed_episodes": completed_campaign,
            "missing_reward_seed_checkpoints": missing_slots,
            "missing_dataset_roles": missing_campaign_roles,
            "complete": (not fixture and not missing_slots and not missing_campaign_roles
                         and completed_campaign == expected_campaign_episodes and matrix["complete"])}
    
    if fixture:
        result["limitations"].append("SYNTHETIC FIXTURE ONLY; no policy or research result.")

    json.dumps(result, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSON build_export keyword arguments")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build_export(**json.loads(args.input.read_text(encoding="utf-8-sig")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
