
"""Offline frozen scenario collections and reward-independent episode schedules."""

import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import numpy as np

from src.environment.generated_scenario import GeneratedScenario
from src.environment.scenario_serialization import digest, load_scenario, save_scenario, scenario_to_dict

COLLECTION_VERSION = 1

# Validates dataset category and prevents improper naming
def check_labels(collection_id: str, split: str) -> None:
    if not isinstance(collection_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", collection_id):
        raise ValueError("collection_id must be a safe directory name")
    
    if split not in {"TRAIN", "CHECKPOINT_VALIDATION", "FINAL_ID"} and not (
        isinstance(split, str) and split.startswith("FINAL_OOD_") and len(split) > len("FINAL_OOD_")
    ):
        raise ValueError("Unrecognized scenario split")

# Reads manifest and verifies integrity
def load_collection_manifest(collection_dir: str | Path) -> dict:

    with (Path(collection_dir) / "manifest.json").open(encoding="utf-8") as file:
        manifest = json.load(file)

    expected = {"schema_version", "collection_id", "split", "code_version", "entries", "checksum"}

    if not isinstance(manifest, dict) or set(manifest) != expected:
        raise ValueError("Invalid collection manifest fields")
    
    unsigned = dict(manifest)

    if unsigned.pop("checksum") != digest(unsigned):
        raise ValueError("Collection manifest checksum mismatch")
    
    if manifest["schema_version"] != COLLECTION_VERSION:
        raise ValueError("Unsupported collection manifest version")
    
    check_labels(manifest["collection_id"], manifest["split"])

    if not isinstance(manifest["code_version"], str) or not manifest["code_version"]:
        raise ValueError("Missing collection code version")
    
    entries = manifest["entries"]

    if not isinstance(entries, list) or not entries:
        raise ValueError("Empty or invalid collection entries")
    
    if any(not isinstance(item, dict) or set(item) != {"scenario_id", "map_id", "checksum", "area_band"}
           or not all(isinstance(item[key], str) and item[key] for key in item)
           for item in entries):
        raise ValueError("Invalid collection entry")
    
    if len({entry["scenario_id"] for entry in entries}) != len(entries):
        raise ValueError("Duplicate scenario IDs in collection")
    
    return manifest

# Archival function
def write_collection(root: str | Path, collection_id: str, split: str, scenarios: Iterable[GeneratedScenario]) -> Path:
    # Destination setup
    check_labels(collection_id, split)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / collection_id

    # If dst already exists
    if destination.exists():
        raise FileExistsError(f"Collection already exists: {destination}")

    previously_used_ids = set()
    previously_used_maps = set()

    # Collects prev used scenario IDs and map ids
    for existing in root.glob("*/manifest.json"):
        old = load_collection_manifest(existing.parent)
        previously_used_ids.update(item["scenario_id"] for item in old["entries"])
        previously_used_maps.update(item["map_id"] for item in old["entries"])

    records = []
    versions = set()

    for scenario in scenarios:
        provenance = scenario.provenance
        if provenance.get("collection_id") != collection_id or provenance.get("split") != split:
            raise ValueError("Scenario provenance does not match its collection")

        # Gets scen ID and checksum from scenario
        record = scenario_to_dict(scenario)
        entry = {
            "scenario_id": record["scenario_id"],
            "map_id": "map_" + digest(record["map"]),
            "checksum": record["checksum"],
            "area_band": provenance["area_band"],
        }

        if entry["scenario_id"] in previously_used_ids or entry["map_id"] in previously_used_maps:
            raise ValueError("Scenario or map already exists in another collection")
        
        previously_used_ids.add(entry["scenario_id"])
        previously_used_maps.add(entry["map_id"])
        versions.add(provenance["code_version"])
        records.append((entry, scenario))

    # Version check
    if not records or len(versions) != 1:
        raise ValueError("Collection needs scenarios from one pinned code version")

    # Writes to temp directory first rather than publishing incomplete output
    with TemporaryDirectory(prefix=".staging-", dir=root) as temporary:
        staging = Path(temporary)
        for entry, scenario in records:
            path = staging / f'{entry["scenario_id"]}.json'
            save_scenario(scenario, path)
            if scenario_to_dict(load_scenario(path))["checksum"] != entry["checksum"]:
                raise ValueError("Snapshot failed round-trip integrity verification")

        manifest = {
            "schema_version": COLLECTION_VERSION,
            "collection_id": collection_id,
            "split": split,
            "code_version": versions.pop(),
            "entries": sorted((entry for entry, _ in records), key=lambda item: item["scenario_id"])
        }
        manifest["checksum"] = digest(manifest)

        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        staging.rename(destination)

    return destination

# Take frozen maifest, dst path, and one schedule seed to create an ordering traing scenario IDs
def write_training_schedule(train_collection_dir: str | Path, schedule_path: str | Path,slot_seeds: dict[int, int]) -> Path:
    manifest = load_collection_manifest(train_collection_dir)
    if manifest["split"] != "TRAIN":
        raise ValueError("Training schedules require a TRAIN collection")
    
    if not slot_seeds or any(type(slot) is not int or slot < 0 or
                             type(seed) is not int or seed < 0
                             for slot, seed in slot_seeds.items()):
        raise ValueError("Slot IDs and schedule seeds must be nonnegative integers")

    ids = [entry["scenario_id"] for entry in manifest["entries"]]
    orders = {}
    for slot, seed in sorted(slot_seeds.items()):
        rng = np.random.default_rng(seed)
        orders[str(slot)] = [ids[int(i)] for i in rng.permutation(len(ids))]

    schedule = {
        "schema_version": COLLECTION_VERSION,
        "collection_id": manifest["collection_id"],
        "collection_checksum": manifest["checksum"],
        "method": "cycle",
        "slot_seeds": {str(slot): seed for slot, seed in sorted(slot_seeds.items())},
        "slot_orders": orders,
    }

    schedule["checksum"] = digest(schedule)
    schedule_path = Path(schedule_path)

    with schedule_path.open("x", encoding="utf-8") as file:
        json.dump(schedule, file, indent=2, sort_keys=True, allow_nan=False)

    return schedule_path

# Verifies schedule after open
def load_training_schedule(path: str | Path, train_manifest: dict) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        schedule = json.load(file)

    expected = {"schema_version", "collection_id", "collection_checksum", "method", "slot_seeds", "slot_orders", "checksum"}

    if not isinstance(schedule, dict) or set(schedule) != expected:
        raise ValueError("Invalid schedule fields")
    
    unsigned = dict(schedule)

    if unsigned.pop("checksum") != digest(unsigned):
        raise ValueError("Schedule checksum mismatch")
    
    if (schedule["schema_version"] != COLLECTION_VERSION or schedule["method"] != "cycle"
            or train_manifest["split"] != "TRAIN"
            or schedule["collection_id"] != train_manifest["collection_id"]
            or schedule["collection_checksum"] != train_manifest["checksum"]):
        raise ValueError("Schedule does not match the frozen TRAIN collection")
    
    ids = {entry["scenario_id"] for entry in train_manifest["entries"]}
    orders = schedule["slot_orders"]
    seeds = schedule["slot_seeds"]

    if (not isinstance(orders, dict) or not orders or not isinstance(seeds, dict)
            or set(orders) != set(seeds)
            or any(not re.fullmatch(r"0|[1-9][0-9]*", str(slot))
                   or not isinstance(order, list) or len(order) != len(ids)
                   or not all(isinstance(item, str) for item in order)
                   or set(order) != ids
                   or type(seeds[slot]) is not int or seeds[slot] < 0
                   for slot, order in orders.items())):
        raise ValueError("Invalid scenario orders")

    return schedule

# Retrieves scenario order and cycles through it
def scheduled_scenario_id(schedule: dict, model_seed_slot: int, episode_index: int) -> str:
    if type(model_seed_slot) is not int or type(episode_index) is not int or episode_index < 0:
        raise ValueError("Invalid model slot or episode index")
    order = schedule["slot_orders"][str(model_seed_slot)]
    return order[episode_index % len(order)]
