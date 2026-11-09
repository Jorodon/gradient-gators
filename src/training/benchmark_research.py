"""Measure the existing CPU pilot runner; never approve or launch production."""

from __future__ import annotations

import argparse
import json
import platform
import threading
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import psutil
import torch

from src.training.run_research import run_batch


def summarize_batch(batch_dir: Path) -> list[dict]:
    """Read authoritative run outputs, without rebuilding their validators."""
    batch = json.loads((batch_dir / "manifest.json").read_text())
    rows = []
    for cell in batch["cells"]:
        directory = Path(cell["run_dir"])
        record = json.loads((directory / "manifest.json").read_text())
        if cell.get("reused"):
            continue  # Historical work did not consume this invocation's wall time.
        train_seconds = record["training_seconds"]
        eval_seconds = record["checkpoint_eval_seconds"]
        train_steps = record["actual_training_interactions"]
        eval_steps = record["actual_checkpoint_evaluation_interactions"]
        episodes = record["checkpoint_evaluation_ep_count"]
        files = [path for path in directory.rglob("*") if path.is_file()]
        rows.append({
            "run_dir": str(directory),
            "reward": record["effective_config"]["reward"]["variant"],
            "seed_slot": record["effective_config"]["model_seed_slot"],
            "training_steps": train_steps,
            "training_wall_seconds": train_seconds,
            "training_steps_per_second": train_steps / train_seconds,
            "validation_episodes": episodes,
            "validation_steps": eval_steps,
            "validation_wall_seconds": eval_seconds,
            "validation_episodes_per_second": episodes / eval_seconds,
            "validation_steps_per_second": eval_steps / eval_seconds,
            "evaluation_status": record["checkpoint_evaluation_status"],
            "artifact_bytes": sum(path.stat().st_size for path in files),
            "checkpoint_bytes": sum((directory / item["file"]).stat().st_size
                                    for item in record["checkpoints"]),
            "checkpoint_count": len(record["checkpoints"]),
            "baseline_checkpoint_bytes": (directory / record["checkpoints"][0]["file"]).stat().st_size,
            "trained_checkpoint_bytes": (directory / record["checkpoints"][-1]["file"]).stat().st_size,
        })
    return rows


def project_campaign(rows: list[dict], wall_seconds: float) -> dict | None:
    """Equal-weight each reward's measured rate in a five-by-five forecast."""
    rewards = {"sparse", "efficiency", "safety", "balanced", "progress"}
    if (len(rows) != 5 or {row["reward"] for row in rows} != rewards
            or any(row["training_steps"] != 50_000
                   or row["validation_episodes"] != 300
                   or row["evaluation_status"] != "completed" for row in rows)
            or len({row["seed_slot"] for row in rows}) != 1):
        return None  # A smoke run or partial restart cannot justify this forecast.
    training = sum(5 * 250_000 / row["training_steps_per_second"] for row in rows)
    validation = sum(5 * 1_100 / row["validation_episodes_per_second"] for row in rows)
    overhead = max(0.0, wall_seconds - sum(
        row["training_wall_seconds"] + row["validation_wall_seconds"] for row in rows
    )) * 5
    # 5 seeds x 1,000 final scenarios per reward, extrapolated from ID validation.
    final_proxy = sum(5 * 1_000 / row["validation_episodes_per_second"] for row in rows)
    # All episodes lasting 400 steps, still at measured ID throughput: not an OOD bound.
    long_episode_proxy = sum(5 * 1_000 * 400 / row["validation_steps_per_second"] for row in rows)
    checkpoint_bytes = sum(5 * (row["baseline_checkpoint_bytes"] + 10 * row["trained_checkpoint_bytes"])
                           for row in rows)
    return {
        "training_interactions": 6_250_000,
        "checkpoint_validation_episodes": 27_500,
        "final_evaluation_episodes": 25_000,
        "training_seconds": training,
        "checkpoint_validation_seconds": validation,
        "setup_and_other_seconds_proxy": overhead,
        "m3_seconds_proxy": training + validation + overhead,
        "m4_seconds_at_id_episode_rate_proxy": final_proxy,
        "m4_seconds_at_400_steps_and_id_step_rate_proxy": long_episode_proxy,
        "checkpoint_storage_bytes_proxy": checkpoint_bytes,
        "approval_status": "pending_human_review",
        "limitations": [
            "Linear one-slot CPU forecast; learning changes episode lengths and rates.",
            "M4 pursuit, trace writing and archive generation need separate measurements.",
            "Checkpoint storage excludes CSVs, traces, datasets and failed attempts.",
            "Reserve explicit calendar and storage headroom; no automatic budget change.",
        ],
    }


def benchmark(protocol_path: Path, output: Path, restart_batch: Path | None = None) -> Path:
    protocol = json.loads(protocol_path.read_text())
    if protocol["common"]["mode"] != "pilot" or protocol["common"]["ppo"]["device"] != "cpu":
        raise ValueError("This benchmark measures CPU pilots only")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve a new report before doing expensive work; preserve prior evidence.
    with output.open("x", encoding="utf-8") as handle:
        handle.write('{}\n')
    process = psutil.Process()
    stop = threading.Event()
    peak_rss = process.memory_info().rss

    def sample_memory():
        nonlocal peak_rss
        while not stop.wait(0.05):
            peak_rss = max(peak_rss, process.memory_info().rss)

    thread = threading.Thread(target=sample_memory, daemon=True)
    report = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(protocol_path.resolve()),
        "protocol": protocol,
        "restart_batch": str(restart_batch.resolve()) if restart_batch else None,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(),
        "host_memory_bytes": psutil.virtual_memory().total,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "psutil_version": version("psutil"),
        "memory_definition": "Process RSS sampled every 50 ms; brief peaks may be missed; no GPU/child memory.",
        "status": "running",
    }
    started = perf_counter()
    thread.start()
    try:
        batch_dir = run_batch(protocol_path, restart_batch=restart_batch)
        elapsed = perf_counter() - started
        report["batch_dir"] = str(batch_dir)
        report["runs"] = summarize_batch(batch_dir)
        report["projection"] = project_campaign(report["runs"], elapsed)
        report["status"] = "completed"
        return output
    except BaseException as error:
        report["status"] = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        report["failure"] = repr(error)
        raise
    finally:
        stop.set()
        thread.join()
        report["wall_seconds"] = perf_counter() - started
        report["peak_sampled_rss_bytes"] = max(peak_rss, process.memory_info().rss)
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restart-batch", type=Path)
    args = parser.parse_args()
    print(benchmark(args.protocol, args.output, args.restart_batch))


if __name__ == "__main__":
    main()
