"""Compare product, Bell and classical mixture states across bases and shots."""

import argparse
import csv
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

from bell_state import LABELS, STATES, exact_probabilities, sample_counts


def run_experiment(backend: str) -> tuple[list[dict], dict]:
    sampler = sample_counts
    environment = {"python": platform.python_version(), "numpy": np.__version__,
                   "os": platform.platform(), "machine": platform.machine(),
                   "backend": backend}
    if backend != "numpy":
        import cudaq
        from cudaq_bell import sample_counts as cudaq_sample_counts

        cudaq.set_target(backend)
        sampler = cudaq_sample_counts
        environment["cudaq"] = version("cuda-quantum-cu12")
        environment["target"] = cudaq.get_target().name

    records = []
    start = time.perf_counter()
    for name in STATES:
        for basis in ("Z", "X"):
            expected = exact_probabilities(name, basis)
            for shots in (100, 1000, 10000):
                for seed in range(42, 47):
                    tick = time.perf_counter()
                    counts = sampler(name, basis, shots, seed)
                    elapsed = time.perf_counter() - tick
                    assert sum(counts.values()) == shots
                    observed = np.array([counts[label] / shots for label in LABELS])
                    records.append({
                        "state": name, "basis": basis, "shots": shots, "seed": seed,
                        **{f"count_{label}": counts[label] for label in LABELS},
                        "p_equal": float(observed[0] + observed[3]),
                        "correlation": float(observed[0] - observed[1] - observed[2] + observed[3]),
                        "total_variation_distance": float(np.abs(observed - expected).sum() / 2),
                        "elapsed_seconds": elapsed,
                    })
    aggregates = []
    for name in STATES:
        for basis in ("Z", "X"):
            for shots in (100, 1000, 10000):
                rows = [r for r in records if (r["state"], r["basis"], r["shots"]) == (name, basis, shots)]
                errors = [r["total_variation_distance"] for r in rows]
                aggregates.append({"state": name, "basis": basis, "shots": shots,
                                   "mean_tvd": float(np.mean(errors)),
                                   "std_tvd": float(np.std(errors, ddof=1)),
                                   "mean_correlation": float(np.mean([r["correlation"] for r in rows]))})
    summary = {
        "environment": environment, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "qubits": 2, "bit_order": "q0 q1; 00, 01, 10, 11",
        "shots": [100, 1000, 10000], "seeds": list(range(42, 47)),
        "noise_model": "none", "execution_seconds": time.perf_counter() - start,
        "record_count": len(records),
        "mixture_preparation": "fair classical choice of |00> or |11>; CUDA-Q batches preparations",
        "timing_note": "includes initial JIT/runtime overhead; not a CPU/GPU benchmark",
        "exact_probabilities": {name: {basis: exact_probabilities(name, basis).tolist()
                                       for basis in ("Z", "X")} for name in STATES},
        "aggregates": aggregates,
    }
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "qpp-cpu", "nvidia"), default="numpy")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or Path(__file__).resolve().parents[2] / "results" / "day04" / args.backend
    records, summary = run_experiment(args.backend)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "shots_sweep.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} records to {output}")


if __name__ == "__main__":
    main()
