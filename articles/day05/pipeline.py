"""Day 05: scalar input -> encoding -> circuit -> counts -> classical output."""

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from articles.day03.quantum_gates import I, KET_ZERO, ry
from articles.day04.bell_state import CNOT, LABELS


def encode(value: float) -> float:
    """The demo accepts an already scaled scalar in [0, 1]."""
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("input must be finite and in [0, 1]")
    return float(np.pi * value)


def state_vector(value: float) -> np.ndarray:
    return CNOT @ np.kron(ry(encode(value)), I) @ np.kron(KET_ZERO, KET_ZERO)


def prediction_from_counts(counts: dict[str, int]) -> float:
    """Estimate <Z0> using q0 as the first bit; retain all four outcomes."""
    if set(counts) - set(LABELS):
        raise ValueError("expected two-bit q0 q1 outcomes")
    if any(not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 0
           for n in counts.values()):
        raise ValueError("counts must be nonnegative integers")
    shots = sum(counts.values())
    if shots == 0:
        raise ValueError("counts must contain at least one shot")
    return float(sum((1 if label[0] == "0" else -1) * n
                     for label, n in counts.items()) / shots)


def run_pipeline(backend: str = "numpy", shots: int = 1000,
                 seeds: tuple[int, ...] = (42, 43, 44)) -> tuple[list[dict], dict]:
    if backend not in ("numpy", "qpp-cpu", "nvidia"):
        raise ValueError("unknown backend")
    if not isinstance(shots, int) or isinstance(shots, bool) or shots <= 0:
        raise ValueError("shots must be a positive integer")
    if not seeds or any(not isinstance(s, int) or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("provide nonnegative 32-bit integer seeds")
    environment = {"python": platform.python_version(), "numpy": np.__version__,
                   "os": platform.platform(), "backend": backend}
    if backend != "numpy":
        import cudaq
        from cudaq_pipeline import sample_counts

        cudaq.set_target(backend)
        environment.update(cudaq=version("cuda-quantum-cu12"), target=cudaq.get_target().name)
    rows = []
    started = time.perf_counter()
    for index, value in enumerate(np.linspace(0, 1, 9)):
        theta = encode(float(value))
        expected = float(np.cos(theta))
        probabilities = np.abs(state_vector(float(value))) ** 2
        for seed in seeds:
            # Unique streams for each input within a repeat; record the actual seed.
            sampling_seed = int(np.random.SeedSequence([seed, index]).generate_state(1)[0])
            tick = time.perf_counter()
            if backend == "numpy":
                counts = dict(zip(LABELS, map(int, np.random.default_rng(sampling_seed).multinomial(shots, probabilities))))
            else:
                counts = sample_counts(theta, shots, sampling_seed)
            prediction = prediction_from_counts(counts)
            if sum(counts.values()) != shots:
                raise RuntimeError("sampler returned incorrect shot count")
            rows.append({"input": float(value), "theta_radians": theta,
                         "repeat_seed": seed, "sampling_seed": sampling_seed, "shots": shots,
                         **{f"count_{label}": counts.get(label, 0) for label in LABELS},
                         "target": expected, "prediction": prediction,
                         "squared_error": (prediction - expected) ** 2,
                         "expected_sampling_variance": (1 - expected**2) / shots,
                         "elapsed_seconds": time.perf_counter() - tick})
    summary = {
        "environment": environment, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "known-function forward-pass verification, no training or held-out evaluation",
        "dataset": "nine equally spaced scalar inputs in [0, 1]; target cos(pi*x)",
        "encoding": "theta=pi*x; RY(theta) on q0", "circuit": "RY(q0) -> CNOT(q0,q1)",
        "qubits": 2, "preparation_depth": 2, "trainable_parameters": 0,
        "observable": "Z0", "bit_order": "q0 q1", "noise_model": "none",
        "shots": shots, "repeat_seeds": list(seeds), "record_count": len(rows),
        "sampled_mse": float(np.mean([r["squared_error"] for r in rows])),
        "expected_sampled_mse": float(np.mean([r["expected_sampling_variance"] for r in rows])),
        "constant_zero_baseline_mse": float(np.mean([r["target"]**2 for r in rows])),
        "analytic_classical_reference_mse": 0.0,
        "baseline_note": "cos(pi*x) is the known generating rule; zero error by construction",
        "per_seed_mse": {str(seed): float(np.mean([r["squared_error"] for r in rows
                                                   if r["repeat_seed"] == seed])) for seed in seeds},
        "execution_seconds": time.perf_counter() - started,
        "timing_note": "includes compilation/initialization; not a performance benchmark",
    }
    return rows, summary


def write_results(rows: list[dict], summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "qpp-cpu", "nvidia"), default="numpy")
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    rows, summary = run_pipeline(args.backend, args.shots)
    output = args.output_dir or ROOT / "results" / "day05" / args.backend
    write_results(rows, summary, output)
    print(f"backend={args.backend}, records={len(rows)}, MSE={summary['sampled_mse']:.8f}")
    print(f"results: {output}")


if __name__ == "__main__":
    main()
