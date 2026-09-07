"""Verify register kernels against a NumPy reference across 36 configurations."""

import argparse
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import cudaq
import numpy as np

from kernels import register_circuit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from articles.day03.quantum_gates import KET_ZERO, ry


def validate(n: int, theta: float, entangle: bool, flip_last: bool,
             shots: int = 1000, seed: int = 42) -> None:
    if type(n) is not int or not 1 <= n <= 8:
        raise ValueError("demo qubit count must be an integer in [1, 8]")
    if not np.isfinite(theta):
        raise ValueError("theta must be finite")
    if type(entangle) is not bool or type(flip_last) is not bool:
        raise ValueError("circuit switches must be bool")
    if type(shots) is not int or shots <= 0:
        raise ValueError("shots must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be a 32-bit nonnegative integer")


def reference_state(n: int, theta: float, entangle: bool, flip_last: bool) -> np.ndarray:
    """Apply the Day 03 RY, then explicit computational-basis permutations."""
    validate(n, theta, entangle, flip_last)
    state = ry(theta) @ KET_ZERO
    for _ in range(n - 1):
        state = np.kron(state, KET_ZERO)
    # q0 is the most significant bit of the NumPy index.
    if entangle:
        for target in range(1, n):
            updated = np.zeros_like(state)
            for index, amplitude in enumerate(state):
                destination = index ^ (1 << (n - 1 - target)) if index & (1 << (n - 1)) else index
                updated[destination] = amplitude
            state = updated
    if flip_last:
        state = state[np.arange(2**n) ^ 1]
    return state


def run_case(n: int, theta: float, entangle: bool, flip_last: bool,
             shots: int = 1000, seed: int = 42) -> dict:
    validate(n, theta, entangle, flip_last, shots, seed)
    expected = reference_state(n, theta, entangle, flip_last)
    labels = [format(i, f"0{n}b") for i in range(2**n)]
    started = time.perf_counter()
    state = cudaq.get_state(register_circuit, n, theta, entangle, flip_last, False)
    # Address basis labels explicitly instead of assuming raw buffer endianness.
    actual = np.array([state.amplitude(label) for label in labels])
    cudaq.set_random_seed(seed)
    counts = {str(k): int(v) for k, v in cudaq.sample(
        register_circuit, n, theta, entangle, flip_last, True,
        shots_count=shots, explicit_measurements=True).items()}
    exact = np.abs(expected)**2
    observed = np.array([counts.get(label, 0) / shots for label in labels])
    fidelity = float(abs(np.vdot(expected, actual))**2)
    norm = float(np.vdot(actual, actual).real)
    checks = {"shots_conserved": sum(counts.values()) == shots,
              "valid_bitstrings": set(counts) <= set(labels),
              "normalized": bool(np.isclose(norm, 1, rtol=0, atol=1e-5)),
              "reference_fidelity": bool(np.isclose(fidelity, 1, rtol=0, atol=1e-5)),
              "probabilities_match": bool(np.allclose(np.abs(actual)**2, exact, rtol=0, atol=1e-5))}
    return {"qubits": n, "theta_radians": theta, "entangle": entangle, "flip_last": flip_last,
            "shots": shots, "seed": seed, "counts": counts, "norm": norm, "fidelity": fidelity,
            "tvd": float(np.abs(observed - exact).sum() / 2), "checks": checks,
            "passed": all(checks.values()), "elapsed_seconds": time.perf_counter() - started}


def run_experiment(backend: str) -> tuple[list[dict], dict]:
    if backend not in ("qpp-cpu", "nvidia"):
        raise ValueError("unsupported backend")
    cudaq.set_target(backend)
    rows = [run_case(n, float(theta), entangle, flip_last)
            for n in (2, 3, 4) for theta in (0.0, np.pi / 2, np.pi)
            for entangle in (False, True) for flip_last in (False, True)]
    summary = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "backend": backend, "target": cudaq.get_target().name,
               "environment": {"python": platform.python_version(), "numpy": np.__version__,
                               "cudaq": version("cuda-quantum-cu12"), "os": platform.platform(),
                               "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                               "precision": str(cudaq.get_target().get_precision())},
               "record_count": len(rows), "shots_per_case": 1000, "seed": 42,
               "noise_model": "none", "bit_order": "q0 q1 ... q(n-1)",
               "max_fidelity_error": max(abs(1-r["fidelity"]) for r in rows),
               "max_tvd": max(r["tvd"] for r in rows),
               "passed": all(r["passed"] for r in rows),
               "scope": "kernel correctness sweep, no training, statistical comparison or speed benchmark"}
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("qpp-cpu", "nvidia"), default="qpp-cpu")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    rows, summary = run_experiment(args.backend)
    output = args.output_dir or ROOT / "results/day07" / args.backend
    output.mkdir(parents=True, exist_ok=True)
    with (output / "kernel_sweep.csv").open("w", encoding="utf-8", newline="") as handle:
        flattened = [{**r, "counts": json.dumps(r["counts"], sort_keys=True),
                       "checks": json.dumps(r["checks"], sort_keys=True)} for r in rows]
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flattened)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "circuit.txt").write_text(cudaq.draw(register_circuit, 3, float(np.pi/2), True, False, True), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
