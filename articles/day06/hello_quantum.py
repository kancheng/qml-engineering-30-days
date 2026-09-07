"""Small CUDA-Q installation smoke test with an explicit simulation backend."""

import argparse
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import cudaq
import numpy as np


@cudaq.kernel
def bell(measure: bool):
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])
    if measure:
        mz(q[0])
        mz(q[1])


@cudaq.kernel
def bit_order_probe():
    q = cudaq.qvector(2)
    x(q[0])
    mz(q[0])
    mz(q[1])


def loaded_gpu_libraries() -> list[str]:
    """Linux loader observation, not proof that every loaded library was used."""
    maps = Path("/proc/self/maps")
    if not maps.exists():
        return []
    names = ("libcuda.so", "libcudart.so", "libcustatevec.so", "libcutensornet.so")
    return sorted({Path(line.split()[-1]).name for line in maps.read_text().splitlines()
                   if any(name in line for name in names)})


def run_smoke(backend: str, shots: int = 1000, seed: int = 42) -> dict:
    if backend not in ("qpp-cpu", "nvidia"):
        raise ValueError("choose qpp-cpu or nvidia")
    if shots <= 0 or not 0 <= seed < 2**32:
        raise ValueError("shots must be positive and seed must be an unsigned 32-bit integer")
    started = time.perf_counter()
    cudaq.set_target(backend)
    cudaq.set_random_seed(seed)
    counts = {str(k): int(v) for k, v in cudaq.sample(
        bell, True, shots_count=shots, explicit_measurements=True).items()}
    state = np.array(cudaq.get_state(bell, False))
    expected = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
    fidelity = float(abs(np.vdot(expected, state)) ** 2)
    order_counts = {str(k): int(v) for k, v in cudaq.sample(
        bit_order_probe, shots_count=16, explicit_measurements=True).items()}
    checks = {"target_matches_request": cudaq.get_target().name == backend,
              "shots_conserved": sum(counts.values()) == shots,
              "bell_support_only": set(counts) <= {"00", "11"},
              "state_matches_bell": bool(np.isclose(fidelity, 1, rtol=0, atol=1e-5)),
              "bit_order_q0_q1": order_counts == {"10": 16}}
    return {"backend": backend, "target": cudaq.get_target().name,
            "python": platform.python_version(), "numpy": np.__version__,
            "cudaq_distribution": "cuda-quantum-cu12", "cudaq_version": version("cuda-quantum-cu12"),
            "qubits": 2, "shots": shots, "seed": seed, "noise_model": "none",
            "counts": counts, "fidelity": fidelity, "bit_order_counts": order_counts,
            "checks": checks, "passed": all(checks.values()),
            "loaded_gpu_libraries": loaded_gpu_libraries(),
            "elapsed_seconds": time.perf_counter() - started,
            "timing_note": "smoke test includes compilation and initialization, not a benchmark"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("qpp-cpu", "nvidia"), default="qpp-cpu")
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_smoke(args.backend, args.shots, args.seed)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
