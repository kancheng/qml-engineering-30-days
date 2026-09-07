"""Generate the reproducible Day 03 single-qubit gate experiment."""

from __future__ import annotations

import csv
import json
import platform
from pathlib import Path

import numpy as np

from quantum_gates import H, KET_ZERO, X, Y, Z, apply_gate, expectation, probabilities, rx, ry, rz


def state_record(gate_name: str, theta: float, state: np.ndarray) -> dict[str, float | str]:
    probs = probabilities(state)
    return {
        "gate": gate_name,
        "theta_radians": float(theta),
        "p0": float(probs[0]),
        "p1": float(probs[1]),
        "expectation_x": expectation(state, X),
        "expectation_y": expectation(state, Y),
        "expectation_z": expectation(state, Z),
        "norm": float(np.vdot(state, state).real),
    }


def run_experiment() -> list[dict[str, float | str]]:
    angles = np.linspace(0.0, np.pi, num=5)
    records: list[dict[str, float | str]] = []
    for gate_name, gate_factory in (("RX", rx), ("RY", ry), ("RZ", rz)):
        for theta in angles:
            state = apply_gate(gate_factory(float(theta)), KET_ZERO)
            records.append(state_record(gate_name, float(theta), state))
    return records


def write_results(records: list[dict[str, float | str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "gate_sweep.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    h_state = apply_gate(H, KET_ZERO)
    summary = {
        "runtime": "NumPy CPU state-vector simulation",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "os": platform.system(),
            "machine": platform.machine(),
            "backend": "numpy-cpu",
            "dtype": "complex128",
        },
        "measurement": "exact probabilities; no finite shots",
        "random_seed": None,
        "input_state": "|0>",
        "angle_count_per_rotation": 5,
        "theta_range_radians": [0.0, float(np.pi)],
        "max_norm_error": max(abs(float(row["norm"]) - 1.0) for row in records),
        "h_on_zero_probabilities": probabilities(h_state).tolist(),
        "checks": {
            "x_on_zero_p1": float(probabilities(apply_gate(X, KET_ZERO))[1]),
            "ry_pi_on_zero_p1": float(probabilities(apply_gate(ry(np.pi), KET_ZERO))[1]),
            "rz_preserves_z_probabilities": bool(
                all(np.isclose(float(row["p0"]), 1.0) for row in records if row["gate"] == "RZ")
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    output_dir = repository_root / "results" / "day03"
    records = run_experiment()
    write_results(records, output_dir)
    print(f"wrote {len(records)} records to {output_dir}")


if __name__ == "__main__":
    main()
