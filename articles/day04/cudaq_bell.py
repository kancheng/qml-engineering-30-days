"""CUDA-Q circuits with explicit q0, q1 measurement order."""

import cudaq
import numpy as np

from bell_state import LABELS


@cudaq.kernel
def circuit(kind: int, x_basis: bool, measure: bool):
    q = cudaq.qvector(2)
    if kind == 0:
        h(q[0])
        h(q[1])
    elif kind == 1:
        h(q[0])
        x.ctrl(q[0], q[1])
    elif kind == 3:
        x(q[0])
        x(q[1])
    if x_basis:
        h(q[0])
        h(q[1])
    if measure:
        mz(q[0])
        mz(q[1])


@cudaq.kernel
def ordering_probe():
    q = cudaq.qvector(2)
    x(q[0])
    mz(q[0])
    mz(q[1])


def sample_counts(name: str, basis: str, shots: int, seed: int) -> dict[str, int]:
    if shots <= 0:
        raise ValueError("shots must be positive")
    if name not in ("product", "bell", "mixture") or basis not in ("Z", "X"):
        raise ValueError("invalid state or basis")
    cudaq.set_random_seed(seed)
    if name == "mixture":
        # Classical preparation choice: independent fair coin on each shot.
        # Batch the chosen |00> and |11> preparations for efficient simulation.
        zeros = int(np.random.default_rng(seed).binomial(shots, 0.5))
        batches = ((2, zeros), (3, shots - zeros))
    else:
        batches = ((0 if name == "product" else 1, shots),)
    counts = dict.fromkeys(LABELS, 0)
    for kind, count in batches:
        if count:
            result = cudaq.sample(circuit, kind, basis == "X", True,
                                  shots_count=count, explicit_measurements=True)
            for bitstring, frequency in result.items():
                counts[bitstring] += int(frequency)
    return counts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("qpp-cpu", "nvidia"), default="qpp-cpu")
    args = parser.parse_args()
    cudaq.set_target(args.backend)
    print("target:", cudaq.get_target().name)
    print(cudaq.draw(circuit, 1, False, True))
    print("Bell Z, 1000 shots, seed 42:", sample_counts("bell", "Z", 1000, 42))
