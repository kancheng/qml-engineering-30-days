"""Run readable single-qubit demonstrations: python articles/day03/demo.py."""

import numpy as np

from quantum_gates import H, KET_ZERO, X, Z, apply_gate, probabilities, ry


def main() -> None:
    examples = {
        "X|0>": apply_gate(X, KET_ZERO),
        "H|0>": apply_gate(H, KET_ZERO),
        "HH|0>": apply_gate(H, apply_gate(H, KET_ZERO)),
        "HZH|0>": apply_gate(H, apply_gate(Z, apply_gate(H, KET_ZERO))),
        "RY(pi/2)|0>": apply_gate(ry(np.pi / 2), KET_ZERO),
    }
    for label, state in examples.items():
        p0, p1 = probabilities(state)
        print(f"{label:14s} P(0)={p0:.6f} P(1)={p1:.6f}")


if __name__ == "__main__":
    main()
