"""Day 02: verify normalization, Born probabilities, and finite shots."""

import numpy as np


def measurement_probabilities(state: np.ndarray) -> np.ndarray:
    """Return computational-basis probabilities for a normalized state."""
    norm = np.vdot(state, state).real
    if not np.isclose(norm, 1.0):
        raise ValueError(f"state must be normalized; received norm={norm}")
    return np.abs(state) ** 2


def main() -> None:
    alpha = 1 / np.sqrt(2)
    beta = 1j / np.sqrt(2)
    state = np.array([alpha, beta], dtype=np.complex128)
    probabilities = measurement_probabilities(state)

    shots = 1_000
    rng = np.random.default_rng(seed=42)
    samples = rng.choice([0, 1], size=shots, p=probabilities)
    counts = np.bincount(samples, minlength=2)

    print(f"state = {state}")
    print(f"norm = {np.vdot(state, state).real:.6f}")
    print(f"P(0) = {probabilities[0]:.6f}")
    print(f"P(1) = {probabilities[1]:.6f}")
    print(f"shots = {shots}, seed = 42")
    print(f"counts = {{'0': {counts[0]}, '1': {counts[1]}}}")


if __name__ == "__main__":
    main()
