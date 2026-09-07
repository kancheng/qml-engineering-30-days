"""Two-qubit NumPy reference. Basis order: |q0 q1> = 00, 01, 10, 11."""

import numpy as np

ZERO = np.array([1, 0], dtype=np.complex128)
H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
I = np.eye(2, dtype=np.complex128)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                 [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128)
LABELS = ("00", "01", "10", "11")
STATES = ("product", "bell", "mixture")


def bell_state() -> np.ndarray:
    return CNOT @ np.kron(H, I) @ np.kron(ZERO, ZERO)


def density_matrix(name: str) -> np.ndarray:
    if name == "mixture":
        return np.diag([0.5, 0.0, 0.0, 0.5]).astype(np.complex128)
    if name == "bell":
        state = bell_state()
    elif name == "product":
        state = np.kron(H @ ZERO, H @ ZERO)
    else:
        raise ValueError(f"unknown state: {name}")
    return np.outer(state, state.conj())


def exact_probabilities(name: str, basis: str) -> np.ndarray:
    rho = density_matrix(name)
    if basis == "X":
        rotation = np.kron(H, H)
        rho = rotation @ rho @ rotation.conj().T
    elif basis != "Z":
        raise ValueError(f"unknown basis: {basis}")
    return np.diag(rho).real.copy()


def sample_counts(name: str, basis: str, shots: int, seed: int) -> dict[str, int]:
    if shots <= 0:
        raise ValueError("shots must be positive")
    counts = np.random.default_rng(seed).multinomial(shots, exact_probabilities(name, basis))
    return dict(zip(LABELS, map(int, counts)))


if __name__ == "__main__":
    print("Bell amplitudes [00, 01, 10, 11]:", bell_state())
    for name in STATES:
        for basis in ("Z", "X"):
            print(name, basis, "exact:", exact_probabilities(name, basis),
                  "counts:", sample_counts(name, basis, 1000, 42))
