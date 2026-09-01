"""Minimal single-qubit gate simulator used by Day 03.

The module intentionally depends only on NumPy. It exposes matrices and
helpers explicitly so readers can inspect every state transformation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]

KET_ZERO: ComplexArray = np.array([1.0, 0.0], dtype=np.complex128)
KET_ONE: ComplexArray = np.array([0.0, 1.0], dtype=np.complex128)

I: ComplexArray = np.eye(2, dtype=np.complex128)
X: ComplexArray = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y: ComplexArray = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z: ComplexArray = np.array([[1, 0], [0, -1]], dtype=np.complex128)
H: ComplexArray = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)


def rx(theta: float) -> ComplexArray:
    """Return RX(theta) = exp(-i theta X / 2)."""
    half = theta / 2
    return np.cos(half) * I - 1j * np.sin(half) * X


def ry(theta: float) -> ComplexArray:
    """Return RY(theta) = exp(-i theta Y / 2)."""
    half = theta / 2
    return np.cos(half) * I - 1j * np.sin(half) * Y


def rz(theta: float) -> ComplexArray:
    """Return RZ(theta) = exp(-i theta Z / 2)."""
    half = theta / 2
    return np.cos(half) * I - 1j * np.sin(half) * Z


def is_unitary(matrix: ComplexArray, *, atol: float = 1e-12) -> bool:
    """Check U†U = I for a square matrix."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    identity = np.eye(matrix.shape[0], dtype=np.complex128)
    return bool(np.allclose(matrix.conj().T @ matrix, identity, atol=atol))


def apply_gate(gate: ComplexArray, state: ComplexArray) -> ComplexArray:
    """Apply a unitary single-qubit gate to a normalized state."""
    gate = np.asarray(gate, dtype=np.complex128)
    state = np.asarray(state, dtype=np.complex128)
    if gate.shape != (2, 2):
        raise ValueError(f"gate must have shape (2, 2); received {gate.shape}")
    if state.shape != (2,):
        raise ValueError(f"state must have shape (2,); received {state.shape}")
    if not is_unitary(gate):
        raise ValueError("gate must be unitary")
    if not np.isclose(np.vdot(state, state).real, 1.0):
        raise ValueError("state must be normalized")
    return gate @ state


def probabilities(state: ComplexArray) -> NDArray[np.float64]:
    """Return probabilities in the computational (Z) basis."""
    state = np.asarray(state, dtype=np.complex128)
    if state.shape != (2,):
        raise ValueError(f"state must have shape (2,); received {state.shape}")
    if not np.isclose(np.vdot(state, state).real, 1.0):
        raise ValueError("state must be normalized")
    return np.abs(state) ** 2


def expectation(state: ComplexArray, observable: ComplexArray) -> float:
    """Return the real expectation value <state|observable|state>."""
    value = np.vdot(state, observable @ state)
    if not np.isclose(value.imag, 0.0, atol=1e-12):
        raise ValueError("expectation value is not real; observable may be invalid")
    return float(value.real)


def equivalent_up_to_global_phase(
    first: ComplexArray, second: ComplexArray, *, atol: float = 1e-12
) -> bool:
    """Check whether two normalized state vectors differ only by global phase."""
    overlap = np.vdot(first, second)
    return bool(np.isclose(abs(overlap), 1.0, atol=atol))
