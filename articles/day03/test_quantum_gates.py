"""Correctness tests for the Day 03 NumPy gate implementation."""

import unittest

import numpy as np

from quantum_gates import (
    H,
    KET_ONE,
    KET_ZERO,
    X,
    Y,
    Z,
    apply_gate,
    equivalent_up_to_global_phase,
    is_unitary,
    probabilities,
    rx,
    ry,
    rz,
)


class QuantumGateTests(unittest.TestCase):
    def test_all_gates_are_unitary(self) -> None:
        gates = [X, Y, Z, H, rx(0.37), ry(-1.2), rz(np.pi)]
        self.assertTrue(all(is_unitary(gate) for gate in gates))

    def test_pauli_gate_actions(self) -> None:
        np.testing.assert_allclose(apply_gate(X, KET_ZERO), KET_ONE, atol=1e-12)
        np.testing.assert_allclose(apply_gate(Y, KET_ZERO), 1j * KET_ONE, atol=1e-12)
        np.testing.assert_allclose(apply_gate(Z, KET_ZERO), KET_ZERO, atol=1e-12)

    def test_hadamard_is_its_own_inverse(self) -> None:
        np.testing.assert_allclose(apply_gate(H, apply_gate(H, KET_ZERO)), KET_ZERO, atol=1e-12)

    def test_hadamard_creates_equal_z_probabilities(self) -> None:
        np.testing.assert_allclose(probabilities(apply_gate(H, KET_ZERO)), [0.5, 0.5], atol=1e-12)

    def test_rotation_endpoints(self) -> None:
        self.assertTrue(equivalent_up_to_global_phase(apply_gate(rx(np.pi), KET_ZERO), KET_ONE))
        self.assertTrue(equivalent_up_to_global_phase(apply_gate(ry(np.pi), KET_ZERO), KET_ONE))
        self.assertTrue(equivalent_up_to_global_phase(apply_gate(rz(np.pi), KET_ZERO), KET_ZERO))

    def test_relative_phase_becomes_measurable_after_hadamard(self) -> None:
        plus = apply_gate(H, KET_ZERO)
        minus = apply_gate(Z, plus)
        np.testing.assert_allclose(probabilities(plus), probabilities(minus), atol=1e-12)
        np.testing.assert_allclose(probabilities(apply_gate(H, plus)), [1, 0], atol=1e-12)
        np.testing.assert_allclose(probabilities(apply_gate(H, minus)), [0, 1], atol=1e-12)

    def test_rejects_non_unitary_matrix(self) -> None:
        invalid = np.array([[1, 1], [0, 1]], dtype=np.complex128)
        with self.assertRaisesRegex(ValueError, "unitary"):
            apply_gate(invalid, KET_ZERO)


if __name__ == "__main__":
    unittest.main()
