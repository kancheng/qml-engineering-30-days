"""Deterministic physical checks and finite-shot invariants."""

import unittest

import numpy as np

from bell_state import CNOT, STATES, bell_state, density_matrix, exact_probabilities, sample_counts


class BellStateTests(unittest.TestCase):
    def test_cnot_truth_table_and_unitarity(self):
        for source, destination in enumerate((0, 1, 3, 2)):
            np.testing.assert_array_equal(CNOT @ np.eye(4)[:, source], np.eye(4)[:, destination])
        np.testing.assert_allclose(CNOT.conj().T @ CNOT, np.eye(4), atol=1e-12)

    def test_bell_amplitudes_and_entanglement(self):
        state = bell_state()
        np.testing.assert_allclose(state, np.array([1, 0, 0, 1]) / np.sqrt(2), atol=1e-12)
        # For a normalized bipartite pure state, rank > 1 means entangled.
        self.assertEqual(np.linalg.matrix_rank(state.reshape(2, 2)), 2)
        reduced = state.reshape(2, 2) @ state.reshape(2, 2).conj().T
        np.testing.assert_allclose(reduced, np.eye(2) / 2, atol=1e-12)

    def test_density_matrices_are_physical(self):
        for name in STATES:
            rho = density_matrix(name)
            np.testing.assert_allclose(rho, rho.conj().T, atol=1e-12)
            self.assertAlmostEqual(float(np.trace(rho).real), 1)
            self.assertGreaterEqual(float(np.linalg.eigvalsh(rho).min()), -1e-12)

    def test_z_correlation_alone_does_not_distinguish_mixture(self):
        np.testing.assert_allclose(exact_probabilities("bell", "Z"), [0.5, 0, 0, 0.5], atol=1e-12)
        np.testing.assert_allclose(exact_probabilities("bell", "Z"), exact_probabilities("mixture", "Z"), atol=1e-12)
        np.testing.assert_allclose(exact_probabilities("bell", "X"), [0.5, 0, 0, 0.5], atol=1e-12)
        np.testing.assert_allclose(exact_probabilities("mixture", "X"), [0.25] * 4, atol=1e-12)

    def test_product_superposition_is_not_bell_state(self):
        np.testing.assert_allclose(exact_probabilities("product", "Z"), [0.25] * 4, atol=1e-12)
        np.testing.assert_allclose(exact_probabilities("product", "X"), [1, 0, 0, 0], atol=1e-12)

    def test_sampling_conserves_shots_and_seed(self):
        for name in STATES:
            for basis in ("Z", "X"):
                counts = sample_counts(name, basis, 1000, 42)
                self.assertEqual(sum(counts.values()), 1000)
                self.assertEqual(counts, sample_counts(name, basis, 1000, 42))
        counts = sample_counts("bell", "Z", 1000, 42)
        self.assertEqual(counts["01"] + counts["10"], 0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            sample_counts("bell", "Z", 0, 42)
        with self.assertRaises(ValueError):
            exact_probabilities("bell", "Y")
        with self.assertRaises(ValueError):
            density_matrix("unknown")


if __name__ == "__main__":
    unittest.main()
