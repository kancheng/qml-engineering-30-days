"""CUDA-Q integration tests; DAY04_TARGET=nvidia selects the GPU backend."""

import os
import unittest

import cudaq
import numpy as np

from bell_state import bell_state
from cudaq_bell import circuit, ordering_probe, sample_counts


class CudaqBellTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cudaq.set_target(os.environ.get("DAY04_TARGET", "qpp-cpu"))

    def test_bell_state_matches_numpy_up_to_global_phase(self):
        state = np.array(cudaq.get_state(circuit, 1, False, False))
        self.assertAlmostEqual(float(abs(np.vdot(bell_state(), state)) ** 2), 1, places=5)

    def test_explicit_measurement_bit_order(self):
        counts = dict(cudaq.sample(ordering_probe, shots_count=32, explicit_measurements=True).items())
        self.assertEqual(counts, {"10": 32})

    def test_bell_support_in_both_bases(self):
        for basis in ("Z", "X"):
            counts = sample_counts("bell", basis, 1000, 42)
            self.assertEqual(sum(counts.values()), 1000)
            self.assertEqual(counts["01"] + counts["10"], 0)
            self.assertGreater(counts["00"], 0)
            self.assertGreater(counts["11"], 0)

    def test_product_x_basis_is_deterministic(self):
        self.assertEqual(sample_counts("product", "X", 128, 42),
                         {"00": 128, "01": 0, "10": 0, "11": 0})

    def test_mixture_sampling_and_repeatability(self):
        for basis in ("Z", "X"):
            counts = sample_counts("mixture", basis, 1000, 42)
            self.assertEqual(sum(counts.values()), 1000)
            self.assertEqual(counts, sample_counts("mixture", basis, 1000, 42))
            if basis == "X":
                # Loose distribution check; exact theory is tested without shots.
                self.assertTrue(all(150 < count < 350 for count in counts.values()))
            else:
                self.assertEqual(counts["01"] + counts["10"], 0)


if __name__ == "__main__":
    unittest.main()
