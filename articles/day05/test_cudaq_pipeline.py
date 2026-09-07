"""CPU by default; DAY05_TARGET=nvidia runs the same integration on GPU."""

import os
import unittest

import cudaq
import numpy as np

from cudaq_pipeline import encoded_pair, sample_counts
from pipeline import prediction_from_counts, run_pipeline, state_vector


class CudaqPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = os.environ.get("DAY05_TARGET", "qpp-cpu")
        cudaq.set_target(cls.backend)

    def test_state_matches_reference_at_multiple_inputs(self):
        for value in (0, 0.125, 0.5, 1):
            state = np.array(cudaq.get_state(encoded_pair, float(np.pi * value), False))
            self.assertAlmostEqual(float(abs(np.vdot(state_vector(value), state))**2), 1, places=5)

    def test_endpoints_and_seed_reproducibility(self):
        for theta, expected in ((0.0, 1), (float(np.pi), -1)):
            self.assertEqual(prediction_from_counts(sample_counts(theta, 128, 42)), expected)
        first = sample_counts(float(np.pi / 3), 1000, 42)
        self.assertEqual(first, sample_counts(float(np.pi / 3), 1000, 42))

    def test_complete_pipeline_is_consistent_with_sampling_error(self):
        rows, summary = run_pipeline(self.backend, shots=1000, seeds=(42,))
        self.assertEqual(len(rows), 9)
        for row in rows:
            self.assertEqual(row["count_01"] + row["count_10"], 0)
        self.assertLess(summary["sampled_mse"], 0.01)


if __name__ == "__main__":
    unittest.main()
