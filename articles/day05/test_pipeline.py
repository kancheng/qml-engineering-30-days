"""Test the full forward pass against independently derived expectations."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline import encode, prediction_from_counts, run_pipeline, state_vector, write_results


class PipelineTests(unittest.TestCase):
    def test_encoding_rejects_out_of_domain_values(self):
        for value in (-0.1, 1.1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                encode(value)

    def test_state_matches_analytic_amplitudes(self):
        for value in (0, 0.125, 0.5, 0.875, 1):
            theta = np.pi * value
            expected = [np.cos(theta / 2), 0, 0, np.sin(theta / 2)]
            np.testing.assert_allclose(state_vector(value), expected, atol=1e-12)

    def test_postprocessing_uses_first_bit_including_unequal_outcomes(self):
        self.assertAlmostEqual(prediction_from_counts({"00": 20, "01": 30, "10": 10, "11": 40}), 0)
        self.assertEqual(prediction_from_counts({"01": 10}), 1)
        self.assertEqual(prediction_from_counts({"10": 10}), -1)
        for counts in ({}, {"00": -1}, {"0": 2}, {"00": 0.5}):
            with self.assertRaises(ValueError):
                prediction_from_counts(counts)

    def test_pipeline_contract_reproducibility_and_saved_metrics(self):
        rows, summary = run_pipeline()
        repeated, _ = run_pipeline()
        self.assertEqual(len(rows), 27)
        for first, second in zip(rows, repeated):
            self.assertEqual({k: v for k, v in first.items() if k != "elapsed_seconds"},
                             {k: v for k, v in second.items() if k != "elapsed_seconds"})
            self.assertEqual(sum(first[f"count_{key}"] for key in ("00", "01", "10", "11")), 1000)
        self.assertAlmostEqual(summary["constant_zero_baseline_mse"], 5 / 9)
        self.assertAlmostEqual(summary["expected_sampled_mse"], 4 / 9000)
        with tempfile.TemporaryDirectory() as temp:
            write_results(rows, summary, Path(temp))
            loaded = json.loads((Path(temp) / "summary.json").read_text())
            self.assertAlmostEqual(loaded["sampled_mse"], np.mean([r["squared_error"] for r in rows]))


if __name__ == "__main__":
    unittest.main()
