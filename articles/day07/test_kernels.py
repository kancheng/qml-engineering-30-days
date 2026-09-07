"""Kernel semantics and host input contracts; DAY07_TARGET selects GPU tests."""

import os
import unittest

import cudaq
import numpy as np

from experiment import reference_state, run_case, validate
from kernels import single_rotation


class KernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cudaq.set_target(os.environ.get("DAY07_TARGET", "qpp-cpu"))

    def test_single_qubit_allocation(self):
        self.assertEqual(dict(cudaq.sample(single_rotation, 0.0, shots_count=32).items()), {"0": 32})
        self.assertEqual(dict(cudaq.sample(single_rotation, float(np.pi), shots_count=32).items()), {"1": 32})

    def test_reference_ghz_and_disabled_control(self):
        expected = np.zeros(8)
        expected[[0, 7]] = 1/np.sqrt(2)
        np.testing.assert_allclose(reference_state(3, float(np.pi/2), True, False), expected, atol=1e-12)
        expected = np.zeros(8)
        expected[[0, 4]] = 1/np.sqrt(2)
        np.testing.assert_allclose(reference_state(3, float(np.pi/2), False, False), expected, atol=1e-12)

    def test_loop_bounds_including_single_qubit(self):
        for n in (1, 2, 4):
            self.assertTrue(run_case(n, float(np.pi/2), True, False)["passed"])

    def test_conditional_branches_and_asymmetric_bit_order(self):
        for entangle, flip_last, bitstring in ((False, False, "100"), (False, True, "101"),
                                               (True, False, "111"), (True, True, "110")):
            result = run_case(3, float(np.pi), entangle, flip_last)
            self.assertTrue(result["passed"])
            self.assertEqual(result["counts"], {bitstring: 1000})

    def test_seed_reproducibility(self):
        a = run_case(3, 0.7, True, True)
        b = run_case(3, 0.7, True, True)
        self.assertEqual(a["counts"], b["counts"])
        self.assertTrue(a["passed"] and b["passed"])

    def test_host_rejects_invalid_inputs(self):
        for n in (0, 9, True, 2.5):
            with self.assertRaises(ValueError):
                validate(n, 0.0, True, False)
        for theta in (np.nan, np.inf):
            with self.assertRaises(ValueError):
                validate(2, theta, True, False)
        for kwargs in ({"shots": 0}, {"seed": -1}):
            with self.assertRaises(ValueError):
                validate(2, 0.0, True, False, **kwargs)


if __name__ == "__main__":
    unittest.main()
