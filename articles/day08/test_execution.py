"""Validate return contracts and physical expectations on either local backend."""

import os
import unittest

import cudaq
import numpy as np

from experiment import compare, counts_from_returns, z0_from_counts


class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cudaq.set_target(os.environ.get('DAY08_TARGET','qpp-cpu'))

    def test_counts_and_integer_mapping_preserve_first_bit(self):
        self.assertEqual(counts_from_returns([0,1,2,3,3]), {'00':1,'01':1,'10':1,'11':2})
        self.assertAlmostEqual(z0_from_counts(counts_from_returns([0,1,2,3,3])), -0.2)
        self.assertEqual(z0_from_counts({'01':4}), 1)
        self.assertEqual(z0_from_counts({'10':4}), -1)

    def test_invalid_host_inputs(self):
        for values in ([],[4],[-1],[True]):
            with self.assertRaises(ValueError): counts_from_returns(values)
        for counts in ({},{'00':0},{'01':-1},{'0':1}):
            with self.assertRaises(ValueError): z0_from_counts(counts)
        with self.assertRaises(ValueError): compare(np.nan)
        with self.assertRaises(ValueError): compare(0.0,shots=0)
        with self.assertRaises(ValueError): compare(0.0,seed=-1)

    def test_deterministic_endpoints_across_apis(self):
        for theta, value, expected in ((0.0,0,1),(float(np.pi),3,-1)):
            result=compare(theta,shots=32)
            self.assertTrue(result['passed'])
            self.assertEqual(result['run_values'],[value]*32)
            for estimate in result['estimates'].values():
                self.assertAlmostEqual(estimate,expected,places=5)

    def test_nontrivial_state_and_observables(self):
        result=compare(float(np.pi/3),shots=256)
        self.assertTrue(result['passed'])
        self.assertAlmostEqual(result['estimates']['observe_exact'],0.5,places=5)
        self.assertAlmostEqual(result['observe_exact_xx'],np.sqrt(3)/2,places=5)
        for api in ('sample','run','observe_shots'):
            self.assertLess(abs(result['estimates'][api]-0.5),0.25)
        self.assertEqual(result['run_histogram'],counts_from_returns(result['run_values']))

    def test_repeated_seed_within_each_api(self):
        first=compare(float(np.pi/2),shots=32,seed=43)
        second=compare(float(np.pi/2),shots=32,seed=43)
        for key in ('counts','run_values','run_histogram','estimates'):
            self.assertEqual(first[key],second[key])


if __name__ == '__main__':
    unittest.main()
