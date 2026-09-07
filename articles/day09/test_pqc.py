"""Test PQC contracts, analytic limiting cases and independent NumPy agreement."""

import os
import unittest

import cudaq
import numpy as np

from pqc import initialize, inputs, parameter_count, predict, reference_prediction, sample_prediction


class PqcTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cudaq.set_target(os.environ.get('DAY09_TARGET','qpp-cpu'))

    def test_shapes_finiteness_and_bounds(self):
        self.assertEqual(parameter_count(2),8)
        for x,w,layers in [([0],np.zeros(4),1),([2,0],np.zeros(4),1),([np.nan,0],np.zeros(4),1),
                           ([0,0],np.zeros(3),1),([0,0],[0,0,0,np.inf],1),([0,0],[],0)]:
            with self.assertRaises(ValueError): inputs(x,w,layers)

    def test_initialization_repeats_without_mutating_caller(self):
        w=initialize(2)
        np.testing.assert_array_equal(w,initialize(2))
        self.assertFalse(np.array_equal(w,initialize(2,43)))
        saved=w.copy()
        predict([0.1,0.2],w,2)
        np.testing.assert_array_equal(w,saved)

    def test_zero_weights_have_analytic_odd_even_cnot_behavior(self):
        x=[0.25,-0.4]
        for layers in (1,2,3):
            expected=np.cos(np.pi*x[1]) if layers%2 else np.cos(np.pi*x[0])*np.cos(np.pi*x[1])
            self.assertAlmostEqual(predict(x,np.zeros(parameter_count(layers)),layers),expected,places=5)

    def test_nonzero_weights_match_independent_matrix_reference(self):
        for layers in (1,2):
            for x in ([0,0],[0.25,-0.4],[-0.75,0.2]):
                w=initialize(layers)
                self.assertAlmostEqual(predict(x,w,layers),reference_prediction(x,w,layers),places=5)

    def test_each_parameter_changes_output_at_generic_setting(self):
        x=[0.25,-0.4]; w=initialize()
        base=predict(x,w)
        for index in range(4):
            candidate=w.copy(); candidate[index]+=0.5
            self.assertGreater(abs(predict(x,candidate)-base),1e-3)

    def test_rotation_periodicity(self):
        x=[0.25,-0.4]; w=initialize()
        base=predict(x,w)
        for index in range(4):
            candidate=w.copy(); candidate[index]+=2*np.pi
            self.assertAlmostEqual(predict(x,candidate),base,places=5)

    def test_sampling_parity_and_seed(self):
        w=np.zeros(4)
        self.assertEqual(sample_prediction([0,0],w)['prediction'],1)
        self.assertEqual(sample_prediction([0,1],w)['prediction'],-1)
        result=sample_prediction([0.25,-0.4],initialize())
        self.assertEqual(sum(result['counts'].values()),1000)
        self.assertEqual(result,sample_prediction([0.25,-0.4],initialize()))


if __name__ == '__main__':
    unittest.main()
