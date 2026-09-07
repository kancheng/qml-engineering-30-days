import os
import unittest
import numpy as np
import cudaq
from train import coordinate_search, dataset, mse
from articles.day09.pqc import predict, reference_prediction


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cudaq.set_target(os.environ.get('DAY10_TARGET', 'qpp-cpu'))

    def test_mse_validation(self):
        self.assertEqual(mse([1, 3], [0, 1]), 2.5)
        for p,y in [([],[]),([1],[1,2]),([float('nan')],[0])]:
            with self.assertRaises(ValueError): mse(p,y)

    def test_optimizer_quadratic_and_accounting(self):
        initial = np.array([2., -1.])
        w, history, reason = coordinate_search(lambda w: float(w@w), initial, sweeps=80)
        self.assertLess(float(w@w), 1e-6)
        np.testing.assert_array_equal(initial, [2., -1.])
        self.assertEqual(history[-1]['evaluations'], 1+4*(len(history)-1))
        self.assertTrue(all(b['loss'] <= a['loss'] for a,b in zip(history, history[1:])))
        self.assertEqual(reason, 'step_tolerance')

    def test_budget_and_invalid_objective(self):
        _, history, reason = coordinate_search(lambda w: float(w@w), [3.], sweeps=1)
        self.assertEqual(reason, 'max_sweeps')
        self.assertEqual(len(history), 2)
        with self.assertRaises(ValueError): coordinate_search(lambda w: np.nan, [0.])
        with self.assertRaises(ValueError): coordinate_search(lambda w: 0., [0.], sweeps=0)

    def test_realizable_target_and_holdout_separation(self):
        train, y, held, held_y = dataset()
        self.assertFalse(set(map(tuple, train)) & set(map(tuple, held)))
        for x,label in zip(np.vstack([train,held]), np.r_[y,held_y]):
            self.assertAlmostEqual(predict(x,[0.,0.6,0.,0.]), label, delta=1e-5)

    def test_hybrid_update_against_numpy(self):
        features, labels, _, _ = dataset()
        def objective(w): return mse([predict(x,w) for x in features],labels)
        w, history, _ = coordinate_search(objective, [0.,0.,0.,0.], sweeps=3)
        self.assertLess(history[-1]['loss'], history[0]['loss'])
        self.assertAlmostEqual(history[-1]['loss'], mse([reference_prediction(x,w) for x in features],labels), delta=1e-5)

if __name__ == '__main__':
    unittest.main()
