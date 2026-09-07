"""Leakage boundaries, representation loss and independent circuit checks."""
import copy
import os
import unittest
import cudaq
from reduction import np, load_data, split_data, fit, forward, representation, COUNTS
from articles.day20.iris_models import fit_preprocessor, preprocess


class ReductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cudaq.set_target(os.getenv('DAY21_TARGET', 'qpp-cpu'))
        cls.rows, _ = load_data()
        cls.splits = split_data(cls.rows, 2028)
        cls.x = np.array([r['features'] for r in cls.splits['train']])
        cls.y = np.array([r['label'] for r in cls.splits['train']])
        cls.prep = fit(cls.x, cls.y)

    def test_split_and_train_only_transform(self):
        ids = [{r['id'] for r in rows} for rows in self.splits.values()]
        self.assertEqual(sum(map(len, ids)), 99)
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        before = copy.deepcopy(self.prep)
        raw = np.vstack([self.x[0], self.x[0] + 100])
        for model, n in COUNTS.items():
            representation(model, raw, self.prep, np.zeros(n))
        self.assertEqual(before, self.prep)
        np.testing.assert_allclose(self.prep['pca']['mean'], self.x.mean(axis=0))
        self.assertEqual(self.prep['pca'], fit_preprocessor(self.x))

    def test_supervised_selection_and_pca_parity(self):
        corr = np.corrcoef(np.column_stack([self.x, self.y]), rowvar=False)[:4, 4]
        np.testing.assert_allclose(corr, self.prep['train_correlations'])
        expected = sorted(range(4), key=lambda j: (-abs(corr[j]), j))[:2]
        self.assertEqual(self.prep['selected_indices'], expected)
        a, _ = representation('pca_vqc', self.x, self.prep, np.zeros(4))
        b, _ = preprocess(self.x, fit_preprocessor(self.x))
        np.testing.assert_allclose(a, b)

    def test_discarded_direction_collision(self):
        # PCA nullspace changes raw features while leaving the two PCs unchanged.
        components = np.array(self.prep['pca']['components'])
        _, _, vt = np.linalg.svd(components, full_matrices=True)
        mean = np.array(self.prep['pca']['mean'])
        pair = np.array([mean, mean + .1 * vt[2] * self.prep['pca']['std']])
        a, _ = representation('pca_vqc', pair, self.prep, np.zeros(4))
        self.assertGreater(np.linalg.norm(pair[0] - pair[1]), 0)
        np.testing.assert_allclose(a[0], a[1], atol=1e-12)

    def test_cpu_or_gpu_matches_reference(self):
        for model, n in COUNTS.items():
            w = np.random.default_rng(42).normal(0, .2, n)
            expected = forward(model, self.x[:3], self.prep, w)
            actual = forward(model, self.x[:3], self.prep, w, 'cudaq')
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)
        # Both encoder and quantum block actually affect this bottleneck output.
        w = np.random.default_rng(42).normal(0, .2, 14)
        p = forward('bottleneck_vqc', self.x[:3], self.prep, w)
        for j in (0, 10):
            changed = w.copy()
            changed[j] += .4
            self.assertGreater(np.max(abs(forward('bottleneck_vqc', self.x[:3], self.prep, changed) - p)), 1e-5)

    def test_invalid_data(self):
        with self.assertRaises(ValueError):
            fit(np.ones((5, 4)), [0, 0, 1, 1, 1])
        with self.assertRaises(ValueError):
            fit(self.x, np.zeros(len(self.x)))
        with self.assertRaises(ValueError):
            forward('bottleneck_vqc', self.x, self.prep, np.zeros(12))
        with self.assertRaises(ValueError):
            forward('pca_vqc', [[float('nan')] * 4], self.prep, np.zeros(4))


if __name__ == '__main__':
    unittest.main()
