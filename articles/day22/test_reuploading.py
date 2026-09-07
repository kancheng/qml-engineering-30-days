"""Independent references, schedule semantics and preprocessing boundaries."""
import copy
import os
import unittest
import cudaq
from reuploading import np, load_data, split_data, fit, forward, representation, COUNTS, SCHEDULES, resources, reference
from articles.day14.model import reference_prediction, Config
from articles.day03.quantum_gates import ry


class ReuploadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cudaq.set_target(os.getenv('DAY22_TARGET', 'qpp-cpu'))
        rows, _ = load_data()
        cls.splits = split_data(rows, 2028)
        cls.raw = np.array([r['features'] for r in cls.splits['train']])
        cls.prep = fit(cls.raw, [r['label'] for r in cls.splits['train']])

    def test_train_only_and_clipping(self):
        saved = copy.deepcopy(self.prep)
        raw = np.array([self.prep['minimum'], self.prep['maximum']])
        x, flags = representation('chunks_repeat4', raw, self.prep, np.zeros(16))
        np.testing.assert_allclose(x, [[-1]*4, [1]*4])
        self.assertFalse(flags.any())
        _, flags = representation('chunks_once4', raw + 100, self.prep, np.zeros(16))
        self.assertTrue(flags.all())
        self.assertEqual(saved, self.prep)
        self.assertEqual(self.prep, fit(self.raw, np.zeros(len(self.raw))))

    def test_single_upload_matches_day14(self):
        w = np.random.default_rng(42).normal(0, .2, 8)
        features, _ = representation('pca_once2', self.raw[:3], self.prep, w)
        expected = [(1-reference_prediction(x, w, Config(layers=2)))/2 for x in features]
        np.testing.assert_allclose(forward('pca_once2', self.raw[:3], self.prep, w), expected, atol=1e-12)

    def test_targets_and_schedules(self):
        for model, n in COUNTS.items():
            for seed in (42, 43):
                w = np.random.default_rng(seed).normal(0, .2, n)
                np.testing.assert_allclose(forward(model, self.raw[:3], self.prep, w, 'cudaq'),
                                           forward(model, self.raw[:3], self.prep, w), atol=1e-5, rtol=0)
        for a, b in [('pca_once2', 'pca_repeat2'), ('chunks_once4', 'chunks_repeat4')]:
            self.assertEqual(COUNTS[a], COUNTS[b])
            self.assertEqual(resources(a)['cnot_gates'], resources(b)['cnot_gates'])
            w = np.random.default_rng(42).normal(0, .2, COUNTS[a])
            self.assertGreater(np.max(abs(forward(a, self.raw[:3], self.prep, w)-forward(b, self.raw[:3], self.prep, w))), 1e-4)

    def test_both_chunks_influence_output(self):
        w = np.random.default_rng(42).normal(0, .2, 16)
        a = np.array([.2, .4, .6, .8])
        for model in ('chunks_once4', 'chunks_repeat4'):
            base = reference(a, w, SCHEDULES[model])
            for j in range(4):
                b = a.copy(); b[j] += .3
                self.assertGreater(abs(base-reference(b, w, SCHEDULES[model])), 1e-6)

    def test_adjacent_rotations_collapse(self):
        for a, b in [(.2, .4), (-1., .7)]:
            np.testing.assert_allclose(ry(b) @ ry(a), ry(a+b), atol=1e-12)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            forward('pca_repeat2', self.raw, self.prep, np.zeros(4))
        with self.assertRaises(ValueError):
            forward('chunks_repeat4', [[float('nan')]*4], self.prep, np.zeros(16))
        with self.assertRaises(ValueError):
            fit(np.ones((5, 4)), [0, 0, 1, 1, 1])
        with self.assertRaises(ValueError):
            forward('missing', self.raw, self.prep, [])


if __name__ == '__main__':
    unittest.main()
