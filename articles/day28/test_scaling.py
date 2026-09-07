"""Check model invariants and the committed report's provenance."""
import hashlib
import json
import unittest
from scaling import ROOT, capacity, latency


class ScalingTests(unittest.TestCase):
    def test_memory_boundary(self):
        self.assertEqual(capacity(1, gib=1, fraction=1, complex_bytes=16), 26)
        self.assertEqual(capacity(2), capacity(1) + 1)
        self.assertEqual(capacity(1, complex_bytes=8), capacity(1) + 1)
        self.assertEqual(capacity(1, gib=1, fraction=.5), 25)

    def test_latency_limits(self):
        self.assertEqual(latency(2, 1, .3, .2), 2)
        self.assertEqual(latency(2, 8, 0, 0), .25)
        self.assertEqual(latency(2, 8, 1, 0), 2)
        self.assertGreater(latency(2, 8, .3, .3), 2)

    def test_invalid_inputs(self):
        for p in (0, -1, 3):
            with self.assertRaises(ValueError):
                capacity(p)
        with self.assertRaises(ValueError):
            latency(1, 2, 1.1, 0)

    def test_saved_results(self):
        data = json.loads((ROOT / 'results/day28/model.json').read_text())
        source = ROOT / data['baseline_source']
        self.assertEqual(data['baseline_sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(len(data['capacities']), 16)
        self.assertEqual(len(data['scenarios']), 12)
        for row in data['scenarios']:
            expected = latency(data['baseline_seconds'], row['gpus'], row['serial_fraction'], row['overhead_fraction'])
            self.assertAlmostEqual(row['modeled_seconds'], expected)
            self.assertAlmostEqual(row['speedup'], data['baseline_seconds']/expected)


if __name__ == '__main__':
    unittest.main()
