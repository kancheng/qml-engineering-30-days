"""Validate aggregation contracts and reject corrupt saved evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from evidence import ROOT, Sources, audit_benchmark, metrics, compare_metrics, pick, build, render


class EvidenceTests(unittest.TestCase):
    def test_metrics_threshold_and_confusion(self):
        result = metrics([.1, .5, .9, .2], [0, 0, 1, 1])
        self.assertEqual(result['confusion_matrix'], [[1, 1], [1, 1]])
        self.assertEqual(result['accuracy'], .5)
        self.assertAlmostEqual(result['brier'], (.01+.25+.01+.64)/4)

    def test_invalid_metrics(self):
        for p, y in [([], []), ([.1], [0, 1]), ([float('nan')], [0]),
                     ([1.1], [1]), ([.2], [2])]:
            with self.assertRaises(ValueError):
                metrics(p, y)
        with self.assertRaises(ValueError):
            compare_metrics(metrics([.1], [0]), metrics([.9], [0]))

    def test_selection_ignores_test_and_breaks_ties(self):
        candidates = [{'seed': 43, 'validation': {'brier': .1}, 'test': {'brier': 0}},
                      {'seed': 42, 'validation': {'brier': .1}, 'test': {'brier': 1}}]
        self.assertEqual(pick(candidates, 'seed')['seed'], 42)
        candidates[0]['validation']['brier'] = .09
        self.assertEqual(pick(candidates, 'seed')['seed'], 43)

    def test_sources_record_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'source.json'
            path.write_text('{"value": 1}\n')
            sources = Sources(root)
            sources.read('source.json')
            self.assertEqual(sources.hashes['source.json'], hashlib.sha256(path.read_bytes()).hexdigest())
            original = sources.hashes['source.json']
            path.write_text('{"value": 2}\n')
            sources.read('source.json')
            self.assertNotEqual(original, sources.hashes['source.json'])

    def test_reject_corrupt_benchmark(self):
        class CorruptSources(Sources):
            def read(self, name):
                data = super().read(name)
                if name.endswith('/selected.json'):
                    data[0]['scores']['test']['probabilities'][0] = float('nan')
                return data
        with self.assertRaises(ValueError):
            audit_benchmark(CorruptSources(), 25)

    def test_reject_missing_selected_model(self):
        class MissingSources(Sources):
            def read(self, name):
                data = super().read(name)
                return data[:-1] if name.endswith('/selected.json') else data
        with self.assertRaises(ValueError):
            audit_benchmark(MissingSources(), 20)

    def test_saved_report_and_scope(self):
        data = build()
        self.assertEqual(data, json.loads((ROOT/'results/day30/evidence.json').read_text()))
        self.assertEqual(render(data), (ROOT/'results/day30/README.md').read_text())
        self.assertEqual(len(data['benchmarks']), 20)
        self.assertEqual(len(data['paired_deltas']), 8)
        self.assertEqual(data['qpu']['physical_qpu_jobs'], 0)
        self.assertEqual(sum(r['shots'] for r in data['qpu']['local_modes']), 280320)
        self.assertEqual([r['day'] for r in data['inventory']], list(range(1, 31)))


if __name__ == '__main__':
    unittest.main()
