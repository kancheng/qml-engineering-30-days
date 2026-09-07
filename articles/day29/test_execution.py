import copy
import json
import math
import unittest
from unittest.mock import patch
from execution import (ROOT, MODES, configure, estimate, validate_counts, reference,
                       submission_plan, ordering_probe, cudaq)
from report import audit


class ExecutionTests(unittest.TestCase):
    def test_asymmetric_bit_order_and_parity(self):
        self.assertEqual(estimate({'10': 20}, 20, 'Z0')['value'], -1)
        self.assertEqual(estimate({'10': 20}, 20, 'ZZ')['value'], -1)
        self.assertEqual(estimate({'11': 20}, 20, 'ZZ')['value'], 1)

    def test_wilson_boundaries(self):
        low = estimate({'10': 100}, 100, 'Z0')['wilson95']
        high = estimate({'00': 100}, 100, 'Z0')['wilson95']
        self.assertAlmostEqual(low[0], -1)
        self.assertAlmostEqual(high[1], 1)
        self.assertGreater(high[1]-high[0], 0)
        wide = estimate({'00': 5, '10': 5}, 10, 'Z0')['wilson95']
        narrow = estimate({'00': 50, '10': 50}, 100, 'Z0')['wilson95']
        self.assertGreater(wide[1]-wide[0], narrow[1]-narrow[0])

    def test_invalid_counts(self):
        for counts, shots in [({}, 1), ({'0': 1}, 1), ({'00': -1}, 1),
                              ({'00': 2}, 1), ({'00': 1.0}, 1), ({'00': 1}, 0)]:
            with self.assertRaises(ValueError):
                validate_counts(counts, shots)

    def test_noise_limits(self):
        self.assertAlmostEqual(reference(math.pi/3, 0)['Z0'], .5)
        self.assertEqual(reference(math.pi/3, .5)['ZZ'], 0)
        self.assertEqual(reference(math.pi/3, 1)['ZZ'], -1)
        self.assertEqual(reference(math.pi/3, 0)['XX'], reference(math.pi/3, 1)['XX'])

    def test_emulation_cannot_select_remote(self):
        with patch.object(cudaq, 'set_target') as target:
            configure('ionq-emulate')
            target.assert_called_once_with('ionq', emulate=True)
        with self.assertRaises(ValueError):
            configure('ionq')

    def test_offline_budget(self):
        plan = submission_plan()
        self.assertEqual(plan['total_requested_shots'], plan['measurement_circuits']*plan['shots_per_circuit'])
        self.assertEqual(plan['status'], 'not_submitted')
        self.assertIsNone(plan['machine'])
        self.assertEqual(plan['job_ids'], [])

    def test_saved_artifacts_and_corruption(self):
        for mode in MODES:
            data = json.loads((ROOT/f'results/day29/{mode}.json').read_text())
            audit(data)
            corrupt = copy.deepcopy(data)
            corrupt['records'][0]['counts'] = {'10': 1}
            with self.assertRaises((ValueError, AssertionError)):
                audit(corrupt)

    def test_actual_cpu_ordering(self):
        configure('qpp-cpu')
        counts = dict(cudaq.sample(ordering_probe, shots_count=32).items())
        self.assertEqual(counts, {'10': 32})


if __name__ == '__main__':
    unittest.main()
