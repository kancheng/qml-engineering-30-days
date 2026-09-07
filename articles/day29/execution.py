"""Local simulator-to-QPU rehearsal. Every executable target stays local."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import math
import cudaq
from articles.day08.kernels import prepare
from articles.day26.noise import noise_model

MODES = ('qpp-cpu', 'ionq-emulate', 'density-bitflip')
SHOTS = (128, 1024, 8192)
SEEDS = (42, 43, 44, 45, 46)
THETA = math.pi / 3
NOISE_P = .08
OBSERVABLES = {0: ('Z0', 'ZZ'), 1: ('X0', 'XX')}


@cudaq.kernel
def state(theta: float):
    q = cudaq.qvector(2)
    prepare(q, theta)
    rz(0.0, q[0])  # Day26 marker: one bit-flip channel on q0, if enabled.


@cudaq.kernel
def measured(theta: float, basis: int):
    q = cudaq.qvector(2)
    prepare(q, theta)
    rz(0.0, q[0])
    if basis == 1:
        h(q[0])
        h(q[1])
    mz(q[0])
    mz(q[1])


@cudaq.kernel
def ordering_probe():
    q = cudaq.qvector(2)
    x(q[0])
    mz(q[0])
    mz(q[1])


def configure(mode):
    if mode not in MODES:
        raise ValueError('Only named local modes are supported')
    cudaq.unset_noise()
    if mode == 'ionq-emulate':
        cudaq.set_target('ionq', emulate=True)
    else:
        cudaq.set_target('density-matrix-cpu' if mode == 'density-bitflip' else 'qpp-cpu')


def reference(theta, p):
    return {'Z0': (1-2*p)*math.cos(theta), 'ZZ': 1-2*p,
            'X0': 0., 'XX': math.sin(theta)}


def validate_counts(counts, shots):
    if type(shots) is not int or shots <= 0:
        raise ValueError('shots must be a positive integer')
    if not counts or any(k not in ('00', '01', '10', '11') or type(v) is not int or v < 0
                         for k, v in counts.items()) or sum(counts.values()) != shots:
        raise ValueError('Expected two-bit q0 q1 counts summing to shots')


def estimate(counts, shots, name):
    """Expectation and pointwise 95% Wilson interval for a +/-1 observable."""
    validate_counts(counts, shots)
    if name not in ('Z0', 'ZZ', 'X0', 'XX'):
        raise ValueError('unknown observable')
    plus = sum(v for k, v in counts.items()
               if (k[0] == '0' if name.endswith('0') else k[0] == k[1]))
    f = plus / shots
    z = 1.959963984540054
    denominator = 1 + z*z/shots
    center = (f + z*z/(2*shots)) / denominator
    radius = z * math.sqrt(f*(1-f)/shots + z*z/(4*shots*shots)) / denominator
    return dict(value=2*f-1, plus_count=plus,
                wilson95=[max(-1., 2*(center-radius)-1), min(1., 2*(center+radius)-1)])


def noise_kwargs(mode):
    return {'noise_model': noise_model('bit_flip', NOISE_P)} if mode == 'density-bitflip' else {}


def sample_counts(mode, basis, shots, seed):
    if mode not in MODES or basis not in OBSERVABLES:
        raise ValueError('invalid mode or basis')
    if type(shots) is not int or shots <= 0:
        raise ValueError('positive integer shots required')
    cudaq.set_random_seed(seed)
    # Terminal mz fixes the intended schema. Hardware lowering may not support
    # explicit_measurements=True; ordering is verified by an asymmetric probe.
    raw = cudaq.sample(measured, THETA, basis, shots_count=shots, **noise_kwargs(mode))
    counts = {str(k): int(v) for k, v in raw.items()}
    validate_counts(counts, shots)
    return counts


def submission_plan():
    """A reviewable offline plan; does not create or submit a remote job."""
    return dict(status='not_submitted', provider_example='ionq', machine=None,
                execution_kind='physical_qpu_requested_only_after_device_selection',
                theta=THETA, qubits=2, bases=['Z', 'X'], shots_per_circuit=1024,
                parameter_sets=1, repeats=1, measurement_circuits=2, total_requested_shots=2048,
                max_automatic_resubmissions=0, job_ids=[],
                queue_seconds=None, device_execution_seconds=None, cost=None,
                required_before_submission=['account access', 'available physical device identifier',
                                            'device constraints and calibration snapshot',
                                            'provider quote and shot limits'],
                noise_model='physical device noise; no synthetic channel requested',
                note='Logical circuit/shot budget, not provider billing or job-count guarantee.')
