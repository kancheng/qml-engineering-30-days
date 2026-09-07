"""Fidelity kernel via compute-uncompute and independent state overlaps."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import cudaq
import numpy as np
from articles.day03.quantum_gates import ry
from articles.day04.bell_state import CNOT
from articles.day20.iris_models import load_data, split_data
from articles.day21.reduction import matrix
from articles.day15.classifier import metrics

MODELS = ('quantum', 'rbf', 'linear')


def fit_scaler(raw):
    x = matrix(raw)
    low, high = x.min(0), x.max(0)
    if np.any(high <= low):
        raise ValueError('constant training column')
    return {'minimum': low.tolist(), 'maximum': high.tolist()}


def transform(raw, prep):
    x = matrix(raw); low, high = np.array(prep['minimum']), np.array(prep['maximum'])
    return 2*(np.clip(x, low, high)-low)/(high-low)-1, (x < low) | (x > high)


def features(x):
    x = matrix(x)
    if np.any(abs(x) > 1):
        raise ValueError('scaled features must be in [-1,1]')
    return x


@cudaq.kernel
def overlap(a: list[float], b: list[float]):
    q = cudaq.qvector(2)
    # U(a) = RY(a2,a3) CNOT RY(a0,a1)
    ry(a[0], q[0]); ry(a[1], q[1])
    x.ctrl(q[0], q[1])
    ry(a[2], q[0]); ry(a[3], q[1])
    # U(b)^dagger: reverse order and negate rotation angles.
    ry(-b[2], q[0]); ry(-b[3], q[1])
    x.ctrl(q[0], q[1])
    ry(-b[0], q[0]); ry(-b[1], q[1])


def state(row):
    a = np.pi*(np.asarray(row)+1)/2
    v = np.kron(ry(a[0]), ry(a[1])) @ np.array([1,0,0,0], dtype=complex)
    return np.kron(ry(a[2]), ry(a[3])) @ CNOT @ v


def kernel(model, x, train, engine='numpy'):
    x, train = features(x), features(train)
    if model not in MODELS or engine not in ('numpy','cudaq'):
        raise ValueError('unknown kernel or engine')
    if model == 'linear':
        return 1 + x @ train.T / 4
    if model == 'rbf':
        return np.exp(-np.sum((x[:,None,:]-train[None,:,:])**2, axis=2))
    if engine == 'numpy':
        a, b = np.array([state(v) for v in x]), np.array([state(v) for v in train])
        return abs(a.conj() @ b.T)**2
    projector = (cudaq.spin.i(0)+cudaq.spin.z(0))*(cudaq.spin.i(1)+cudaq.spin.z(1))*.25
    a, b = np.pi*(x+1)/2, np.pi*(train+1)/2
    return np.array([[float(cudaq.observe(overlap, projector, u.tolist(), v.tolist(), shots_count=-1).expectation())
                      for v in b] for u in a])


def solve(gram, labels, regularization):
    k, y = np.asarray(gram, dtype=float), np.asarray(labels, dtype=float)
    if k.ndim != 2 or k.shape != (len(y),len(y)) or not len(y) or not np.isfinite(k).all() or not np.isin(y,[0,1]).all():
        raise ValueError('finite square Gram matrix and binary labels required')
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError('positive regularization required')
    return np.linalg.solve(k + regularization*np.eye(len(y)), y)


def diagnostics(k):
    return {'symmetry_error': float(np.max(abs(k-k.T))),
            'diagonal_error_from_one': float(np.max(abs(np.diag(k)-1))),
            'minimum_eigenvalue_symmetrized': float(np.linalg.eigvalsh((k+k.T)/2).min()),
            'minimum': float(k.min()), 'maximum': float(k.max())}
