"""Two-qubit sequential uploads, with matched trainable-block controls."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import cudaq
import numpy as np
from articles.day20.iris_models import load_data, split_data, fit_preprocessor, preprocess
from articles.day21.reduction import matrix
from articles.day03.quantum_gates import ry, Z
from articles.day04.bell_state import CNOT
from articles.day15.classifier import metrics, probabilities

# Indices into the transformed features, one pair per block; -1 means no upload.
SCHEDULES = {'pca_once2': [0, -1], 'pca_repeat2': [0, 0],
             'chunks_once4': [0, 2, -1, -1], 'chunks_repeat4': [0, 2, 0, 2]}
COUNTS = {name: 4 * len(schedule) for name, schedule in SCHEDULES.items()}
COUNTS['logistic4'] = 5


def resources(model):
    if model == 'logistic4':
        return {'qubits': 0, 'parameters': 5, 'upload_blocks': 0, 'ry_gates': 0, 'cnot_gates': 0, 'unoptimized_logical_depth': 0}
    schedule = SCHEDULES[model]
    layers, uploads = len(schedule), sum(i >= 0 for i in schedule)
    return {'qubits': 2, 'parameters': COUNTS[model], 'upload_blocks': uploads,
            'ry_gates': 4 * layers + 2 * uploads, 'cnot_gates': layers,
            'unoptimized_logical_depth': 3 * layers + uploads}


def fit(raw, labels):
    x = matrix(raw)
    pca = fit_preprocessor(x)
    return {'pca': pca, 'minimum': x.min(axis=0).tolist(), 'maximum': x.max(axis=0).tolist()}


def representation(model, raw, prep, weights):
    x = matrix(raw)
    if model not in COUNTS:
        raise ValueError('unknown model')
    w = np.asarray(weights, dtype=float)
    if w.shape != (COUNTS[model],) or not np.isfinite(w).all():
        raise ValueError('invalid weights')
    if model.startswith('pca_'):
        return preprocess(x, prep['pca'])
    if model == 'logistic4':
        z = (x - prep['pca']['mean']) / prep['pca']['std']
        return z, np.zeros_like(z, dtype=bool)
    low, high = np.array(prep['minimum']), np.array(prep['maximum'])
    flags = (x < low) | (x > high)
    return 2 * (np.clip(x, low, high) - low) / (high - low) - 1, flags


@cudaq.kernel
def circuit(angles: list[float], weights: list[float], schedule: list[int]):
    q = cudaq.qvector(2)
    for layer in range(len(schedule)):
        j = schedule[layer]
        if j >= 0:
            ry(angles[j], q[0])
            ry(angles[j + 1], q[1])
        k = 4 * layer
        ry(weights[k], q[0])
        ry(weights[k + 1], q[1])
        x.ctrl(q[0], q[1])
        ry(weights[k + 2], q[0])
        ry(weights[k + 3], q[1])


def reference(angles, weights, schedule):
    state = np.array([1, 0, 0, 0], dtype=complex)
    for layer, j in enumerate(schedule):
        if j >= 0:
            state = np.kron(ry(angles[j]), ry(angles[j + 1])) @ state
        a, b, c, d = weights[4 * layer:4 * layer + 4]
        state = np.kron(ry(a), ry(b)) @ state
        state = CNOT @ state
        state = np.kron(ry(c), ry(d)) @ state
    return float(np.vdot(state, np.kron(Z, Z) @ state).real)


def forward(model, raw, prep, weights, engine='numpy'):
    if engine not in ('numpy', 'cudaq'):
        raise ValueError('unknown engine')
    features, _ = representation(model, raw, prep, weights)
    w = np.asarray(weights, dtype=float)
    if model == 'logistic4':
        return .5 * (1 + np.tanh((features @ w[:4] + w[4]) / 2))
    schedule = SCHEDULES[model]
    values = []
    for row in features:
        angles = (np.pi * (row + 1) / 2).tolist()
        if engine == 'numpy':
            values.append(reference(angles, w, schedule))
        else:
            values.append(float(cudaq.observe(circuit, cudaq.spin.z(0) * cudaq.spin.z(1),
                                              angles, w.tolist(), schedule, shots_count=-1).expectation()))
    return probabilities(values)
