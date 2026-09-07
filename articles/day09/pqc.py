"""Two-feature, two-qubit PQC with separate data and weight arguments."""

from pathlib import Path
import sys

import cudaq
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from articles.day03.quantum_gates import KET_ZERO, Z, ry
from articles.day04.bell_state import CNOT


def parameter_count(layers: int) -> int:
    if type(layers) is not int or not 1 <= layers <= 3:
        raise ValueError('demo layers must be an integer in [1, 3]')
    return 4 * layers


def inputs(features, weights, layers: int) -> tuple[list[float], list[float]]:
    count = parameter_count(layers)
    x = np.asarray(features, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x.shape != (2,) or not np.all(np.isfinite(x)) or np.any(np.abs(x) > 1):
        raise ValueError('features must contain two finite values in [-1, 1]')
    if w.shape != (count,) or not np.all(np.isfinite(w)):
        raise ValueError(f'weights must contain {count} finite angles')
    return (np.pi * x).tolist(), w.tolist()


def initialize(layers: int = 1, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 0.2, parameter_count(layers))


@cudaq.kernel
def feature_map(q: cudaq.qview, angles: list[float]):
    ry(angles[0], q[0])
    ry(angles[1], q[1])


@cudaq.kernel
def ansatz(q: cudaq.qview, weights: list[float], layers: int):
    for layer in range(layers):
        offset = 4 * layer
        ry(weights[offset], q[0])
        ry(weights[offset + 1], q[1])
        x.ctrl(q[0], q[1])
        ry(weights[offset + 2], q[0])
        ry(weights[offset + 3], q[1])


@cudaq.kernel
def model(angles: list[float], weights: list[float], layers: int):
    q = cudaq.qvector(2)
    feature_map(q, angles)
    ansatz(q, weights, layers)


@cudaq.kernel
def sampled_model(angles: list[float], weights: list[float], layers: int):
    q = cudaq.qvector(2)
    feature_map(q, angles)
    ansatz(q, weights, layers)
    mz(q[0])
    mz(q[1])


def reference_state(features, weights, layers: int = 1) -> np.ndarray:
    angles, w = inputs(features, weights, layers)
    state = np.kron(ry(angles[0]) @ KET_ZERO, ry(angles[1]) @ KET_ZERO)
    for layer in range(layers):
        offset = 4 * layer
        state = np.kron(ry(w[offset]), ry(w[offset + 1])) @ state
        state = CNOT @ state
        state = np.kron(ry(w[offset + 2]), ry(w[offset + 3])) @ state
    return state


def reference_prediction(features, weights, layers: int = 1) -> float:
    state = reference_state(features, weights, layers)
    return float(np.vdot(state, np.kron(Z, Z) @ state).real)


def predict(features, weights, layers: int = 1) -> float:
    """Exact simulator expectation; caller selects the CUDA-Q target once."""
    angles, w = inputs(features, weights, layers)
    return float(cudaq.observe(model, cudaq.spin.z(0) * cudaq.spin.z(1),
                               angles, w, layers, shots_count=-1).expectation())


def sample_prediction(features, weights, layers: int = 1, shots: int = 1000, seed: int = 42) -> dict:
    angles, w = inputs(features, weights, layers)
    if type(shots) is not int or shots <= 0:
        raise ValueError('shots must be a positive integer')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be a nonnegative 32-bit integer')
    cudaq.set_random_seed(seed)
    counts = {str(k): int(v) for k,v in cudaq.sample(sampled_model, angles, w, layers,
                                                   shots_count=shots, explicit_measurements=True).items()}
    # ZZ is +1 for equal bits and -1 for unequal bits.
    value = sum((1 if k[0] == k[1] else -1) * v for k,v in counts.items()) / shots
    return {'counts': counts, 'prediction': value, 'shots': shots, 'seed': seed}
