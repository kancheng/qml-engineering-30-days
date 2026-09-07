"""Train-only preprocessing and an explicit two-feature encoding boundary."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import cudaq
import numpy as np
from articles.day09.pqc import feature_map


def matrix(values):
    data = np.asarray(values, dtype=float)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] != 2 or not np.isfinite(data).all():
        raise ValueError('expected a nonempty finite matrix with two feature columns')
    return data


def fit_scaler(train):
    data = matrix(train)
    low, high = data.min(axis=0), data.max(axis=0)
    if np.any(high == low):
        raise ValueError('constant training columns require an explicit feature policy')
    return {'minimum': low.tolist(), 'maximum': high.tolist(), 'policy': 'clip to training range'}


def transform(values, scaler):
    data = matrix(values)
    low, high = np.asarray(scaler['minimum']), np.asarray(scaler['maximum'])
    if low.shape != (2,) or high.shape != (2,) or not np.isfinite([low, high]).all() or np.any(high <= low):
        raise ValueError('invalid scaler bounds')
    outside = (data < low) | (data > high)
    scaled = 2*(np.clip(data, low, high)-low)/(high-low)-1
    return scaled, outside


def angle_reference(features):
    x = np.asarray(features, dtype=float)
    if x.shape != (2,) or not np.isfinite(x).all() or np.any(np.abs(x)>1):
        raise ValueError('expected two finite scaled features in [-1,1]')
    a = np.pi*x/2
    return np.kron([np.cos(a[0]),np.sin(a[0])], [np.cos(a[1]),np.sin(a[1])])


def basis_reference(bits):
    b = np.asarray(bits)
    if b.shape != (2,) or not np.isin(b, [0,1]).all():
        raise ValueError('basis demo expects two bits')
    state = np.zeros(4)
    state[int(2*b[0]+b[1])] = 1
    return state


def amplitude_reference(values):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or not 2 <= a.size <= 4 or not np.isfinite(a).all() or not np.any(a):
        raise ValueError('amplitude demo expects 2-4 finite real values, not all zero')
    a = np.pad(a, (0, (1 << (a.size-1).bit_length())-a.size))
    a = a/np.max(np.abs(a))  # Avoid overflow when normalizing large finite values.
    return a/np.linalg.norm(a)


@cudaq.kernel
def encoded(angles: list[float]):
    q = cudaq.qvector(2)
    feature_map(q, angles)


@cudaq.kernel
def measured(angles: list[float]):
    q = cudaq.qvector(2)
    feature_map(q, angles)
    mz(q[0])
    mz(q[1])
