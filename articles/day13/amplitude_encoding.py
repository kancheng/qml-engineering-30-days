"""Small real-vector amplitude encoding: simulator loading and explicit gates."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import cudaq
import numpy as np


def normalize(values):
    if np.iscomplexobj(values):
        raise ValueError('this gate demo supports real values only')
    v = np.asarray(values, dtype=float)
    if v.ndim != 1 or not 2 <= v.size <= 4 or not np.isfinite(v).all() or not np.any(v):
        raise ValueError('expected 2-4 finite real values, not all zero')
    n = (v.size-1).bit_length()
    padded = np.pad(v, (0, 2**n-v.size))
    scale = float(np.max(np.abs(padded)))
    scaled = padded/scale
    norm_scaled = float(np.linalg.norm(scaled))
    return scaled/norm_scaled, {'features':v.tolist(), 'original_dimension':len(v),
                                'padded_dimension':len(padded), 'qubits':n,
                                'norm_scale':scale, 'norm_scaled':norm_scaled}


def gate_angles(amplitudes):
    a = np.asarray(amplitudes, dtype=float)
    if a.shape not in ((2,), (4,)) or not np.isfinite(a).all() or not np.isclose(a@a,1.,atol=1e-12,rtol=0):
        raise ValueError('expected a normalized real vector of length 2 or 4')
    if len(a)==2:
        return [float(2*np.arctan2(a[1],a[0])),0.,0.]
    left,right=np.linalg.norm(a[:2]),np.linalg.norm(a[2:])
    alpha=2*np.arctan2(right,left)
    b0=2*np.arctan2(a[1],a[0]) if left>0 else 0.
    b1=2*np.arctan2(a[3],a[2]) if right>0 else 0.
    return [float(alpha),float((b0+b1)/2),float((b0-b1)/2)]


@cudaq.kernel
def prepare(q: cudaq.qview, angles: list[float], n: int):
    ry(angles[0],q[0])
    if n==2:
        ry(angles[1],q[1])
        x.ctrl(q[0],q[1])
        ry(angles[2],q[1])
        x.ctrl(q[0],q[1])


@cudaq.kernel
def gates(angles: list[float], n: int):
    q=cudaq.qvector(n)
    prepare(q,angles,n)


@cudaq.kernel
def sampled_gates(angles: list[float], n: int):
    q=cudaq.qvector(n)
    prepare(q,angles,n)
    for i in range(n):
        mz(q[i])


@cudaq.kernel
def loaded(state: cudaq.State):
    q=cudaq.qvector(state)


def labels(n):
    return [format(i,f'0{n}b') for i in range(2**n)]


def simulator_state(amplitudes):
    """Convert logical |q0 q1> order into the CUDA-Q little-endian buffer."""
    n=(len(amplitudes)-1).bit_length()
    order=[int(label[::-1],2) for label in labels(n)]
    dtype=np.complex64 if cudaq.get_target().get_precision()==cudaq.SimulationPrecision.fp32 else np.complex128
    return cudaq.State.from_data(np.asarray(amplitudes,dtype=dtype)[order].copy())


def extract(state,n):
    return np.array([state.amplitude(label) for label in labels(n)])
