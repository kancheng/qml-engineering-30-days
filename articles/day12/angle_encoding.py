"""Two-feature angle maps and rotation-axis experiments."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import cudaq
import numpy as np
from articles.day03.quantum_gates import KET_ZERO, H, X, Y, Z, I, rx, ry, rz
from articles.day11.encoding import fit_scaler, transform

MAPS = ('centered_full', 'centered_half', 'positive_half')
AXES = ('ry', 'rx', 'rz', 'h_rz')
LABELS = ('00', '01', '10', '11')


def to_angles(features, mapping='positive_half'):
    x = np.asarray(features, dtype=float)
    if x.shape != (2,) or not np.isfinite(x).all() or np.any(np.abs(x) > 1):
        raise ValueError('features must contain two finite values in [-1,1]')
    if mapping not in MAPS:
        raise ValueError('unknown angle mapping')
    if mapping == 'centered_full':
        return (np.pi*x).tolist()
    if mapping == 'centered_half':
        return (np.pi*x/2).tolist()
    return (np.pi*(x+1)/2).tolist()


def validate(angles, axis):
    a = np.asarray(angles, dtype=float)
    if a.shape != (2,) or not np.isfinite(a).all() or axis not in AXES:
        raise ValueError('expected two finite radians and a supported axis')
    return a.tolist(), AXES.index(axis)


@cudaq.kernel
def prepare(q: cudaq.qview, angles: list[float], axis: int):
    for i in range(2):
        if axis == 0:
            ry(angles[i], q[i])
        elif axis == 1:
            rx(angles[i], q[i])
        elif axis == 2:
            rz(angles[i], q[i])
        else:
            h(q[i])
            rz(angles[i], q[i])


@cudaq.kernel
def encoded(angles: list[float], axis: int):
    q = cudaq.qvector(2)
    prepare(q, angles, axis)


@cudaq.kernel
def measured(angles: list[float], axis: int):
    q = cudaq.qvector(2)
    prepare(q, angles, axis)
    mz(q[0])
    mz(q[1])


def reference(angles, axis='ry'):
    a, _ = validate(angles, axis)
    states=[]
    for theta in a:
        if axis == 'h_rz':
            states.append(rz(theta) @ H @ KET_ZERO)
        else:
            states.append({'ry':ry, 'rx':rx, 'rz':rz}[axis](theta) @ KET_ZERO)
    return np.kron(*states)


def state_vector(angles, axis='ry'):
    a, code = validate(angles, axis)
    state = cudaq.get_state(encoded, a, code)
    return np.array([state.amplitude(label) for label in LABELS])


def analytic_bloch(theta, axis):
    if axis == 'ry': return [np.sin(theta), 0., np.cos(theta)]
    if axis == 'rx': return [0., -np.sin(theta), np.cos(theta)]
    if axis == 'rz': return [0., 0., 1.]
    if axis == 'h_rz': return [np.cos(theta), np.sin(theta), 0.]
    raise ValueError('unknown axis')


def product_fidelity(first_angles, second_angles):
    """Overlap formula for two-qubit RY/RX or H-then-RZ product encodings."""
    a, _ = validate(first_angles, 'ry')
    b, _ = validate(second_angles, 'ry')
    return float(np.prod(np.cos((np.array(a)-b)/2)**2))
