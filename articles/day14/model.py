"""Composable two-qubit model: feature preparation, shared Ansatz, ZZ readout."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from dataclasses import dataclass
import cudaq
import numpy as np
from articles.day09.pqc import feature_map, ansatz, parameter_count, initialize
from articles.day12.angle_encoding import to_angles
from articles.day13.amplitude_encoding import normalize, gate_angles, prepare as amplitude_prepare
from articles.day03.quantum_gates import KET_ZERO, Z, ry
from articles.day04.bell_state import CNOT


@dataclass(frozen=True)
class Config:
    encoding: str = 'angle'
    layers: int = 1

    def __post_init__(self):
        if self.encoding not in ('angle','amplitude'):raise ValueError('unknown encoding')
        parameter_count(self.layers)


def encode(features, config):
    if config.encoding=='angle':
        return to_angles(features,'positive_half'),0
    a,meta=normalize(features)
    if meta['qubits']!=2:raise ValueError('this two-qubit amplitude model requires 3 or 4 features')
    return gate_angles(a),1


def arguments(features, weights, config):
    data,code=encode(features,config)
    w=np.asarray(weights,dtype=float)
    if w.shape!=(parameter_count(config.layers),) or not np.isfinite(w).all():
        raise ValueError('weights must be a finite vector of length 4*layers')
    return data,w.tolist(),code,config.layers


@cudaq.kernel
def preparation(q: cudaq.qview, data: list[float], weights: list[float], encoding: int, layers: int):
    if encoding==0:
        feature_map(q,data)
    else:
        amplitude_prepare(q,data,2)
    ansatz(q,weights,layers)


@cudaq.kernel
def circuit(data: list[float], weights: list[float], encoding: int, layers: int):
    q=cudaq.qvector(2)
    preparation(q,data,weights,encoding,layers)


@cudaq.kernel
def measured(data: list[float], weights: list[float], encoding: int, layers: int):
    q=cudaq.qvector(2)
    preparation(q,data,weights,encoding,layers)
    mz(q[0])
    mz(q[1])


def predict(features, weights, config=Config()):
    return float(cudaq.observe(circuit,cudaq.spin.z(0)*cudaq.spin.z(1),
                              *arguments(features,weights,config),shots_count=-1).expectation())


def predict_batch(features, weights, config=Config()):
    rows=list(features)
    if not rows:raise ValueError('batch cannot be empty')
    return np.array([predict(row,weights,config) for row in rows])


def reference_state(features, weights, config=Config()):
    data,w,_,layers=arguments(features,weights,config)
    if config.encoding=='angle':
        state=np.kron(ry(data[0])@KET_ZERO,ry(data[1])@KET_ZERO)
    else:state=normalize(features)[0].astype(complex)
    for i in range(layers):
        b=w[4*i:4*i+4]
        state=np.kron(ry(b[0]),ry(b[1]))@state
        state=CNOT@state
        state=np.kron(ry(b[2]),ry(b[3]))@state
    return state


def reference_prediction(features,weights,config=Config()):
    state=reference_state(features,weights,config)
    return float(np.vdot(state,np.kron(Z,Z)@state).real)
