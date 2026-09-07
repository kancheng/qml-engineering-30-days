"""Explicit gate-location noise via a zero-angle RZ marker."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import cudaq
import numpy as np
from articles.day03.quantum_gates import I,X,Y,Z,H
from articles.day04.bell_state import CNOT
CHANNELS={'bit_flip':cudaq.BitFlipChannel,'phase_flip':cudaq.PhaseFlipChannel,'depolarizing':cudaq.DepolarizationChannel}
OBS={'Z0':np.kron(Z,I),'X0':np.kron(X,I),'ZZ':np.kron(Z,Z),'XX':np.kron(X,X)}


def noise_model(channel,p):
    if channel not in CHANNELS or not np.isfinite(p) or not 0<=p<=1:raise ValueError('known channel and probability in [0,1] required')
    model=cudaq.NoiseModel();model.add_channel('rz',[0],CHANNELS[channel](p));return model


@cudaq.kernel
def prepared(kind:int):
    q=cudaq.qvector(2)
    if kind>0:h(q[0])
    if kind==2:x.ctrl(q[0],q[1])
    rz(0.0,q[0])


@cudaq.kernel
def measured(kind:int,basis:int):
    q=cudaq.qvector(2)
    if kind>0:h(q[0])
    if kind==2:x.ctrl(q[0],q[1])
    rz(0.0,q[0])
    if basis==1:
        h(q[0]);h(q[1])
    mz(q[0]);mz(q[1])


def kraus(channel,p):
    noise_model(channel,p)
    if channel=='bit_flip':return [np.sqrt(1-p)*I,np.sqrt(p)*X]
    if channel=='phase_flip':return [np.sqrt(1-p)*I,np.sqrt(p)*Z]
    return [np.sqrt(1-p)*I]+[np.sqrt(p/3)*a for a in (X,Y,Z)]


def density(kind,channel,p):
    if kind not in (0,1,2):raise ValueError('state kind must be 0,1,2')
    v=np.array([1,0,0,0],dtype=complex)
    if kind>0:v=np.kron(H,I)@v
    if kind==2:v=CNOT@v
    rho=np.outer(v,v.conj());result=np.zeros((4,4),dtype=complex)
    for k in kraus(channel,p):
        a=np.kron(k,I);result+=a@rho@a.conj().T
    return result


def observable(name):
    return {'Z0':cudaq.spin.z(0),'X0':cudaq.spin.x(0),'ZZ':cudaq.spin.z(0)*cudaq.spin.z(1),
            'XX':cudaq.spin.x(0)*cudaq.spin.x(1)}[name]


def exact(kind,channel,p,name):
    return float(cudaq.observe(prepared,observable(name),kind,noise_model=noise_model(channel,p),shots_count=-1).expectation())


def probabilities(kind,channel,p,basis):
    rho=density(kind,channel,p)
    if basis==1:
        op=np.kron(H,H);rho=op@rho@op.conj().T
    return np.diag(rho).real


def sample(kind,channel,p,basis,shots,seed):
    if type(shots)is not int or shots<=0:raise ValueError('positive integer shots required')
    cudaq.set_random_seed(seed)
    return {str(k):int(v) for k,v in cudaq.sample(measured,kind,basis,noise_model=noise_model(channel,p),shots_count=shots,explicit_measurements=True).items()}
