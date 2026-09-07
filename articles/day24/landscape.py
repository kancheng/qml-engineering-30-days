"""Gradient concentration probe, with an independent tangent-state reference."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import cudaq
import numpy as np
from articles.day03.quantum_gates import ry, rz, Y


def validate(n,depth,weights):
    if type(n)is not int or not 2<=n<=8 or type(depth)is not int or not 1<=depth<=8:
        raise ValueError('integer qubits 2..8 and depth 1..8 required')
    w=np.asarray(weights,dtype=float)
    if w.shape!=(2*n*depth,) or not np.isfinite(w).all(): raise ValueError('invalid weights')
    return w


def initialize(n,depth,mode,seed):
    rng=np.random.default_rng(seed)
    if mode=='uniform': return rng.uniform(-np.pi,np.pi,2*n*depth)
    if mode=='small': return rng.normal(0,.1,2*n*depth)
    raise ValueError('unknown initialization')


@cudaq.kernel
def circuit(n:int,depth:int,w:list[float]):
    q=cudaq.qvector(n)
    for layer in range(depth):
        for j in range(n):
            ry(w[2*(layer*n+j)],q[j])
            rz(w[2*(layer*n+j)+1],q[j])
        for j in range(n-1):
            z.ctrl(q[j],q[j+1])


def gate(v,g,j,n):
    tensor=np.moveaxis(v.reshape([2]*n),j,0)
    return np.moveaxis((g@tensor.reshape(2,-1)).reshape(tensor.shape),0,j).reshape(-1)


def signs(n):
    ids=np.arange(2**n)
    local=1-2*((ids>>(n-1))&1)
    parity=np.array([1-2*(int(i).bit_count()%2) for i in ids])
    return local,parity


def reference(n,depth,weights):
    w=validate(n,depth,weights);v=np.zeros(2**n,dtype=complex);v[0]=1;dv=np.zeros_like(v)
    ids=np.arange(2**n)
    for layer in range(depth):
        for j in range(n):
            index=2*(layer*n+j);g=ry(w[index]);before=v
            v=gate(v,g,j,n);dv=gate(dv,g,j,n)
            if index==0: dv+=gate(before,(-.5j*Y)@g,j,n)
            v=gate(v,rz(w[index+1]),j,n);dv=gate(dv,rz(w[index+1]),j,n)
        for j in range(n-1):
            phase=1-2*(((ids>>(n-1-j))&1)*((ids>>(n-2-j))&1))
            v=v*phase;dv=dv*phase
    cost={};gradient={}
    for name,diagonal in zip(('local','global'),signs(n)):
        cost[name]=float(np.vdot(v,diagonal*v).real)
        gradient[name]=float(2*np.vdot(dv,diagonal*v).real)
    return cost,gradient


def expectation(n,depth,weights,cost):
    w=validate(n,depth,weights)
    if cost not in ('local','global'): raise ValueError('unknown cost')
    observable=cudaq.spin.z(0)
    if cost=='global':
        for j in range(1,n): observable=observable*cudaq.spin.z(j)
    return float(cudaq.observe(circuit,observable,n,depth,w.tolist(),shots_count=-1).expectation())


def shift(n,depth,weights,cost):
    w=validate(n,depth,weights);plus=w.copy();minus=w.copy();plus[0]+=np.pi/2;minus[0]-=np.pi/2
    a=expectation(n,depth,plus,cost);b=expectation(n,depth,minus,cost)
    return (a-b)/2, [a,b]


def product_control(n,seed):
    # Product RY states, C_local=<Z0>, C_global=<Z0...Zn-1>.
    a=np.random.default_rng(seed).uniform(-np.pi,np.pi,n)
    return {'local':float(-np.sin(a[0])), 'global':float(-np.sin(a[0])*np.prod(np.cos(a[1:])))}
