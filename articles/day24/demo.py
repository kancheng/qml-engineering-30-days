"""Inspect one gradient with tangent-state, parameter-shift and finite difference."""
import argparse
import cudaq
from landscape import np,initialize,reference,shift

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--qubits',type=int,choices=(2,4,6,8),default=4)
    p.add_argument('--depth',type=int,choices=(1,4,8),default=4);p.add_argument('--initialization',choices=('uniform','small'),default='uniform')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--backend',choices=('qpp-cpu','nvidia'),default='qpp-cpu')
    a=p.parse_args();cudaq.set_target(a.backend);w=initialize(a.qubits,a.depth,a.initialization,a.seed)
    costs,g=reference(a.qubits,a.depth,w)
    plus=w.copy();minus=w.copy();plus[0]+=1e-5;minus[0]-=1e-5
    for cost in ('local','global'):
        actual,_=shift(a.qubits,a.depth,w,cost)
        fd=(reference(a.qubits,a.depth,plus)[0][cost]-reference(a.qubits,a.depth,minus)[0][cost])/2e-5
        print(cost,'cost=',costs[cost],'tangent=',g[cost],'parameter-shift=',actual,'finite-difference=',fd)
