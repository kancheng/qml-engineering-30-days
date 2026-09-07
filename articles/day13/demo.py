"""Prepare a small signed real vector with explicit gates."""
import argparse
from amplitude_encoding import cudaq,np,normalize,gate_angles,gates,extract


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--values',type=float,nargs='+',default=[3,4,0])
    p.add_argument('--backend',choices=('qpp-cpu','nvidia'),default='qpp-cpu')
    args=p.parse_args();a,meta=normalize(args.values);angles=gate_angles(a)
    cudaq.set_target(args.backend);n=meta['qubits']
    actual=extract(cudaq.get_state(gates,angles,n),n)
    print('input:',args.values)
    print('normalized amplitudes:',a.tolist())
    print('qubits:',n,'angles:',angles)
    print(cudaq.draw(gates,angles,n))
    print('probabilities:',(abs(actual)**2).tolist())
    print('fidelity:',float(abs(np.vdot(a,actual))**2))

if __name__=='__main__':main()
