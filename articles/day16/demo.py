"""Print one untrained QNN / MLP forward example."""
import argparse
from models import cudaq,initialize,mlp_initialize,mlp_forward,qnn_probability


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backend',choices=('qpp-cpu','nvidia'),default='qpp-cpu');p.add_argument('--features',type=float,nargs=2,default=[.25,-.4])
    args=p.parse_args();cudaq.set_target(args.backend)
    q=qnn_probability(args.features,initialize());m=float(mlp_forward([args.features],mlp_initialize())[0])
    print('Untrained seed 42; scaled features:',args.features)
    print('QNN (4 parameters), p1:',q)
    print('MLP (9 parameters; NumPy CPU), p1:',m)

if __name__=='__main__':main()
