"""Inspect the composed feature map + Ansatz model."""
import argparse
from model import cudaq,Config,initialize,arguments,circuit,predict,reference_prediction


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--encoding',choices=('angle','amplitude'),default='angle')
    p.add_argument('--features',type=float,nargs='+')
    p.add_argument('--layers',type=int,default=1)
    p.add_argument('--backend',choices=('qpp-cpu','nvidia'),default='qpp-cpu')
    args=p.parse_args();config=Config(args.encoding,args.layers)
    features=args.features if args.features is not None else ([.25,-.4] if args.encoding=='angle' else [1,-2,3,-4])
    weights=initialize(config.layers,42)
    kernel_args=arguments(features,weights,config)
    cudaq.set_target(args.backend)
    print(config,'features:',features,'weights:',weights.tolist())
    print(cudaq.draw(circuit,*kernel_args))
    print('exact ZZ:',predict(features,weights,config))
    print('NumPy reference:',reference_prediction(features,weights,config))

if __name__=='__main__':main()
