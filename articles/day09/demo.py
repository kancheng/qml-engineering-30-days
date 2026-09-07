"""Inspect one PQC input and its independently supplied weights."""

import argparse
import json

import cudaq

from pqc import initialize, inputs, model, predict, reference_prediction, sample_prediction


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('qpp-cpu','nvidia'),default='qpp-cpu')
    parser.add_argument('--features',nargs=2,type=float,default=[0.25,-0.4])
    parser.add_argument('--layers',type=int,default=1)
    parser.add_argument('--weights',nargs='+',type=float)
    args=parser.parse_args()
    weights=initialize(args.layers) if args.weights is None else args.weights
    angles,w=inputs(args.features,weights,args.layers)
    cudaq.set_target(args.backend)
    print(cudaq.draw(model,angles,w,args.layers))
    exact=predict(args.features,w,args.layers)
    expected=reference_prediction(args.features,w,args.layers)
    print(json.dumps({'backend':cudaq.get_target().name,'features':args.features,'encoded_angles':angles,
                      'layers':args.layers,'weights':w,'parameter_count':len(w),'exact_zz':exact,
                      'numpy_reference':expected,'sampled':sample_prediction(args.features,w,args.layers)},indent=2))
    raise SystemExit(0 if abs(exact-expected) <= 1e-5 else 1)


if __name__ == '__main__':
    main()
