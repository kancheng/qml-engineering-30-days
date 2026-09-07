"""Infer from four raw Iris measurements using frozen Day21 pipelines."""
import argparse
import cudaq
from experiment import read
from reduction import forward, representation

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', nargs=4, type=float, default=[6., 2.9, 4.5, 1.5])
    p.add_argument('--split-seed', type=int, choices=(2028, 2029), default=2028)
    p.add_argument('--backend', choices=('qpp-cpu', 'nvidia'), default='qpp-cpu')
    a = p.parse_args()
    cudaq.set_target(a.backend)
    prep = read('datasets')[str(a.split_seed)]['preprocessor']
    for model in read('selected'):
        if model['split_seed'] != a.split_seed:
            continue
        x, flags = representation(model['model'], [a.features], prep, model['weights'])
        prob = float(forward(model['model'], [a.features], prep, model['weights'], 'cudaq')[0])
        print(model['model'], 'representation=', x.tolist(), 'clipped=', int(flags.sum()),
              'p(virginica)=', prob, 'class=', 'virginica' if prob >= .5 else 'versicolor')
