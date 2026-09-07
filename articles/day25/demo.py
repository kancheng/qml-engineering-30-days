"""Infer from 13 raw Wine measurements using frozen Day25 pipelines."""
import argparse
import cudaq
from experiment import read
from wine_models import forward, representation

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', nargs=13, type=float)
    p.add_argument('--split-seed', type=int, choices=(2030, 2031), default=2030)
    p.add_argument('--backend', choices=('qpp-cpu', 'nvidia'), default='qpp-cpu')
    a = p.parse_args()
    cudaq.set_target(a.backend)
    prep = read('datasets')[str(a.split_seed)]['preprocessor']
    if a.features is None:
        # Default is a saved test sample; labels are not passed to inference.
        a.features = read('datasets')[str(a.split_seed)]['splits']['test'][0]['features']
        print('default: first saved test sample, raw features=', a.features)
    for model in read('selected'):
        if model['split_seed'] != a.split_seed:
            continue
        x, flags = representation(model['model'], [a.features], prep, model['weights'])
        prob = float(forward(model['model'], [a.features], prep, model['weights'], 'cudaq')[0])
        print(model['model'], 'representation=', x.tolist(), 'clipped=', int(flags.sum()),
              'p(cultivar 3)=', prob, 'class=', 'cultivar 3' if prob >= .5 else 'cultivar 2')
