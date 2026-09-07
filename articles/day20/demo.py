"""Reload frozen Iris models and preprocess four raw centimeter measurements."""
import argparse,json
from iris_models import ROOT,preprocess,forward
import cudaq


def main():
    p=argparse.ArgumentParser();p.add_argument('--features',type=float,nargs=4,default=[6.,2.9,4.5,1.5]);p.add_argument('--split-seed',type=int,choices=(2028,2029),default=2028);p.add_argument('--backend',choices=('qpp-cpu','nvidia'),default='qpp-cpu');a=p.parse_args()
    folder=ROOT/'results/day20';datasets=json.loads((folder/'datasets.json').read_text());selected=json.loads((folder/'selected.json').read_text());cudaq.set_target(a.backend)
    x,flags=preprocess([a.features],datasets[str(a.split_seed)]['preprocessor']);print('scaled PCs:',x.tolist(),'clipped:',flags.tolist())
    for r in selected:
        if r['split_seed']==a.split_seed:
            value=float(forward(r['model'],x,r['weights'],'cudaq')[0]);print(r['model'],'p(virginica):',value,'class:', 'virginica' if value>=.5 else 'versicolor')

if __name__=='__main__':main()
