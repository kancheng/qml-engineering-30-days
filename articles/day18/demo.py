"""Reload the selected models and apply their shared saved scaler."""
import argparse,json
from benchmark import ROOT,cudaq,forward,transform


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backend',choices=('qpp-cpu','nvidia'),default='qpp-cpu');p.add_argument('--features',type=float,nargs=2,default=[-.7,.7]);a=p.parse_args()
    folder=ROOT/'results/day18'/a.backend;models=json.loads((folder/'selected.json').read_text());scaler=json.loads((folder/'scaler.json').read_text())
    x,flags=transform([a.features],scaler);cudaq.set_target(a.backend)
    print('raw:',a.features,'clipped:',flags[0].tolist())
    for model in models:
        value=float(forward(model['model'],x,model['weights'])[0]);print(model['model'],'p1:',value,'class:',int(value>=.5))

if __name__=='__main__':main()
