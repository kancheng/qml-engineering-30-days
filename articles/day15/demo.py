"""Reload a saved classifier and predict raw two-dimensional inputs."""
import argparse
from pathlib import Path
from classifier import ROOT,load_checkpoint,predict_proba
from articles.day14.model import cudaq


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'results/day15/qpp-cpu/checkpoint.json')
    p.add_argument('--features',type=float,nargs=2,default=[-.7,.7])
    p.add_argument('--backend',choices=('qpp-cpu','nvidia'),default='qpp-cpu')
    args=p.parse_args();model=load_checkpoint(args.checkpoint);cudaq.set_target(args.backend)
    prob,flags=predict_proba([args.features],model['weights'],model['scaler'])
    print('raw:',args.features,'p(class 1):',float(prob[0]),'class:',int(prob[0]>=model['threshold']),'clipped:',flags[0].tolist())

if __name__=='__main__':main()
