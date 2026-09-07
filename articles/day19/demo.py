"""Reload saved hybrid parameters and print all intermediate outputs."""
import argparse,json
from hybrid import ROOT,cudaq,forward


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=('qpp-cpu','nvidia'),default='qpp-cpu');p.add_argument('--features',type=float,nargs=2,default=[-.6,.4]);a=p.parse_args()
    checkpoint=json.loads((ROOT/'results/day19'/a.backend/'checkpoint.json').read_text())
    if checkpoint['schema_version']!=1:raise ValueError('unsupported checkpoint schema')
    cudaq.set_target(a.backend);p1,cache=forward([a.features],checkpoint['parameters'])
    print('features:',a.features)
    for key,value in cache.items():print(key,value.tolist())
    print('p1:',p1.tolist())

if __name__=='__main__':main()
