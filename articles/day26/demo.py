"""Compare exact density-matrix observables with a noisy counts sample."""
import argparse,cudaq
from noise import CHANNELS,exact,sample
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--channel',choices=tuple(CHANNELS),default='phase_flip')
    p.add_argument('--probability',type=float,default=.3);p.add_argument('--shots',type=int,default=4096);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();cudaq.set_target('density-matrix-cpu')
    print('Bell state; noise on q0 after preparation')
    for observable in ('ZZ','XX'):print(observable,exact(2,a.channel,a.probability,observable))
    for basis,label in [(0,'Z'),(1,'X')]:print(label,'counts',sample(2,a.channel,a.probability,basis,a.shots,a.seed))
