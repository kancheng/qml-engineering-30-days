"""Plot the saved untrained model outputs, without ranking them."""
import argparse
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/day16-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/day16-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backend',choices=('qpp-cpu','nvidia'),default='qpp-cpu')
    args=p.parse_args();folder=ROOT/'results/day16'/args.backend
    rows=json.loads((folder/'predictions.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,seed in zip(axes,(42,43)):
        selected=[r for r in rows if r['seed']==seed];x=[r['features'][0] for r in selected]
        ax.plot(x,[r['qnn_p1'] for r in selected],label='PQC, 4 parameters')
        ax.plot(x,[r['mlp_p1'] for r in selected],label='MLP, 9 parameters')
        ax.set(xlabel='scaled x0 (x1=0.25)',title=f'Initialization seed {seed}',ylim=(-.02,1.02))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel('p1');fig.suptitle(f'Untrained forward slices — {args.backend} QNN; CPU NumPy MLP')
    fig.tight_layout();fig.savefig(folder/'forward_comparison.png',dpi=160);plt.close(fig)
    print(folder/'forward_comparison.png')

if __name__=='__main__':main()
