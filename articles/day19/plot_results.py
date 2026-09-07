"""Plot loss and gradient norms for the three hybrid parameter groups."""
import argparse,json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/day19-matplotlib');os.environ.setdefault('XDG_CACHE_HOME','/tmp/day19-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hybrid import ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=('qpp-cpu','nvidia'),default='qpp-cpu');a=p.parse_args();folder=ROOT/'results/day19'/a.backend
    h=json.loads((folder/'history.json').read_text());iterations=[r['iteration'] for r in h]
    fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].plot(iterations,[r['loss'] for r in h],'o-');axes[0].set(xlabel='Completed updates',ylabel='Brier loss',title='Three-point integration task')
    for group in ('encoder','quantum','head'):axes[1].plot(iterations,[r['group_gradient_norms'][group] for r in h],'o-',label=group)
    axes[1].set(xlabel='Completed updates',ylabel='Gradient L2 norm',title='All three groups receive gradients');axes[1].legend()
    fig.suptitle(f'Classical → quantum → classical — {a.backend} quantum simulator')
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(folder/'hybrid_training.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
