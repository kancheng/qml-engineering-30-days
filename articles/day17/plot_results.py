"""Plot finite-difference error versus step using saved measurements."""
import argparse,json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/day17-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/day17-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gradients import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backend',choices=('qpp-cpu','nvidia'),default='qpp-cpu');args=p.parse_args()
    folder=ROOT/'results/day17'/args.backend;rows=json.loads((folder/'gradients.json').read_text())
    steps=sorted({r['step'] for r in rows});errors=[max(r['fd_max_error'] for r in rows if r['step']==s) for s in steps]
    fig,ax=plt.subplots(figsize=(6,4));ax.loglog(steps,errors,'o-',label='Central difference: worst case')
    ax.axhline(max(r['shift_max_error'] for r in rows),ls='--',color='orange',label='Parameter-shift: worst case')
    ax.set(xlabel='Finite-difference step',ylabel='Max absolute gradient error',title=f'Gradient checks — {args.backend}')
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(folder/'gradient_error.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
