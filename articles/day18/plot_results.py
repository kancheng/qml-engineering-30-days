"""Plot budget-aligned training curves and selected test metrics."""
import argparse,json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/day18-matplotlib');os.environ.setdefault('XDG_CACHE_HOME','/tmp/day18-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmark import ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=('qpp-cpu','nvidia'),default='qpp-cpu');a=p.parse_args();folder=ROOT/'results/day18'/a.backend
    candidates=json.loads((folder/'candidates.json').read_text());selected=json.loads((folder/'selected.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4));colors={'logistic':'tab:gray','mlp':'tab:orange','vqc':'tab:blue'}
    for r in candidates:
        axes[0].plot([h['evaluations'] for h in r['history']],[h['loss'] for h in r['history']],color=colors[r['model']],ls='-' if r['seed']==42 else '--',label=f"{r['model']} seed {r['seed']}")
    axes[0].set(xlabel='Training objective evaluations',ylabel='Brier loss',title='Same evaluation budget, different compute cost');axes[0].legend(fontsize=7);axes[0].grid(alpha=.2)
    names=[r['model'] for r in selected]+['constant'];values=[r['scores']['test']['metrics']['brier'] for r in selected]+[.25]
    axes[1].bar(names,values,color=['tab:gray','tab:orange','tab:blue','lightgray']);axes[1].set(ylabel='Test Brier',title='Selected by validation; one 12-point test split')
    fig.suptitle(f'XOR fixed-protocol comparison — {a.backend} VQC / CPU classical');fig.tight_layout();fig.savefig(folder/'benchmark.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
