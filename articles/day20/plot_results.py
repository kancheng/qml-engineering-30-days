"""Show two-split stability without treating initialization as new test samples."""
import json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/day20-matplotlib');os.environ.setdefault('XDG_CACHE_HOME','/tmp/day20-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iris_models import ROOT


def main():
    folder=ROOT/'results/day20';selected=json.loads((folder/'selected.json').read_text());candidates=json.loads((folder/'candidates.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4));colors={'mlp':'tab:orange','vqc':'tab:blue','hybrid':'tab:green'}
    for model in colors:
        rows=[r for r in selected if r['model']==model]
        axes[0].plot([r['split_seed'] for r in rows],[r['scores']['test']['metrics']['brier'] for r in rows],'o-',label=model,color=colors[model])
        for seed in (42,43):
            row=next(r for r in candidates if r['model']==model and r['split_seed']==2028 and r['seed']==seed)
            axes[1].plot([h['evaluations'] for h in row['history']],[h['loss'] for h in row['history']],color=colors[model],ls='-' if seed==42 else '--',label=f'{model} seed {seed}')
    axes[0].set(xlabel='Dataset split seed',ylabel='Test Brier',title='Two splits, validation-selected initialization',xticks=[2028,2029]);axes[0].legend()
    axes[1].set(xlabel='Training objective evaluations',ylabel='Train Brier',title='Split 2028: shared NumPy training engine');axes[1].legend(fontsize=7)
    fig.suptitle('Iris versicolor vs virginica — four raw features, train-only PCA2')
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(folder/'comparison.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
