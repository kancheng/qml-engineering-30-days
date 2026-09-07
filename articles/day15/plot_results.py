"""Export training curves and NumPy-reference decision boundary from a checkpoint."""
import argparse
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/day15-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/day15-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from classifier import np,ROOT,load_checkpoint,predict_proba


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backend',choices=('qpp-cpu','nvidia'),default='qpp-cpu')
    args=p.parse_args();folder=ROOT/'results/day15'/args.backend
    checkpoint=load_checkpoint(folder/'checkpoint.json');candidates=json.loads((folder/'candidates.json').read_text());data=json.loads((folder/'dataset.json').read_text())
    fig,ax=plt.subplots(figsize=(7,4))
    for c in candidates:
        h=c['history'];line=ax.plot([r['sweep'] for r in h],[r['loss'] for r in h],label=f"seed {c['seed']} train")[0]
        ax.plot([r['sweep'] for r in h],[r['validation_brier_reference'] for r in h],ls='--',color=line.get_color(),label=f"seed {c['seed']} validation (NumPy)")
    ax.set(xlabel='Completed sweeps',ylabel='Brier loss',title=f'XOR training — {args.backend}')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(folder/'training_curve.png',dpi=160);plt.close(fig)
    grid=np.linspace(-1,1,81);gx,gy=np.meshgrid(grid,grid)
    raw=np.c_[gx.ravel(),gy.ravel()];prob,_=predict_proba(raw,checkpoint['weights'],checkpoint['scaler'],reference=True);z=prob.reshape(gx.shape)
    np.savez_compressed(folder/'boundary_grid.npz',x=grid,y=grid,p1=z)
    fig,ax=plt.subplots(figsize=(6,5))
    shade=ax.contourf(gx,gy,z,levels=np.linspace(0,1,21),cmap='RdBu_r',vmin=0,vmax=1)
    ax.contour(gx,gy,z,levels=[.5],colors='black',linewidths=1.2)
    for split,marker in [('train','o'),('validation','s'),('test','^')]:
        x=np.array(data[split]['raw']);y=np.array(data[split]['labels'])
        ax.scatter(x[:,0],x[:,1],c=y,cmap='RdBu_r',vmin=0,vmax=1,marker=marker,edgecolors='black',s=45,label=split)
    ax.set(xlabel='raw x0',ylabel='raw x1',title=f"XOR boundary — {args.backend}, seed {checkpoint['seed']}\nNumPy evaluation of saved PQC; p1=0.5 contour",xlim=(-1,1),ylim=(-1,1))
    ax.legend(fontsize=8);fig.colorbar(shade,ax=ax,label='p(class 1)');fig.tight_layout();fig.savefig(folder/'decision_boundary.png',dpi=160);plt.close(fig)
    print('Saved plots and boundary grid:',folder)

if __name__=='__main__':main()
