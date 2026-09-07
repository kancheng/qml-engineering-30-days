"""Generate kernel heatmaps and report from persisted artifacts."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/day23-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/day23-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import read, OUT, fingerprint
from kernels import np, MODELS


def main():
    selected,candidates=read('selected'),read('candidates')
    summaries={b:read(f'{b}/summary') for b in ('qpp-cpu','nvidia')}
    assert all(s['passed'] and s['artifact_sha256']==fingerprint() for s in summaries.values())
    assert read('training_summary')['artifact_sha256']==fingerprint()
    matrices=np.load(OUT/'matrices.npz')
    fig,axes=plt.subplots(1,3,figsize=(11,3.7))
    for ax,model in zip(axes,MODELS):
        im=ax.imshow(matrices[f'2028_{model}_train'],origin='lower',cmap='viridis')
        ax.set(title=model,xlabel='Train column index',ylabel='Train row index')
        ax.axhline(29.5,color='white',lw=.5);ax.axvline(29.5,color='white',lw=.5)
        fig.colorbar(im,ax=ax)
    fig.suptitle('Day23: split 2028 training Gram matrices / separate color scales')
    fig.tight_layout();fig.savefig(OUT/'gram_matrices.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    for model in MODELS:
        group=[s for s in selected if s['model']==model]
        ax.plot([s['split_seed'] for s in group],[s['scores']['test']['metrics']['brier'] for s in group],'o-',label=model)
    ax.set(xticks=[2028,2029],xlabel='Dataset seed',ylabel='Test Brier of clipped ridge score',title='Validation-selected regularization')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    lines=['# Day23 實驗結果','','由 `plot_results.py` 依保存的 JSON／NPZ 產生。Binary Iris99筆、兩個Day20 splits，train／validation／test=59／19／21。三種 kernel×三個 λ×兩splits，共18次 NumPy ridge solves。','','## 選定結果','','| Split | Kernel | λ | Test Brier | Accuracy | Confusion | Clipped scores train/val/test |','|---|---|---:|---:|---:|---|---|']
    for s in selected:
        m=s['scores']['test']['metrics'];clips=[s['scores'][p]['clipped_scores_count'] for p in ('train','validation','test')]
        lines.append(f"| {s['split_seed']} | {s['model']} | {s['lambda']} | {m['brier']:.6f} | {m['accuracy']:.2%} | {m['confusion_matrix']} | {clips} |")
    baseline=selected[0]['scores']['test']['constant_baseline']
    lines += ['',f"Train-prior baseline p=29/59：test Brier={baseline['brier']:.6f}、accuracy={baseline['accuracy']:.2%}（兩split相同）。Confusion row=true／column=predicted。Clipped ridge scores 不是經校準的機率。",'', '![Comparison](comparison.png)','','## Gram matrix','','![Gram matrices](gram_matrices.png)','','圖中前30筆為class0、後29筆為class1；排序只為顯示，未使用labels計算kernel。各圖色階獨立。','','| Split | Kernel | Symmetry error | Min eigenvalue | Diagonal error from 1 |','|---|---|---:|---:|---:|']
    for s in selected:
        d=s['gram_diagnostics'];lines.append(f"| {s['split_seed']} | {s['model']} | {d['symmetry_error']:.3e} | {d['minimum_eigenvalue_symmetrized']:.3e} | {d['diagonal_error_from_one']:.3e} |")
    lines += ['','Linear kernel 不要求 diagonal=1；接近零的負特徵值可能是浮點誤差。未做 PSD projection 或替換對角線。','','## 全部 validation 候選','','| Split | Kernel | λ | Validation Brier | NumPy solve seconds |','|---|---|---:|---:|---:|']
    for c in candidates:
        lines.append(f"| {c['split_seed']} | {c['model']} | {c['lambda']} | {c['validation']['brier']:.6f} | {c['solve_seconds_numpy']:.6f} |")
    lines += ['','## CPU／GPU 完整矩陣驗證','','| Backend | Observe calls | Max kernel error | Max clipped-score error | Passed |','|---|---:|---:|---:|---|']
    for b,s in summaries.items():
        lines.append(f"| {b} | {s['observe_calls']} | {s['max_kernel_error']:.3e} | {s['max_score_error']:.3e} | {s['passed']} |")
    lines += ['','每backend核對2×(59²+19×59+21×59)=11,682個矩陣元素，未利用對稱性省略量測；每元素一次projector observe，shots=-1、無噪聲。','GPU fp32 kernel 誤差會經alpha加權放大，故矩陣與下游分數分開檢查。alpha凍結於NumPy結果，沒有用GPU矩陣重訓或重新選λ。','GPU退出有cudaErrorCudartUnloading，數值驗證與程序exit code=0；根因未定位。時間含可能的JIT／cache，不宣稱GPU加速。','','## 限制','','固定小型 feature map、固定RBF gamma=1、三個λ，沒有窮盡模型調參。Linear使用常數feature，其係數也受regularization。','KRR解的是未clip squared-error正則化問題，validation才以clipped-score Brier選λ；不是SVM、不是直接最小化clipped Brier。','沿用已公開test與重疊splits，屬探索性比較；不宣稱量子優勢。本次量子模型未一致勝過classical kernels。','','## 原始紀錄','','- [Protocol](protocol.json)、[資料／scaler](datasets.json)、[Candidates](candidates.json)、[Selected](selected.json)','- [Training summary](training_summary.json)、[Reference matrices NPZ](matrices.npz)','- [CPU summary](qpp-cpu/summary.json)、[CPU verification](qpp-cpu/verification.json)、[CPU matrices](qpp-cpu/matrices.npz)','- [GPU summary](nvidia/summary.json)、[GPU verification](nvidia/verification.json)、[GPU matrices](nvidia/matrices.npz)','- [教學與重跑](../../articles/day23/README.md)','']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':main()
