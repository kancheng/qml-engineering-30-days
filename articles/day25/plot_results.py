"""Render the second dataset benchmark directly from saved artifacts."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/day25-matplotlib');os.environ.setdefault('XDG_CACHE_HOME','/tmp/day25-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import OUT,read,fingerprint
from wine_models import COUNTS


def main():
    selected,candidates,data=read('selected'),read('candidates'),read('datasets')
    summaries={b:read(f'{b}/summary') for b in ('qpp-cpu','nvidia')}
    assert read('training_summary')['artifact_sha256']==fingerprint()
    assert all(s['passed'] and s['artifact_sha256']==fingerprint() for s in summaries.values())
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for model in COUNTS:
        group=[r for r in selected if r['model']==model]
        axes[0].plot([r['split_seed'] for r in group],[r['scores']['test']['metrics']['brier'] for r in group],'o-',label=model)
        for c in candidates:
            if c['model']==model and c['split_seed']==2030:
                axes[1].plot([r['evaluations'] for r in c['history']],[r['loss'] for r in c['history']],label=f"{model}/{c['seed']}")
    axes[0].set(xticks=[2030,2031],xlabel='Dataset seed',ylabel='Test Brier',title='Validation-selected models')
    axes[1].set(xlabel='Objective evaluations',ylabel='Train Brier',title='Split 2030 / both initializations')
    for ax in axes:ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle('Day25: Wine cultivar 2/3 / NumPy reference training')
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    lines=['# Day25 Wine Benchmark Report','','由 `plot_results.py` 依保存的JSON產生。第二個真實資料集使用UCI Wine的cultivar2／3；119筆、13個原始特徵，非完整三分類，也非Wine Quality。','','## Protocol與資料','','- Dataset seeds2030／2031，各train70／validation23／test26；初始化seeds42／43。','- 16次NumPy CPU reference fits，每次73個Brier objective、5,110筆train輸入評估；CUDA-Q training calls=0。','- MLP／VQC／Hybrid同用train-only PCA2；logistic13使用全部標準化13維。','- 每模型每split只依final validation Brier選seed，test不參與當次選擇。','','## 選定模型','','| Split | Model | Seed | Test Brier | Accuracy | Confusion matrix | Clipped train/val/test |','|---|---|---:|---:|---:|---|---|']
    for r in selected:
        m=r['scores']['test']['metrics'];clips=[r['scores'][p]['clipped_values'] for p in ('train','validation','test')]
        lines.append(f"| {r['split_seed']} | {r['model']} | {r['seed']} | {m['brier']:.6f} | {m['accuracy']:.2%} | {m['confusion_matrix']} | {clips} |")
    b=selected[0]['scores']['test']['constant_baseline']
    lines += ['',f"Train-prior baseline p1=28/70=0.4：test Brier={b['brier']:.6f}、accuracy={b['accuracy']:.2%}，兩split相同。Confusion row=true／column=predicted；clipping按座標計數。",'','![Comparison](comparison.png)','','## PCA與資料完整性','','| Split | PC1 variance | PC2 variance | Sum |','|---|---:|---:|---:|']
    for seed,d in data.items():
        a,b=d['preprocessor']['explained_variance_ratio'];lines.append(f'| {seed} | {a:.6f} | {b:.6f} | {a+b:.6f} |')
    lines += ['',f"原始檔SHA-256：`{read('protocol')['source']['sha256']}`。沒有完全重複feature＋label rows。Variance ratio針對標準化train，不等於保留的分類資訊比例。",'','## 全部候選與成本','','| Split | Model | Seed | Train Brier | Validation Brier | NumPy fit seconds | Full sweeps |','|---|---|---:|---:|---:|---:|---:|']
    for c in candidates:
        lines.append(f"| {c['split_seed']} | {c['model']} | {c['seed']} | {c['history'][-1]['loss']:.6f} | {c['validation']['brier']:.6f} | {c['fit_seconds_numpy']:.3f} | {c['full_sweeps']} |")
    lines += ['','## 凍結模型驗證','','| Backend | Records | Observe calls | Max probability error | Passed |','|---|---:|---:|---:|---|']
    for b,s in summaries.items():
        lines.append(f"| {b} | {s['records']} | {s['observe_calls']} | {s['max_reference_error']:.3e} | {s['passed']} |")
    lines += ['','每backend：2splits×119筆×2量子模型=476次exact observe；classical模型保持NumPy CPU，合計24筆partition records。','GPU退出有cudaErrorCudartUnloading，驗證與測試exit code=0；根因未定位。時間可能含compilation／cache，非GPU training或隔離speedup測量。','','## 解讀與限制','','MLP與VQC／Hybrid有相同二維輸入；logistic13可見更多資訊，所以是實用完整輸入對照，非隔離量子層效果的配對。','73次objective不等於相同參數更新次數或充分收斂；沒有early stopping或依test追加預算。','兩個test splits重疊，每組僅26筆；不給顯著性或量子優勢結論，也不把Iris與Wine不同任務的accuracy直接排行。','本日沒有重跑Day23 quantum kernel或Day24梯度分布；第二份benchmark聚焦原定Classical／VQC／Hybrid交付。','','## 原始紀錄','','- [Protocol](protocol.json)、[資料／preprocessor](datasets.json)、[Candidates](candidates.json)、[Selected](selected.json)','- [Training summary](training_summary.json)','- [CPU summary](qpp-cpu/summary.json)、[CPU predictions](qpp-cpu/verification.json)','- [GPU summary](nvidia/summary.json)、[GPU predictions](nvidia/verification.json)','- [資料來源與授權](../../data/day25/README.md)、[教學與重跑](../../articles/day25/README.md)','']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':main()
