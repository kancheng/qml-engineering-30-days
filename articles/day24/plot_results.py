"""Plot empirical gradient variance without fitting an asymptotic scaling law."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/day24-matplotlib');os.environ.setdefault('XDG_CACHE_HOME','/tmp/day24-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import OUT,read,fingerprint,stats


def main():
    groups=read('statistics');controls=read('product_control');summaries={b:read(f'{b}/summary') for b in ('qpp-cpu','nvidia')}
    assert read('summary')['artifact_sha256']==fingerprint()
    assert all(s['passed'] and s['artifact_sha256']==fingerprint() for s in summaries.values())
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),sharey=True)
    for ax,mode in zip(axes,('uniform','small')):
        for depth,color in zip((1,4,8),('tab:blue','tab:orange','tab:green')):
            group=[g for g in groups if g['depth']==depth and g['initialization']==mode]
            for cost,style in [('local','o-'),('global','s--')]:
                ax.semilogy([g['qubits'] for g in group],[g['statistics'][cost]['sample_variance_ddof1'] for g in group],style,color=color,label=f'{cost}, blocks={depth}')
        ax.set(title=mode,xlabel='Qubits',ylabel='Sample variance of dC/dtheta[0]',xticks=[2,4,6,8])
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Day24: 32 initializations per setting / exact reference')
    fig.tight_layout();fig.savefig(OUT/'variance.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    for cost,marker in [('local','o'),('global','s')]:
        ax.semilogy([c['qubits'] for c in controls],[c['exact_variance'][cost] for c in controls],'-',label=f'{cost} exact ensemble')
        ax.semilogy([c['qubits'] for c in controls],[stats([s[cost] for s in c['samples']])['sample_variance_ddof1'] for c in controls],marker+'--',label=f'{cost} sample (32 seeds)')
    ax.set(xlabel='Qubits',ylabel='Gradient variance',xticks=[2,4,6,8],title='Analytic product-state control');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'product_control.png',dpi=160);plt.close(fig)
    lines=['# Day24 實驗結果','','本檔由 `plot_results.py` 產生；768 個初始化電路、1,536 個單參數梯度。沒有資料集分類或optimizer訓練。','','## 梯度統計','','![Variance](variance.png)','','每組32個seeds，以ddof=1估計variance。Local=Z0，global=全域Z parity；都用第一個RY參數，不混合不同參數的梯度。','','| Qubits | Blocks | Init | Cost | Mean | Variance | RMS | Median abs | Fraction abs<0.01 |','|---:|---:|---|---|---:|---:|---:|---:|---:|']
    for g in groups:
        for cost,s in g['statistics'].items():
            lines.append(f"| {g['qubits']} | {g['depth']} | {g['initialization']} | {cost} | {s['mean']:.3e} | {s['sample_variance_ddof1']:.3e} | {s['rms']:.3e} | {s['median_abs']:.3e} | {s['fraction_abs_lt_001']:.2%} |")
    lines += ['','## 可解析 classical 對照','','![Product control](product_control.png)','','Product RY states、uniform angles的精確ensemble variance為local=1/2、global=2^(-n)。樣本只有32筆，sample variance不會恰好等於理論值。對照另保存自己的angles sampling seeds；與主實驗的交錯RY／RZ隨機向量不逐筆相同。','','## CUDA-Q 驗證','','| Backend | Gradient records | Observe calls | Max absolute gradient error | Passed |','|---|---:|---:|---:|---|']
    for b,s in summaries.items():
        lines.append(f"| {b} | {s['records']} | {s['observe_calls']} | {s['max_gradient_error']:.3e} | {s['passed']} |")
    lines += ['','CUDA-Q逐個樣本以±π/2計算parameter-shift；NumPy使用tangent-state導數。每gradient兩次observe，共3,072次／backend。沒有finite shots或noise。','GPU退出有cudaErrorCudartUnloading，程序exit code=0且檢查通過；根因未定位。CUDA-Q時間含可能的compilation／cache，未做隔離效能比較。','','## 解讀邊界','','32 seeds與n≤8不足以由曲線證明漸近exponential scaling；不擬合直線後宣稱證明Barren Plateau。','單層CZ與RZ皆對Z讀出對易，所以depth1的local／global成本與product-state解析式相同；global variance下降在此不需要深電路或複雜entanglement解釋。','小角度初始化的variance小也可能因靠近stationary point，並非較差或較好的訓練能力證明。全零權重兩個cost皆1、梯度為0，是明確測試的例子。','Global與local是不同objective，換成local不保證仍解同一個任務。只測index0，不代表全gradient norm或所有參數的可訓練性。','','## 原始紀錄','','- [Protocol](protocol.json)、[全部weights／costs／gradients](samples.json)、[Statistics](statistics.json)','- [Product control](product_control.json)、[Reference summary](summary.json)','- [CPU summary](qpp-cpu/summary.json)、[CPU shifted expectations](qpp-cpu/verification.json)','- [GPU summary](nvidia/summary.json)、[GPU shifted expectations](nvidia/verification.json)','- [教學與重跑](../../articles/day24/README.md)','']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':main()
