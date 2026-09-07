import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/day26-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/day26-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import OUT,read

def main():
    exact,wine=read('exact'),read('wine');summary=read('summary')
    sampling={b:read(f'{b}/summary') for b in ('density-matrix-cpu','nvidia')}
    assert summary['passed'] and all(s['passed'] for s in sampling.values())
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for channel in ('bit_flip','phase_flip','depolarizing'):
        for name,style in [('ZZ','o-'),('XX','s--')]:
            rows=[r for r in exact if r['kind']==2 and r['channel']==channel and r['observable']==name]
            axes[0].plot([r['p'] for r in rows],[r['actual'] for r in rows],style,label=f'{channel}/{name}')
        for seed,style in [(2030,'o-'),(2031,'s--')]:
            rows=[r for r in wine if r['channel']==channel and r['split_seed']==seed]
            axes[1].plot([r['p'] for r in rows],[r['metrics']['brier'] for r in rows],style,label=f'{channel}/{seed}')
    axes[0].set(title='Bell state / noise on q0',xlabel='Channel p',ylabel='Exact expectation')
    axes[1].set(title='Frozen Wine VQC / terminal noise',xlabel='Channel p',ylabel='Test Brier')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(OUT/'noise_effects.png',dpi=160);plt.close(fig)
    lines=['# Day26 Noise 實測報告','','![Noise effects](noise_effects.png)','','## 精確驗證','',f"Density-matrix CPU：{summary['exact_records']}筆基礎observable records＋{summary['wine_records']}筆Wine records，共{summary['exact_observe_calls']}次observe。",f"基礎最大誤差={summary['max_exact_error']:.3e}，Wine解析對照最大誤差={summary['max_wine_error']:.3e}。",'','## 有限shots','','| Backend | Records | Shots each | Total shots | Max bin-frequency error | Passed |','|---|---:|---:|---:|---:|---|']
    for b,s in sampling.items():lines.append(f"| {b} | {s['records']} | {s['shots_per_record']} | {s['total_shots']} | {s['max_frequency_error']:.6f} | {s['passed']} |")
    lines+=['','每record對照四種bitstrings；固定absolute tolerance .06是工程smoke threshold，不是confidence interval。GPU是noisy finite-shot sampling，不是exact density matrix。CPU與GPU不要求同seed產生相同counts。','','## 凍結Wine VQC','','| Split | Channel | p | Test Brier | Accuracy |','|---|---|---:|---:|---:|']
    for r in wine:lines.append(f"| {r['split_seed']} | {r['channel']} | {r['p']} | {r['metrics']['brier']:.6f} | {r['metrics']['accuracy']:.2%} |")
    lines+=['','Noise只在末端q0注入，並非每個gate皆有error。Phase flip與ZZ對易，這裡不改變預測；不表示完整VQC對phase noise免疫。','本日bit flip p≤.3與depolarizing p≤.3使ZZ乘正縮放，因此0.5threshold的分類通常保持，Brier仍改變。這是特定插入位置的解析效果，不是noise-aware training。','Classical參考為NumPy Kraus density matrix與Day25保存的train-prior constant baseline；baseline不經量子channel。','','## 紀錄','','- [Protocol](protocol.json)、[Exact records](exact.json)、[Summary](summary.json)','- [Wine scores](wine.json)、[Wine source hashes](wine_source.json)','- [CPU counts](density-matrix-cpu/samples.json)、[CPU summary](density-matrix-cpu/summary.json)','- [GPU counts](nvidia/samples.json)、[GPU summary](nvidia/summary.json)','- [教學與重跑](../../articles/day26/README.md)','']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__':main()
