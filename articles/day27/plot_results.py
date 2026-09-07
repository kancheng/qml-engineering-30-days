"""Compare matching outputs before reporting CPU/GPU timing ratios."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/day27-matplotlib');os.environ.setdefault('XDG_CACHE_HOME','/tmp/day27-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmark import OUT,PROTOCOL,read,write,np


def main():
    data={b:read(f'{b}/records') for b in PROTOCOL['targets']};summaries={b:read(f'{b}/summary') for b in data}
    assert all(s['passed'] and s['protocol']==PROTOCOL for s in summaries.values())
    assert 'fp64' in summaries['qpp-cpu']['precision'] and 'fp64' in summaries['nvidia-fp64']['precision']
    comparison=[]
    for cpu in data['qpp-cpu']:
        for backend in ('nvidia-fp64','nvidia'):
            gpu=next(r for r in data[backend] if (r['qubits'],r['blocks'])==(cpu['qubits'],cpu['blocks']))
            assert gpu['weights_sha256']==cpu['weights_sha256']
            error=max(abs(np.array(gpu['values'])-cpu['values']))
            assert error<1e-5
            comparison.append({'qubits':cpu['qubits'],'blocks':cpu['blocks'],'gpu_backend':backend,'max_output_error':float(error),
                               'cpu_median_over_gpu_median':cpu['median_seconds']/gpu['median_seconds']})
    write('comparison',comparison)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for ax,blocks in zip(axes,(4,12)):
        for backend,rows in data.items():
            group=[r for r in rows if r['blocks']==blocks]
            x=[r['qubits'] for r in group];med=np.array([r['median_seconds'] for r in group])
            ax.plot(x,med,'o-',label=backend)
            ax.fill_between(x,[r['q25_seconds'] for r in group],[r['q75_seconds'] for r in group],alpha=.15)
        ax.set(yscale='log',xticks=PROTOCOL['qubits'],xlabel='Qubits',ylabel='Warm observe seconds',title=f'{blocks} blocks / median and IQR')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'latency.png',dpi=160);plt.close(fig)
    lines=['# Day27 CPU／GPU 實測報告','','![Latency](latency.png)','','同一個exact global-Z observe、同一weights，backend依序執行，不同時跑CPU與GPU。OMP_NUM_THREADS=1；每設定first1次、warmup2次、量測7次。','','## 環境','',f"CPU：{summaries['qpp-cpu']['cpu']}。",f"GPU run後快照：{summaries['nvidia-fp64']['gpu_snapshot_after_run']}。",f"CUDA-Q：{summaries['qpp-cpu']['cudaq']}；Python：{summaries['qpp-cpu']['python']}。",'', '## First call與Warm latency','','| Backend | Qubits | Blocks | First ms | Median ms | Q25 ms | Q75 ms |','|---|---:|---:|---:|---:|---:|---:|']
    for b,rows in data.items():
        for r in rows:lines.append(f"| {b} | {r['qubits']} | {r['blocks']} | {1000*r['first_call_seconds']:.4f} | {1000*r['median_seconds']:.4f} | {1000*r['q25_seconds']:.4f} | {1000*r['q75_seconds']:.4f} |")
    lines+=['','First是該設定在此process的第一次呼叫，不保證cold compiler／disk cache；只有process首個case包含此process首次kernel執行。Process總wall time另存，包含Python import、輸出、setup與驗證。','','## CPU median / GPU median','','| Qubits | Blocks | GPU mode | Ratio (>1 GPU較快) | Max expectation error |','|---:|---:|---|---:|---:|']
    for r in comparison:lines.append(f"| {r['qubits']} | {r['blocks']} | {r['gpu_backend']} | {r['cpu_median_over_gpu_median']:.3f} | {r['max_output_error']:.3e} |")
    lines+=['','主比較為CPU fp64對GPU fp64；nvidia預設fp32是另一個precision條件，不把差異全歸因於GPU。','Ratio只適用本機、此電路、此thread設定與量測期間，不能當成通用加速倍數。IQR是7次重複的描述統計，不是confidence interval。','','## 限制','','最高16qubits的單份statevector下限為fp64 1MiB、fp32 0.5MiB，未量測peak RAM／VRAM，也未測試6GiB可容納的最大qubit數。','固定backend順序與固定case順序，未鎖頻、未控制OS背景負載或散熱；後段case可能受thermal／cache影響。CPU單thread不是最佳CPU調校。','相同參數反覆執行，不是QML training、batch throughput、noise trajectory或QPU benchmark；沒有手動關閉backend gate fusion。','GPU退出可能有cudaErrorCudartUnloading；請以保存的數值驗證與process exit status判斷本次執行，未定位該退出訊息根因。','','## 原始紀錄','','- [Protocol](protocol.json)、[Process wall times](process_times.json)、[Comparisons](comparison.json)']
    for b in data:lines.append(f'- [{b} records]({b}/records.json)、[{b} summary]({b}/summary.json)')
    lines+=['- [教學與重跑](../../articles/day27/README.md)',''];(OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':main()
