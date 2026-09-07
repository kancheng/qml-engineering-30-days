"""Render saved benchmark metrics and a fixed-weight re-uploading probe."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/day22-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/day22-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import OUT, read, write, fingerprint
from reuploading import np, COUNTS, SCHEDULES, reference, resources


def main():
    selected, candidates = read('selected'), read('candidates')
    summaries = {b: read(f'{b}/summary') for b in ('qpp-cpu', 'nvidia')}
    if read('training_summary')['artifact_sha256'] != fingerprint() or any(
            not s['passed'] or s['artifact_sha256'] != fingerprint() for s in summaries.values()):
        raise ValueError('verify current artifacts before generating the report')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name in COUNTS:
        group = [r for r in selected if r['model'] == name]
        axes[0].plot([r['split_seed'] for r in group], [r['scores']['test']['metrics']['brier'] for r in group], 'o-', label=name)
        for c in candidates:
            if c['model'] == name and c['split_seed'] == 2028:
                axes[1].plot([r['evaluations'] for r in c['history']], [r['loss'] for r in c['history']], label=f"{name}/{c['seed']}")
    axes[0].set(xticks=[2028, 2029], xlabel='Dataset seed', ylabel='Test Brier', title='Validation-selected seeds')
    axes[1].set(xlabel='Objective evaluations', ylabel='Train Brier', title='Split 2028 / both initializations')
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=6)
    fig.suptitle('Day22: binary Iris / NumPy reference training')
    fig.tight_layout(); fig.savefig(OUT/'comparison.png', dpi=160); plt.close(fig)

    # Hold all eight weights fixed: only the schedule changes. No training or labels.
    weights = np.random.default_rng(42).normal(0, .2, 8)
    grid = np.linspace(-1, 1, 31)
    probe = {'scope': 'fixed seed42 weights; synthetic scaled inputs, no labels or fitting',
             'weights': weights.tolist(), 'grid': grid.tolist(), 'surfaces': {}, 'mixed_contrast': {}}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, name in zip(axes, ('pca_once2', 'pca_repeat2')):
        def probability(a, b):
            return (1-reference(np.pi*(np.array([a, b])+1)/2, weights, SCHEDULES[name]))/2
        surface = np.array([[probability(a, b) for a in grid] for b in grid])
        probe['surfaces'][name] = surface.tolist()
        # Nonzero mixed contrast rules out additive f(x0)+g(x1) on these four points.
        probe['mixed_contrast'][name] = probability(.5, .5)-probability(.5, -.5)-probability(-.5, .5)+probability(-.5, -.5)
        im = ax.imshow(surface, origin='lower', extent=(-1, 1, -1, 1), vmin=0, vmax=1, cmap='viridis')
        ax.set(title=name, xlabel='Scaled feature 0', ylabel='Scaled feature 1')
        fig.colorbar(im, ax=ax, label='p1')
    fig.suptitle('Same fixed weights / different upload schedules')
    fig.tight_layout(); fig.savefig(OUT/'schedule_probe.png', dpi=160); plt.close(fig)
    write(OUT/'schedule_probe.json', probe)
    lines = ['# Day22 實驗結果', '', '由 `articles/day22/plot_results.py` 依 JSON 產生。', '',
             'Day20 binary Iris 99 筆，兩個 splits，各 train／validation／test=59／19／21。20 次 NumPy CPU reference fits，每次 73 個 objective、4,307 筆 train 輸入評估；CUDA-Q training calls=0。', '',
             '## 電路資源', '', '| Model | Parameters | Upload blocks | RY | CNOT | Logical depth |', '|---|---:|---:|---:|---:|---:|']
    for name in COUNTS:
        r = resources(name)
        lines.append(f"| {name} | {r['parameters']} | {r['upload_blocks']} | {r['ry_gates']} | {r['cnot_gates']} | {r['unoptimized_logical_depth']} |")
    lines += ['', '所有量子模型為兩 qubit。Depth 是未合併 gates、平行單 qubit 層計算的邏輯深度，不含 readout；不是編譯器或 QPU 實測深度。', '',
              '## Validation 選定模型', '', '| Split | Model | Seed | Test Brier | Accuracy | Confusion matrix | Clipped train/val/test |', '|---|---|---:|---:|---:|---|---|']
    for r in selected:
        m = r['scores']['test']['metrics']; clips = [r['scores'][s]['clipped_values'] for s in ('train','validation','test')]
        lines.append(f"| {r['split_seed']} | {r['model']} | {r['seed']} | {m['brier']:.6f} | {m['accuracy']:.2%} | {m['confusion_matrix']} | {clips} |")
    b = selected[0]['scores']['test']['constant_baseline']
    lines += ['', f"Train-prior baseline p1=29/59：test Brier={b['brier']:.6f}、accuracy={b['accuracy']:.2%}，兩 split 相同。Confusion matrix row=true、column=predicted；clipping 按輸入座標值計數，不乘 upload 次數。", '',
              '![Comparison](comparison.png)', '', '## 全部候選與訓練成本', '',
              '| Split | Model | Seed | Train Brier | Validation Brier | NumPy fit seconds | Full sweeps |', '|---|---|---:|---:|---:|---:|---:|']
    for c in candidates:
        lines.append(f"| {c['split_seed']} | {c['model']} | {c['seed']} | {c['history'][-1]['loss']:.6f} | {c['validation']['brier']:.6f} | {c['fit_seconds_numpy']:.3f} | {c['full_sweeps']} |")
    lines += ['', '## 凍結模型驗證', '', '| Backend | Records | Observe calls | Max probability error | Passed |', '|---|---:|---:|---:|---|']
    for b, s in summaries.items():
        lines.append(f"| {b} | {s['records']} | {s['observe_calls']} | {s['max_reference_error']:.3e} | {s['passed']} |")
    lines += ['', '每 backend：2 splits×99 筆×4 量子模型=792 次 exact observe。另核對 logistic4 NumPy CPU，合計 30 個 partition records。版本、precision、UTC timestamp 與 artifact SHA-256 見 summaries。',
              'GPU 程序結束有 cudaErrorCudartUnloading 訊息；驗證及測試 exit code=0，根因未定位。時間可能含 compilation／cache，不用於 GPU speedup 結論。', '',
              '## 相同權重的輸入反應', '', '![Schedule probe](schedule_probe.png)', '',
              '兩張圖使用相同 seed42 的八個未訓練參數。只改 upload schedule；這不是 test decision boundary，也不是 expressibility 的統計測量。', '',
              '在 scaled inputs (±0.5, ±0.5) 的 mixed contrast：']
    for name, value in probe['mixed_contrast'].items():
        lines.append(f'- {name}: {value:.8f}')
    lines += ['', '非零 contrast 表示在這四點無法寫成 f(x0)+g(x1)。單次 upload 也可能有 interaction；re-uploading 改變輸入反應，不保證 interaction 或 accuracy 必然增加。', '',
              '## 解讀與限制', '',
              '配對內 trainable blocks、初始化與 objective 預算相同，額外 data gates 仍增加操作量；配對間的 PCA2／四維 chunks 表示及參數數量不同，不能直接隔離單一原因。',
              '這次結果沒有顯示 re-uploading 必然優於 once controls。固定 73 次 objective 對 8／16 參數只有 4／2 次 full sweeps；未證明充分收斂。',
              '兩個重疊的小 splits、Day20 已公開的 test、exact 無噪聲 simulator，只支持探索性示範。沒有論文完整復現、universal approximation 或量子優勢結論。', '',
              '## 原始紀錄', '',
              '- [Protocol](protocol.json)、[資料與 preprocessing](datasets.json)、[全部 candidates](candidates.json)',
              '- [選定模型](selected.json)、[Training summary](training_summary.json)、[固定權重 probe](schedule_probe.json)',
              '- [CPU summary](qpp-cpu/summary.json)、[CPU predictions](qpp-cpu/verification.json)',
              '- [GPU summary](nvidia/summary.json)、[GPU predictions](nvidia/verification.json)',
              '- [教學及重跑指令](../../articles/day22/README.md)', '']
    (OUT/'README.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
