"""Generate plots and a report directly from saved, verified artifacts."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/day21-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/day21-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import OUT, read, fingerprint
from reduction import np, representation


def main():
    selected, candidates, datasets = read('selected'), read('candidates'), read('datasets')
    summaries = {b: read(f'{b}/summary') for b in ('qpp-cpu', 'nvidia')}
    if read('training_summary')['artifact_sha256'] != fingerprint() or any(
            not s['passed'] or s['artifact_sha256'] != fingerprint() for s in summaries.values()):
        raise ValueError('re-run verification for the current training artifacts')
    names = ['pca_vqc', 'selection_vqc', 'bottleneck_vqc', 'logistic4']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name in names:
        group = [r for r in selected if r['model'] == name]
        axes[0].plot([r['split_seed'] for r in group], [r['scores']['test']['metrics']['brier'] for r in group], 'o-', label=name)
        for c in candidates:
            if c['model'] == name and c['split_seed'] == 2028:
                axes[1].plot([r['evaluations'] for r in c['history']], [r['loss'] for r in c['history']],
                             label=f"{name} / {c['seed']}")
    axes[0].set(xticks=[2028, 2029], xlabel='Dataset seed', ylabel='Test Brier (lower is better)', title='Validation-selected seeds')
    axes[1].set(xlabel='Objective evaluations', ylabel='Train Brier', title='Split 2028 / both initializations')
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.suptitle('Day21: binary Iris / NumPy reference training')
    fig.tight_layout()
    fig.savefig(OUT / 'comparison.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    data = datasets['2028']
    for ax, name in zip(axes, names[:3]):
        model = next(r for r in selected if r['split_seed'] == 2028 and r['model'] == name)
        for split, marker in [('train', 'o'), ('test', 'x')]:
            rows = data['splits'][split]
            x, _ = representation(name, [r['features'] for r in rows], data['preprocessor'], model['weights'])
            for label, color in [(0, 'tab:blue'), (1, 'tab:orange')]:
                mask = np.array([r['label'] == label for r in rows])
                ax.scatter(*x[mask].T, color=color, marker=marker, s=22, alpha=.65, label=f'{split} / y={label}')
        ax.set(title=name, xlabel='Encoded feature 0', ylabel='Encoded feature 1', xlim=(-1.08, 1.08), ylim=(-1.08, 1.08))
    axes[0].legend(fontsize=6)
    fig.suptitle('Split 2028: frozen representations (test only transformed)')
    fig.tight_layout()
    fig.savefig(OUT / 'representations.png', dpi=160)
    plt.close(fig)
    lines = ['# Day21 實驗結果', '', '本檔由 `articles/day21/plot_results.py` 依保存的 JSON 產生。', '',
             '資料：Day20 去重 binary Iris 99 筆，split seeds 2028／2029，train／validation／test=59／19／21。',
             '16 次 fit 全部是 NumPy CPU exact reference；每次 73 個 Brier objectives、4,307 筆 train 輸入評估。CUDA-Q 僅驗證凍結模型。', '',
             '## 選定模型', '', '| Split | Model | Seed | Test Brier | Accuracy | Confusion matrix |', '|---|---|---:|---:|---:|---|']
    for r in selected:
        m = r['scores']['test']['metrics']
        lines.append(f"| {r['split_seed']} | {r['model']} | {r['seed']} | {m['brier']:.6f} | {m['accuracy']:.2%} | {m['confusion_matrix']} |")
    baseline = selected[0]['scores']['test']['constant_baseline']
    lines += ['', f"Train-prior baseline p1=29/59：test Brier={baseline['brier']:.6f}，accuracy={baseline['accuracy']:.2%}（兩組 test 類別數相同）。Confusion matrix row=true、column=predicted。", '',
              '![Comparison](comparison.png)', '', '## 表示與範圍', '',
              '| Split | 選取欄位（zero-based，依 score 排序） | PCA2 variance ratio 合計 |', '|---|---|---:|']
    for seed, d in datasets.items():
        p = d['preprocessor']
        lines.append(f"| {seed} | {p['selected_indices']} | {sum(p['pca']['explained_variance_ratio']):.6f} |")
    lines += ['', '欄位 0／1／2／3 為 sepal length／sepal width／petal length／petal width；完整 train correlations 與 scaler 見 datasets.json。', '',
              '| Split | Model | Train / validation / test clipped values | Bottleneck test abs(h)>0.99 |', '|---|---|---|---:|']
    for r in selected:
        clips = [r['scores'][s]['clipped_values'] for s in ('train', 'validation', 'test')]
        saturated = r['scores']['test']['saturated_values_abs_gt_099']
        lines.append(f"| {r['split_seed']} | {r['model']} | {clips} | {saturated if saturated is not None else '—'} |")
    lines += ['', 'Clipping 按座標值計數；bottleneck 使用 tanh，logistic4 不做 clipping。PCA variance 是標準化 train variance，不是分類資訊保留率。', '',
              '![Representations](representations.png)', '', '## 所有訓練候選', '',
              '| Split | Model | Seed | Train Brier | Validation Brier | NumPy fit seconds | Full sweeps |', '|---|---|---:|---:|---:|---:|---:|']
    for c in candidates:
        lines.append(f"| {c['split_seed']} | {c['model']} | {c['seed']} | {c['history'][-1]['loss']:.6f} | {c['validation']['brier']:.6f} | {c['fit_seconds_numpy']:.3f} | {c['full_sweeps']} |")
    lines += ['', '## CUDA-Q 驗證', '', '| Backend | Records | Observe calls | Max probability error | Passed |', '|---|---:|---:|---:|---|']
    for backend, s in summaries.items():
        lines.append(f"| {backend} | {s['records']} | {s['observe_calls']} | {s['max_reference_error']:.3e} | {s['passed']} |")
    lines += ['', '每 backend：2 splits × 99 筆 × 3 量子模型=594 observe；classical baseline 仍用 NumPy CPU。shots=-1、無噪聲，時間包含可能的 compilation／cache，不是隔離效能測量。', '',
              '環境版本、precision、UTC timestamp 與 artifact SHA-256 見各 summary。', '',
              '## 範圍與解讀', '',
              '這是固定小預算、兩個重疊 splits 的示範。PCA／selection 固定表示，bottleneck 同時訓練 10 個 encoder 與 4 個量子參數；objective 次數相同不等於相同 sweeps 或模型容量。',
              '完整四維 logistic-link baseline 使用 Brier＋座標搜尋，並非標準 cross-entropy LogisticRegression solver。結果不證明量子優勢，也不能單獨把準確率差異歸因於降維方法。',
              '沿用 Day20 已公開的 test，屬探索性系列實驗；正式模型選擇需新的 holdout 或 nested cross-validation。', '',
              '## 原始紀錄', '',
              '- [Protocol](protocol.json)、[資料與 preprocessing](datasets.json)、[全部 candidates](candidates.json)',
              '- [Selected models](selected.json)、[Training summary](training_summary.json)',
              '- [CPU summary](qpp-cpu/summary.json)、[CPU predictions](qpp-cpu/verification.json)',
              '- [GPU summary](nvidia/summary.json)、[GPU predictions](nvidia/verification.json)',
              '- [教學與重跑指令](../../articles/day21/README.md)', '']
    (OUT / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
