"""Recompute selected project evidence from saved artifacts; no training or QPU jobs."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/day30'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metrics(probabilities, labels):
    require(len(probabilities) == len(labels) and len(labels) > 0, 'prediction/label length mismatch')
    require(all(y in (0, 1) for y in labels), 'nonbinary labels')
    require(all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities), 'invalid probabilities/scores')
    cm = [[0, 0], [0, 0]]
    for p, y in zip(probabilities, labels):
        cm[y][int(p >= .5)] += 1
    return {'brier': sum((p-y)**2 for p, y in zip(probabilities, labels))/len(labels),
            'accuracy': (cm[0][0]+cm[1][1])/len(labels), 'confusion_matrix': cm}


def compare_metrics(actual, saved):
    require(actual['confusion_matrix'] == saved['confusion_matrix'], 'confusion mismatch')
    for name in ('brier', 'accuracy'):
        require(math.isclose(actual[name], saved[name], rel_tol=0, abs_tol=1e-12), f'{name} mismatch')


def pick(candidates, field):
    require(bool(candidates), 'missing candidates')
    require(all(math.isfinite(c['validation']['brier']) for c in candidates), 'invalid validation score')
    return min(candidates, key=lambda c: (c['validation']['brier'], c[field]))


class Sources:
    def __init__(self, root=ROOT):
        self.root = root
        self.hashes = {}

    def read(self, relative):
        raw = (self.root/relative).read_bytes()
        self.hashes[relative] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)


def audit_benchmark(sources, day):
    prefix = f'results/day{day}'
    selected = sources.read(f'{prefix}/selected.json')
    candidates = sources.read(f'{prefix}/candidates.json')
    datasets = sources.read(f'{prefix}/datasets.json')
    protocol = sources.read(f'{prefix}/protocol.json')
    field = 'lambda' if day == 23 else 'seed'
    values = protocol['lambdas'] if day == 23 else protocol['initialization_seeds']
    expected = {(s, model) for s in protocol['split_seeds'] for model in protocol['models']}
    require(len(selected) == len(expected) and {(r['split_seed'], r['model']) for r in selected} == expected,
            'incomplete selected models')
    candidate_keys = {(r['split_seed'], r['model'], r[field]) for r in candidates}
    require(len(candidates) == len(expected)*len(values) and
            candidate_keys == {(s, m, v) for s, m in expected for v in values}, 'incomplete candidates')
    for dataset in datasets.values():
        partitions = [part for part in dataset['splits'].values()]
        ids = [r['id'] for part in partitions for r in part]
        require(len(ids) == len(set(ids)), 'overlapping or duplicate IDs within a split')
    rows = []
    for row in selected:
        peers = [c for c in candidates if c['split_seed'] == row['split_seed'] and c['model'] == row['model']]
        chosen = pick(peers, field)
        require(chosen[field] == row[field], 'selection is not validation-optimal')
        params = 'alpha' if day == 23 else 'weights'
        require(chosen[params] == row[params], 'selected parameters mismatch')
        for partition in ('train', 'validation', 'test'):
            score = row['scores'][partition]
            probabilities = score['clipped_scores' if day == 23 else 'probabilities']
            if day == 23:
                require(probabilities == [min(1., max(0., p)) for p in score['raw_scores']], 'clipping mismatch')
            labels = [r['label'] for r in datasets[str(row['split_seed'])]['splits'][partition]]
            actual = metrics(probabilities, labels)
            compare_metrics(actual, score['metrics'])
            if partition == 'validation':
                compare_metrics(actual, chosen['validation'])
            if partition == 'test':
                rows.append(dict(day=day, task='Wine binary' if day == 25 else 'Iris binary',
                                 split=row['split_seed'], model=row['model'], selection_field=field,
                                 selected_value=row[field], test_size=len(labels), **actual))
    return rows


def audit_timing(sources):
    cpu = sources.read('results/day27/qpp-cpu/records.json')
    gpu = sources.read('results/day27/nvidia-fp64/records.json')
    saved = sources.read('results/day27/comparison.json')
    protocol = sources.read('results/day27/protocol.json')
    for target in ('qpp-cpu', 'nvidia-fp64'):
        summary = sources.read(f'results/day27/{target}/summary.json')
        require(summary['passed'] and 'fp64' in summary['precision'].lower(), 'invalid timing precision')
        require(summary['protocol'] == protocol and summary['OMP_NUM_THREADS'] == '1', 'protocol mismatch')
    expected = {(n, b) for n in protocol['qubits'] for b in protocol['blocks']}
    require(len(cpu) == len(gpu) == len(expected), 'timing case count mismatch')
    for group in (cpu, gpu):
        require({(r['qubits'], r['blocks']) for r in group} == expected, 'timing cases mismatch')
    result = []
    for c in cpu:
        g = next(r for r in gpu if (r['qubits'], r['blocks']) == (c['qubits'], c['blocks']))
        require(c['weights_sha256'] == g['weights_sha256'], 'timing weights mismatch')
        for r in (c, g):
            require(r['passed'] and len(r['seconds']) == len(r['values']) == 7, 'invalid timing repeats')
            require(all(math.isfinite(t) and t > 0 for t in r['seconds']), 'invalid latency')
            require(all(math.isfinite(v) for v in r['values']), 'invalid outputs')
            require(math.isclose(statistics.median(r['seconds']), r['median_seconds']), 'median mismatch')
        # All pairs of warm outputs, rather than assuming paired random seeds.
        error = max(abs(a-b) for a in c['values'] for b in g['values'])
        require(error < 1e-5, 'CPU/GPU expectation mismatch')
        ratio = statistics.median(c['seconds'])/statistics.median(g['seconds'])
        comparison = next(r for r in saved if r['qubits'] == c['qubits'] and r['blocks'] == c['blocks']
                          and r['gpu_backend'] == 'nvidia-fp64')
        require(math.isclose(ratio, comparison['cpu_median_over_gpu_median']), 'ratio mismatch')
        result.append(dict(qubits=c['qubits'], blocks=c['blocks'], cpu_over_gpu=ratio,
                           cpu_ms=c['median_seconds']*1000, gpu_ms=g['median_seconds']*1000))
    return result


def build():
    sources = Sources()
    benchmarks = sum((audit_benchmark(sources, day) for day in (20, 23, 25)), [])
    deltas = []
    for row in benchmarks:
        if row['day'] in (20, 25) and row['model'] in ('vqc', 'hybrid'):
            baseline = next(r for r in benchmarks if r['day'] == row['day'] and r['split'] == row['split'] and r['model'] == 'mlp')
            deltas.append(dict(day=row['day'], split=row['split'], model=row['model'],
                               delta_brier_vs_mlp=row['brier']-baseline['brier']))
    training = {str(day): sources.read(f'results/day{day}/training_summary.json') for day in (20, 25)}
    require(all(s['passed'] and s['cudaq_training_calls'] == 0 for s in training.values()), 'training scope changed')
    noise = sources.read('results/day26/wine.json')
    datasets = sources.read('results/day25/datasets.json')
    require(len(noise) == 18 and len({(r['split_seed'], r['channel'], r['p']) for r in noise}) == 18, 'noise case count')
    for r in noise:
        require(r['passed'], 'failed noise record')
        labels = [d['label'] for d in datasets[str(r['split_seed'])]['splits']['test']]
        compare_metrics(metrics(r['probabilities'], labels), r['metrics'])
    noise_rows = [dict(split=r['split_seed'], channel=r['channel'], p=r['p'], **r['metrics']) for r in noise]
    timing = audit_timing(sources)
    model = sources.read('results/day28/model.json')
    require('not multi-GPU measurements' in model['kind'], 'Day28 evidence type changed')
    sources.read(model['baseline_source'])
    require(model['baseline_sha256'] == sources.hashes[model['baseline_source']], 'stale Day28 baseline')
    shot_modes = []
    for mode in ('qpp-cpu', 'ionq-emulate', 'density-bitflip'):
        d = sources.read(f'results/day29/{mode}.json')
        require(d['passed'] and d['remote_submitted'] is False and d['job_ids'] == [], 'Day29 execution scope changed')
        require(len(d['records']) == 30 and d['sample_calls'] == 30, 'missing shot records')
        total = 0
        for r in d['records']:
            counts = r['counts']
            require(all(k in ('00', '01', '10', '11') and type(v) is int and v >= 0 for k, v in counts.items()), 'invalid counts')
            require(sum(counts.values()) == r['shots'], 'shots mismatch')
            total += r['shots']
        require(total == d['requested_shots'], 'shot total mismatch')
        shot_modes.append(dict(mode=mode, kind=d['execution_kind'], sample_calls=30, shots=total))
    plan = sources.read('results/day29/submission_plan.json')
    require(plan['status'] == 'not_submitted' and plan['job_ids'] == [], 'submission scope changed')
    inventory = []
    for day in range(1, 31):
        directory = ROOT/f'articles/day{day:02}'
        article = directory/'README.md'
        require(article.exists(), f'missing Day{day} article')
        scripts = sorted(str(p.relative_to(ROOT)) for p in directory.glob('*.py'))
        notebooks = sorted(str(p.relative_to(ROOT)) for p in (ROOT/'notebooks').glob(f'day{day:02}*.ipynb'))
        require(day < 3 or scripts or notebooks, f'missing executable Day{day} artifact')
        inventory.append(dict(day=day, article=str(article.relative_to(ROOT)), scripts=scripts, notebooks=notebooks))
    return dict(scope='Saved evidence audit; no experiment reruns, no new training, no remote jobs',
                benchmarks=benchmarks, paired_deltas=deltas, timing_fp64=timing, noise=noise_rows,
                training={day: {'fits': r['fits'], 'cudaq_training_calls': r['cudaq_training_calls']} for day, r in training.items()},
                multi_gpu={'kind': model['kind'], 'capacity_rows': len(model['capacities']), 'latency_rows': len(model['scenarios'])},
                qpu={'status': plan['status'], 'physical_qpu_jobs': 0, 'local_modes': shot_modes,
                     'planned_shots': plan['total_requested_shots']}, inventory=inventory,
                source_sha256=sources.hashes)


def render(data):
    lines = ['# Day30｜系列證據彙整', '',
             '由保存的JSON重算；本日沒有重訓、重跑GPU benchmark或提交QPU。來源SHA-256見[evidence.json](evidence.json)。', '',
             '## 分類與Kernel結果', '',
             '20個選定模型、60組train／validation／test metrics重新核算。依各日validation選seed或λ；test未用於本次選擇。',
             'Day20與Day25的MLP／VQC／Hybrid用PCA2；Day25 logistic13用完整13維；Day23三kernel共用四維縮放輸入。',
             '各任務分開解讀，不跨日合併成總accuracy或顯著性檢定。', '',
             '| Day | Split | Model | Test N | Brier | Accuracy |', '|---:|---:|---|---:|---:|---:|']
    for r in data['benchmarks']:
        lines.append(f"| {r['day']} | {r['split']} | {r['model']} | {r['test_size']} | {r['brier']:.6f} | {r['accuracy']:.2%} |")
    lines += ['', '## 相同PCA2輸入的描述性差值', '',
              'ΔBrier = model − MLP；負值表示本次較低。不是confidence interval或統計優勢。', '',
              '![Paired Brier differences](paired_brier.png)', '',
              '| Day | Split | Model | ΔBrier |', '|---:|---:|---|---:|']
    for r in data['paired_deltas']:
        lines.append(f"| {r['day']} | {r['split']} | {r['model']} | {r['delta_brier_vs_mlp']:+.6f} |")
    lines += ['', '## Day27 fp64計時核對', '',
              '由每case的7次warm latency重算median與比值，確認weights hash及輸出一致。CPU固定OMP_NUM_THREADS=1。',
              '這是特定simulator inference比較；不是QPU或QML training speedup。', '',
              '| Qubits | Blocks | CPU ms | GPU ms | CPU/GPU |', '|---:|---:|---:|---:|---:|']
    for r in data['timing_fp64']:
        lines.append(f"| {r['qubits']} | {r['blocks']} | {r['cpu_ms']:.4f} | {r['gpu_ms']:.4f} | {r['cpu_over_gpu']:.3f} |")
    lines += ['', '## Noise與執行邊界', '',
              'Day26的18組Wine noise metrics由保存機率與Day25 labels重新核算。以下保留bit flip兩端情境：', '',
              '| Split | p | Test Brier | Accuracy |', '|---:|---:|---:|---:|']
    for r in data['noise']:
        if r['channel'] == 'bit_flip' and r['p'] in (0., .3):
            lines.append(f"| {r['split']} | {r['p']} | {r['brier']:.6f} | {r['accuracy']:.2%} |")
    lines += ['', '末端q0 channel不是全電路noise，也不是noise-aware training。', '',
              '- Day20／25共28次NumPy reference fits；兩日CUDA-Q training calls皆為0。',
              '- Day28：16組容量與12組延遲模型，核對Day27 baseline hash；無多卡實測。',
              '- Day29：90組本地counts，280,320 requested shots；不含96個probe shots。Physical QPU jobs=0。',
              '- Day29的2048-shot工作規格仍為not_submitted；費用與queue未知。', '',
              '## 系列交付索引', '',
              '已核對30篇文章存在，Day03–30各有Python程式或Notebook。這是檔案存在性檢查，不是全部歷史測試重跑。', '',
              '| Day | Article | Python files | Notebooks |', '|---:|---|---:|---:|']
    for r in data['inventory']:
        lines.append(f"| {r['day']:02} | [文章](../../{r['article']}) | {len(r['scripts'])} | {len(r['notebooks'])} |")
    lines += ['', '重跑：`.venv/bin/python articles/day30/evidence.py`；驗證保存彙整未過期：加`--check`。',
              '此audit核對保存資料的一致性，未重算所有模型forward、preprocessing或先前全部測試。', '',
              '[總結文章與後續研究規格](../../articles/day30/README.md)', '']
    return '\n'.join(lines)


def plot(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, day in zip(axes, (20, 25)):
        rows = [r for r in data['paired_deltas'] if r['day'] == day]
        ax.bar(range(len(rows)), [r['delta_brier_vs_mlp'] for r in rows],
               color=['#2463a5' if r['model'] == 'vqc' else '#c66b29' for r in rows])
        ax.set_xticks(range(len(rows)), [f"{r['split']}\n{r['model']}" for r in rows])
        ax.axhline(0, color='black', linewidth=.8)
        ax.set_title('Iris binary / Day20' if day == 20 else 'Wine binary / Day25')
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Test Brier difference vs MLP (lower is better)')
    fig.suptitle('Same PCA2 inputs within each task — descriptive differences only')
    fig.tight_layout()
    fig.savefig(OUT/'paired_brier.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Compare current sources to saved JSON/report without writing')
    args = parser.parse_args()
    data = build()
    report = render(data)
    if args.check:
        require(json.loads((OUT/'evidence.json').read_text()) == data, 'saved evidence is stale; rerun generator')
        require((OUT/'README.md').read_text() == report, 'saved report is stale')
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        plot(data)
        (OUT/'evidence.json').write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
        (OUT/'README.md').write_text(report)
    print(f"Validated {len(data['source_sha256'])} source artifacts, 20 selected models, 8 timing pairs, 30 article entries.")


if __name__ == '__main__':
    main()
