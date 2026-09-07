"""Audit persisted counts and produce the local rehearsal report."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from execution import ROOT, MODES, SHOTS, SEEDS, THETA, NOISE_P, OBSERVABLES, estimate, reference, submission_plan
OUT = ROOT/'results/day29'


def audit(data):
    assert data['mode'] in MODES and not data['remote_submitted']
    assert data['passed'] and data['theta'] == THETA
    assert data['ordering_probe'] == {'10': 32}
    assert data['noise_p'] == (NOISE_P if data['mode'] == 'density-bitflip' else 0.)
    assert data['analytic_reference'] == reference(THETA, data['noise_p'])
    for source, digest in data['source_sha256'].items():
        assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest() == digest
    expected = {(s, seed, b) for s in SHOTS for seed in SEEDS for b in ('Z', 'X')}
    actual = {(r['shots'], r['seed'], r['basis']) for r in data['records']}
    assert actual == expected and len(data['records']) == len(expected)
    assert data['sample_calls'] == len(expected)
    assert data['requested_shots'] == sum(r['shots'] for r in data['records'])
    exact_names = set() if data['mode'] == 'ionq-emulate' else {'Z0', 'ZZ', 'X0', 'XX'}
    assert set(data['exact_simulator']) == exact_names
    for name, value in data['exact_simulator'].items():
        assert math.isfinite(value) and abs(value-data['analytic_reference'][name]) < 1e-10
    for row in data['records']:
        names = OBSERVABLES[0 if row['basis'] == 'Z' else 1]
        assert set(row['statistics']) == set(names)
        bound = math.sqrt(2*math.log(2*180/.01)/row['shots'])
        assert math.isclose(row['hoeffding_family99_bound'], bound)
        for name in names:
            saved = row['statistics'][name]
            recomputed = estimate(row['counts'], row['shots'], name)
            assert all(saved[k] == v for k, v in recomputed.items())
            assert saved['reference'] == data['analytic_reference'][name]
            assert saved['error'] == saved['value']-saved['reference']
            assert abs(saved['error']) <= bound
        assert row['passed'] and math.isfinite(row['sample_wall_seconds']) and row['sample_wall_seconds'] > 0


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    all_data = [json.loads((OUT/f'{mode}.json').read_text()) for mode in MODES]
    for data in all_data:
        audit(data)
    summaries = []
    for data in all_data:
        for shots in SHOTS:
            for name in ('Z0', 'ZZ', 'X0', 'XX'):
                values = [r['statistics'][name]['value'] for r in data['records']
                          if r['shots'] == shots and name in r['statistics']]
                ref = data['analytic_reference'][name]
                summaries.append(dict(mode=data['mode'], shots=shots, observable=name,
                                      mean=float(np.mean(values)), reference=ref,
                                      rmse=float(np.sqrt(np.mean((np.array(values)-ref)**2)))))
    (OUT/'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
    plan = submission_plan()
    plan['source_sha256'] = all_data[0]['source_sha256']
    (OUT/'submission_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for mode in MODES:
        for name, ax in [('Z0', axes[0]), ('ZZ', axes[1])]:
            rows = [r for r in summaries if r['mode'] == mode and r['observable'] == name]
            ax.plot(SHOTS, [r['mean'] for r in rows], 'o-', label=mode)
            ax.axhline(rows[0]['reference'], color='gray', alpha=.35, linestyle='--')
            ax.set(xscale='log', xlabel='Shots per circuit', ylabel=f'Mean {name} (5 seeds)')
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Local rehearsal only — no physical QPU results')
    fig.tight_layout()
    fig.savefig(OUT/'shots.png', dpi=160)
    plt.close(fig)
    lines = ['# Day29｜Simulator → QPU 本地預演報告', '',
             '**全部結果來自本地執行；未提交遠端工作，沒有QPU排隊、費用或硬體時間實測。**', '',
             f"CUDA-Q {all_data[0]['cudaq']}、Python {all_data[0]['python']}；OMP_NUM_THREADS=1、CUDAQ_DEFAULT_SIMULATOR=qpp。", '',
             '## 實驗設定', '',
             '固定theta=π/3、2qubits；Z／X兩基底；128／1024／8192 shots；seeds42–46。每模式30次sample，共93,440 shots。',
             '三模式合計90次sample、280,320 shots，另有三次32-shot位元順序probe與8次exact observe核對。', '',
             '| Mode | 執行性質 | Noise | Exact reference |', '|---|---|---|---|']
    for data in all_data:
        lines.append(f"| {data['mode']} | {data['execution_kind']} | p={data['noise_p']} | {'解析式＋exact observe' if data['exact_simulator'] else '解析式'} |")
    lines += ['', '![Shot comparison](shots.png)', '',
              '## 五個Seeds的統計', '',
              'RMSE相對各模式自己的期望值；不是相對理想電路的硬體誤差。每列只有5次重複，趨勢不保證單調。', '',
              '| Mode | Shots | Observable | Mean | Reference | RMSE |', '|---|---:|---|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['mode']} | {r['shots']} | {r['observable']} | {r['mean']:.5f} | {r['reference']:.5f} | {r['rmse']:.5f} |")
    lines += ['', '## 驗證與限制', '',
              '三模式的非對稱|10⟩ probe皆為10:32。CPU理想／density matrix合計8個exact expectations與解析式誤差<1e-10。',
              '全部180個抽樣expectations通過預先指定的Hoeffding family-wise 99%誤差界；此寬鬆檢查用於抓明顯實作錯誤。',
              '各原始JSON另存pointwise 95% Wilson區間；不是180組同時95%區間，也不包含校準漂移或noise模型不確定性。',
              '每次sample的host wall time包含編譯／執行等成本，沒有分離暖機；不作backend效能比較。', '',
              '## 原始資料與待提交規格', '',
              '- [CPU counts](qpp-cpu.json)、[IonQ local emulation counts](ionq-emulate.json)、[Synthetic bit-flip counts](density-bitflip.json)',
              '- [Summary](summary.json)、[未提交的工作規格](submission_plan.json)',
              '- [文章與重跑](../../articles/day29/README.md)', '']
    (OUT/'README.md').write_text('\n'.join(lines))
    print('Audited 90 count records / 180 estimates; wrote report and offline plan.')


if __name__ == '__main__':
    main()
