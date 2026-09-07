"""Reproducible hypothetical scaling analysis; never a multi-GPU measurement."""
from pathlib import Path
import hashlib
import json
import math
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/day28'


def capacity(gpus, gib=6, fraction=0.75, complex_bytes=16):
    """Single evenly sharded buffer, hypothetical usable-memory fraction."""
    if gpus < 1 or gpus & (gpus - 1):
        raise ValueError('GPU count must be a positive power of two')
    if gib <= 0 or not 0 < fraction <= 1 or complex_bytes not in (8, 16):
        raise ValueError('invalid memory assumptions')
    return math.floor(math.log2(gpus * gib * 2**30 * fraction / complex_bytes))


def latency(t1, gpus, serial_fraction, overhead_fraction):
    """Amdahl + assumed log2(P) overhead, all relative to measured T1."""
    if t1 <= 0 or gpus < 1 or gpus & (gpus - 1):
        raise ValueError('invalid baseline or GPU count')
    if not 0 <= serial_fraction <= 1 or overhead_fraction < 0:
        raise ValueError('invalid fractions')
    return t1 * (serial_fraction + (1 - serial_fraction) / gpus
                 + overhead_fraction * math.log2(gpus))


def probe():
    cmd = ['nvidia-smi', '--query-gpu=name,uuid,memory.total', '--format=csv,noheader']
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        gpu = dict(command=cmd, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        gpu = dict(command=cmd, returncode=None, error=str(exc))
    return dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                python=platform.python_version(), cudaq=version('cuda-quantum-cu12'),
                nvidia_smi=gpu, mpiexec=shutil.which('mpiexec'),
                multi_gpu_measurement='not performed',
                note='Probe visibility is not proof of physical GPU count or MPI readiness.')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source = ROOT / 'results/day27/nvidia-fp64/records.json'
    records = json.loads(source.read_text())
    baseline = next(r for r in records if r['qubits'] == 16 and r['blocks'] == 12)
    if not baseline['passed'] or not math.isfinite(baseline['median_seconds']) or baseline['median_seconds'] <= 0:
        raise ValueError('invalid Day27 baseline')
    t1 = baseline['median_seconds']
    counts = [1, 2, 4, 8]
    capacities = [dict(gpus=p, precision=precision, assumed_gib_per_gpu=6,
                       usable_fraction=f, max_qubits_buffer_only=capacity(p, fraction=f, complex_bytes=b))
                  for precision, b in [('fp32', 8), ('fp64', 16)]
                  for f in [1.0, 0.75] for p in counts]
    scenarios = []
    for s, c in [(0, 0), (0.1, 0.05), (0.3, 0.2)]:
        for p in counts:
            tp = latency(t1, p, s, c)
            scenarios.append(dict(gpus=p, serial_fraction=s, overhead_fraction=c,
                                  modeled_seconds=tp, speedup=t1/tp, efficiency=t1/tp/p))
    OUT.mkdir(parents=True, exist_ok=True)
    data = dict(kind='hypothetical model, not multi-GPU measurements',
                baseline_source=str(source.relative_to(ROOT)),
                baseline_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                baseline_qubits=16, baseline_blocks=12, baseline_seconds=t1,
                formula='T(P)=T1*(s+(1-s)/P+c*log2(P))',
                capacities=capacities, scenarios=scenarios)
    for name, obj in [('model', data), ('environment', probe())]:
        (OUT / f'{name}.json').write_text(json.dumps(obj, indent=2) + '\n')
    fig, ax = plt.subplots(figsize=(7, 4))
    for s, c in [(0, 0), (0.1, 0.05), (0.3, 0.2)]:
        rows = [r for r in scenarios if r['serial_fraction'] == s and r['overhead_fraction'] == c]
        ax.plot(counts, [r['speedup'] for r in rows], 'o-', label=f'assumed s={s}, c={c}')
    ax.set(xlabel='Hypothetical GPU count', ylabel='Modeled T(1) / T(P)',
           title='Scenario analysis — NOT measured multi-GPU speedup', xticks=counts)
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'scaling.png', dpi=160)
    plt.close(fig)
    lines = ['# Day28｜可重現 Scaling Model 報告', '',
             '**本報告沒有 multi-GPU 實測。**容量與延遲皆為明確假設下的模型輸出。', '',
             f'Day27 單GPU fp64、16qubits／12blocks warm median：{t1*1000:.4f} ms。來源與 SHA-256 見 [model.json](model.json)。', '',
             '![Hypothetical scaling](scaling.png)', '',
             '## 容量模型', '',
             '假設每卡6GiB、均勻切分單份statevector；75%是任意預留情境，非runtime實測。表中數字不是可執行上限。', '',
             '| GPUs | Precision | 可用比例 | 單buffer qubit上限 |', '|---:|---|---:|---:|']
    lines += [f"| {r['gpus']} | {r['precision']} | {r['usable_fraction']:.0%} | {r['max_qubits_buffer_only']} |" for r in capacities]
    lines += ['', '## 延遲敏感度', '', 's是序列比例，c是每log2(P)級通訊／協調成本相對T1的比例；兩者都未經量測或擬合。', '',
              '| GPUs | s | c | 模型 ms | Speedup | Efficiency |', '|---:|---:|---:|---:|---:|---:|']
    lines += [f"| {r['gpus']} | {r['serial_fraction']} | {r['overhead_fraction']} | {r['modeled_seconds']*1000:.4f} | {r['speedup']:.3f} | {r['efficiency']:.3f} |" for r in scenarios]
    lines += ['', '固定問題規模的情境分析；不外推更大qubit時間，不代表mgpu backend對16qubits實際啟用分散。', '',
              '硬體可見性與MPI executable探測見 [environment.json](environment.json)；探測失敗不表示主機沒有GPU。', '',
              '重跑：`OMP_NUM_THREADS=1 .venv/bin/python articles/day28/scaling.py`。會覆寫本日報告、JSON與圖，不執行量子模擬。', '',
              '[教學與模型限制](../../articles/day28/README.md)', '']
    (OUT / 'README.md').write_text('\n'.join(lines))
    print(f'Wrote {OUT}; 16 capacity rows, 12 hypothetical latency rows.')


if __name__ == '__main__':
    main()
