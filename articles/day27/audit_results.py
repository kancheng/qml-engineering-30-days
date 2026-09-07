"""Validate saved measurements without repeating the timing workload."""
from benchmark import PROTOCOL,read,np,weights,hashlib
from articles.day24.landscape import reference

def main():
    expected={(n,l) for n in PROTOCOL['qubits'] for l in PROTOCOL['blocks']}
    all_records={}
    for backend in PROTOCOL['targets']:
        rows=read(f'{backend}/records');summary=read(f'{backend}/summary')
        assert summary['protocol']==PROTOCOL and summary['passed']
        assert summary['observe_calls']==len(expected)*10
        assert {(r['qubits'],r['blocks']) for r in rows}==expected and len(rows)==len(expected)
        for r in rows:
            w=weights(r['qubits'],r['blocks'])
            assert hashlib.sha256(np.array(w,dtype=np.float64).tobytes()).hexdigest()==r['weights_sha256']
            assert len(r['seconds'])==7 and len(r['warmup_seconds'])==2
            assert all(t>0 and np.isfinite(t) for t in r['seconds'])
            assert float(np.median(r['seconds']))==r['median_seconds']
            values=[r['first_value']]+r['warmup_values']+r['values']
            assert np.ptp(values)<1e-5 and np.isfinite(values).all()
            if r['qubits']<=8 and r['blocks']<=8:
                ref=reference(r['qubits'],r['blocks'],w)[0]['global']
                assert max(abs(np.array(values)-ref))<1e-5
        all_records[backend]={(r['qubits'],r['blocks']):r for r in rows}
    for comparison in read('comparison'):
        key=comparison['qubits'],comparison['blocks'];cpu=all_records['qpp-cpu'][key];gpu=all_records[comparison['gpu_backend']][key]
        assert max(abs(np.array(cpu['values'])-gpu['values']))<1e-5
        assert cpu['median_seconds']/gpu['median_seconds']==comparison['cpu_median_over_gpu_median']
    print('Audit passed: 24 cases, 240 observe results, timing statistics, weights and CPU/GPU output agreement.')
if __name__=='__main__':main()
