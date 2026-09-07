"""Run backends sequentially in separate processes, then write comparison."""
import os,sys,time,subprocess
from benchmark import ROOT,PROTOCOL,write
if __name__=='__main__':
    env=os.environ.copy();env['OMP_NUM_THREADS']='1'
    write('protocol',PROTOCOL);runs=[]
    for backend in PROTOCOL['targets']:
        start=time.perf_counter()
        subprocess.run([sys.executable,str(ROOT/'articles/day27/benchmark.py'),'--backend',backend],env=env,check=True)
        runs.append({'backend':backend,'process_wall_seconds':time.perf_counter()-start})
    write('process_times',runs)
    subprocess.run([sys.executable,str(ROOT/'articles/day27/plot_results.py')],env=env,check=True)
