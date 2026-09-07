"""Run the three local modes in isolated processes, then generate the report."""
import os
from pathlib import Path
import subprocess
import sys
HERE = Path(__file__).resolve().parent

if __name__ == '__main__':
    env = dict(os.environ, OMP_NUM_THREADS='1', CUDAQ_DEFAULT_SIMULATOR='qpp')
    for mode in ('qpp-cpu', 'ionq-emulate', 'density-bitflip'):
        subprocess.run([sys.executable, str(HERE/'experiment.py'), '--mode', mode],
                       env=env, check=True)
    subprocess.run([sys.executable, str(HERE/'report.py')], env=env, check=True)
