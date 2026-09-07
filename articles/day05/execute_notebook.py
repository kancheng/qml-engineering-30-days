"""Execute the tutorial with this Python interpreter, without global registration."""

import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    source = ROOT / "notebooks" / "day05_quantum_fundamentals.ipynb"
    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)
    with tempfile.TemporaryDirectory(prefix="day05-notebook-") as temp:
        kernel_dir = Path(temp) / "day05"
        kernel_dir.mkdir()
        (kernel_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Day 05 project Python", "language": "python",
        }), encoding="utf-8")
        manager = AsyncKernelManager(kernel_name="day05",
                                kernel_spec_manager=KernelSpecManager(kernel_dirs=[temp]))
        client = NotebookClient(notebook, km=manager, timeout=180, startup_timeout=60,
                                resources={"metadata": {"path": str(ROOT)}})
        # All Jupyter/IPython transient files stay in the temporary directory.
        env = dict(os.environ, IPYTHONDIR=temp, JUPYTER_RUNTIME_DIR=temp)
        # Two-qubit demos benefit from avoiding large OpenMP thread pools.
        env.setdefault("OMP_NUM_THREADS", "1")
        client.execute(env=env, cleanup_kc=True)
    nbformat.write(notebook, source)
    print(f"Executed {sum(c.cell_type == 'code' for c in notebook.cells)} code cells: {source}")


if __name__ == "__main__":
    main()
