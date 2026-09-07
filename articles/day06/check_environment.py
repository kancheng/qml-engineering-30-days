"""Record installed packages, host tools and isolated CUDA-Q execution probes."""

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def text_output(value) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")


def run_command(command: list[str], timeout: int = 60, env: dict | None = None) -> dict:
    started = time.perf_counter()
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=env)
        report = {"status": "ok" if result.returncode == 0 else "failed",
                  "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except FileNotFoundError as error:
        report = {"status": "missing", "returncode": None, "stdout": "", "stderr": str(error)}
    except subprocess.TimeoutExpired as error:
        report = {"status": "timeout", "returncode": None,
                  "stdout": text_output(error.stdout), "stderr": text_output(error.stderr)}
    except OSError as error:
        report = {"status": "failed", "returncode": None, "stdout": "", "stderr": str(error)}
    return {"command": command, **report, "elapsed_seconds": time.perf_counter() - started}


def probe_passed(process: dict, payload: dict | None, backend: str) -> bool:
    return (process["status"] == "ok" and process["returncode"] == 0
            and isinstance(payload, dict) and payload.get("passed") is True
            and payload.get("backend") == backend and payload.get("target") == backend
            and bool(payload.get("checks")) and all(v is True for v in payload["checks"].values()))


def inspect_environment(backends: list[str], timeout: int = 60) -> dict:
    if not backends or any(b not in ("qpp-cpu", "nvidia") for b in backends):
        raise ValueError("request at least one supported backend")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    packages = {d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"]}
    tools = {
        "nvidia_smi_version": run_command(["nvidia-smi", "--version"], timeout),
        "gpu_inventory": run_command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
                                      "--format=csv,noheader,nounits"], timeout),
        "nvcc": run_command(["nvcc", "--version"], timeout),
        "pip_check": run_command([sys.executable, "-m", "pip", "check"], timeout),
    }
    probes = {}
    for backend in dict.fromkeys(backends):
        # Native initialization failures and shutdown messages stay in the child.
        with tempfile.TemporaryDirectory(prefix="day06-probe-") as temp:
            output = Path(temp) / "smoke.json"
            process = run_command([sys.executable, str(Path(__file__).with_name("hello_quantum.py")),
                                   "--backend", backend, "--output", str(output)], timeout, env)
            payload = None
            if output.exists():
                try:
                    payload = json.loads(output.read_text())
                except (json.JSONDecodeError, OSError) as error:
                    process["payload_error"] = str(error)
            probes[backend] = {"process": process, "result": payload,
                               "passed": probe_passed(process, payload, backend)}
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {"executable": sys.executable, "version": platform.python_version(),
                   "prefix": sys.prefix, "base_prefix": sys.base_prefix,
                   "is_venv": sys.prefix != sys.base_prefix},
        "platform": platform.platform(), "os_release": platform.freedesktop_os_release(),
        "packages": dict(sorted(packages.items())), "tools": tools, "backends": probes,
        "probe_environment": {"OMP_NUM_THREADS": env["OMP_NUM_THREADS"]},
        "passed": tools["pip_check"]["status"] == "ok" and all(p["passed"] for p in probes.values()),
        "scope": "pip check and explicitly requested backend smoke tests; optional host tools recorded separately",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", choices=("qpp-cpu", "nvidia"), default=["qpp-cpu", "nvidia"])
    parser.add_argument("--timeout", type=int, default=60, help="per command timeout in seconds")
    parser.add_argument("--output", type=Path, default=ROOT / "results/day06/environment.json")
    args = parser.parse_args()
    report = inspect_environment(args.backends, args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for backend, probe in report["backends"].items():
        print(f"{backend}: {'PASS' if probe['passed'] else 'FAIL'}")
    print(f"report: {args.output}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
