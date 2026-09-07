"""Modify register allocation and circuit structure from the command line."""

import argparse
import json

import cudaq
import numpy as np

from experiment import run_case, validate
from kernels import register_circuit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("qpp-cpu", "nvidia"), default="qpp-cpu")
    parser.add_argument("--qubits", type=int, default=3)
    parser.add_argument("--theta", type=float, default=float(np.pi/2))
    parser.add_argument("--no-entangle", action="store_true")
    parser.add_argument("--flip-last", action="store_true")
    args = parser.parse_args()
    validate(args.qubits, args.theta, not args.no_entangle, args.flip_last)
    cudaq.set_target(args.backend)
    print("target:", cudaq.get_target().name)
    print(cudaq.draw(register_circuit, args.qubits, args.theta, not args.no_entangle, args.flip_last, True))
    result = run_case(args.qubits, args.theta, not args.no_entangle, args.flip_last)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
