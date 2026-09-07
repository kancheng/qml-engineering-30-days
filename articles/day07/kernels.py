"""CUDA-Q kernels: allocation, typed inputs, host-controlled branches and loops."""

import cudaq


@cudaq.kernel
def single_rotation(theta: float):
    q = cudaq.qubit()
    ry(theta, q)
    mz(q)


@cudaq.kernel
def register_circuit(n: int, theta: float, entangle: bool, flip_last: bool, measure: bool):
    q = cudaq.qvector(n)
    ry(theta, q[0])
    if entangle:
        for i in range(1, n):
            x.ctrl(q[0], q[i])
    if flip_last:
        x(q[n - 1])
    if measure:
        for i in range(n):
            mz(q[i])
