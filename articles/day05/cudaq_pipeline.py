"""CUDA-Q execution stage; target selection belongs to the caller."""

import cudaq


@cudaq.kernel
def encoded_pair(theta: float, measure: bool):
    q = cudaq.qvector(2)
    ry(theta, q[0])
    x.ctrl(q[0], q[1])
    if measure:
        mz(q[0])
        mz(q[1])


def sample_counts(theta: float, shots: int, seed: int) -> dict[str, int]:
    cudaq.set_random_seed(seed)
    result = cudaq.sample(encoded_pair, theta, True, shots_count=shots,
                          explicit_measurements=True)
    return {label: int(count) for label, count in result.items()}
