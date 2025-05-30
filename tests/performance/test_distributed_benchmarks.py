import pytest
import numpy as np

@pytest.mark.benchmark(group="distributed")
def test_simulated_allreduce_benchmark(benchmark):
    # Simulate an allreduce operation (sum across arrays)
    n_workers = 8
    arr_size = 1024 * 1024  # 1M elements
    arrays = [np.random.rand(arr_size) for _ in range(n_workers)]

    def allreduce():
        # Simulate sum-reduce across all workers
        result = np.sum(arrays, axis=0)
        return result

    benchmark(allreduce) 