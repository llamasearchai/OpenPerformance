import pytest
from mlperf.optimization.distributed import DistributedOptimizer, CommunicationConfig
from mlperf.utils.config import get_config

@pytest.fixture
def dist_optimizer():
    return DistributedOptimizer(
        config=CommunicationConfig(),
        framework="pytorch",
        profile_first_step=False
    )

def test_distributed_initialization(dist_optimizer):
    dist_optimizer.initialize(
        rank=0,
        local_rank=0,
        world_size=1,
        master_addr="localhost"
    )
    assert dist_optimizer.initialized

def test_optimize_data_parallel(dist_optimizer):
    batch_size, grad_steps = dist_optimizer.optimize_data_parallel(
        batch_size=32,
        model_size_gb=10,
        num_gpus=4
    )
    assert batch_size > 0
    assert grad_steps >= 1

def test_benchmark_allreduce(dist_optimizer):
    results = dist_optimizer.benchmark_allreduce(100)
    assert "bandwidth_gb_s" in results
    assert results["bandwidth_gb_s"] > 0 