import pytest
import torch
import torch.nn as nn

@pytest.mark.benchmark(group="inference")
def test_pytorch_inference_benchmark(benchmark):
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(torch.device("cpu"))
    x = torch.randn(64, 128)
    model.eval()

    def inference_step():
        with torch.no_grad():
            _ = model(x)

    benchmark(inference_step) 