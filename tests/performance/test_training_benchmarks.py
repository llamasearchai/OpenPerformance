import pytest
import torch
import torch.nn as nn
import torch.optim as optim

@pytest.mark.benchmark(group="training")
def test_pytorch_training_benchmark(benchmark):
    # Simple model and data
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(torch.device("cpu"))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(64, 128)
    y = torch.randint(0, 10, (64,))

    def train_step():
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    benchmark(train_step) 