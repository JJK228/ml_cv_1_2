import statistics
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def prepare_data() -> TensorDataset:
    X = torch.randn(10000, 128)
    y = torch.randint(0, 2, (10000,))
    dataset = TensorDataset(X, y)
    return dataset


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    dataloader = DataLoader(
        prepare_data(),
        batch_size=256,
        shuffle=True,
        pin_memory=use_cuda,
        num_workers=2 if use_cuda else 0,
        persistent_workers=use_cuda,
    )

    model = nn.Sequential(
        nn.Linear(128, 512), nn.ReLU(),
        nn.Linear(512, 128), nn.ReLU(),
        nn.Linear(128, 2)
    ).to(device).train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    losses_history = []
    forward_times = []
    backward_times = []

    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        data = data + torch.randn_like(data)

        optimizer.zero_grad(set_to_none=True)

        if use_cuda:
            forward_start = torch.cuda.Event(enable_timing=True)
            forward_end = torch.cuda.Event(enable_timing=True)
            backward_start = torch.cuda.Event(enable_timing=True)
            backward_end = torch.cuda.Event(enable_timing=True)
            forward_start.record()
        else:
            forward_start = time.perf_counter()

        output = model(data)
        loss = criterion(output, target)

        if use_cuda:
            forward_end.record()
            backward_start.record()
        else:
            forward_times.append(time.perf_counter() - forward_start)
            backward_start = time.perf_counter()

        loss.backward()

        if use_cuda:
            backward_end.record()
        else:
            backward_times.append(time.perf_counter() - backward_start)

        optimizer.step()

        if use_cuda:
            torch.cuda.synchronize()
            forward_times.append(forward_start.elapsed_time(forward_end) / 1000)
            backward_times.append(backward_start.elapsed_time(backward_end) / 1000)

        losses_history.append(loss.detach())
        if batch_idx % 20 == 0:
            loss_value = loss.detach().item()
            print(f"Batch {batch_idx} loss: {loss_value:.4f}")

    print(f"Epoch finished, avg forward time is {statistics.mean(forward_times)}, "
          f"avg backward time is {statistics.mean(backward_times)}")

if __name__ == '__main__':
    train()
