import argparse
from typing import Tuple
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.distributed import Backend
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import torchvision.models as models

def main(rank: int,
         epochs: int,
         model: nn.Module,
         input,
         label) -> nn.Module:
    device = torch.device(f'cuda:{rank}')
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    # initialize optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss = nn.CrossEntropyLoss()

    # train the model
    for i in range(epochs):
        model.train()
        epoch_loss = 0
        # train the model for one epoch
        x = input.to(device, non_blocking=True)
        y = label.to(device, non_blocking=True)
        optimizer.zero_grad()
        y_hat = model(x)
        batch_loss = loss(y_hat, y)
        batch_loss.backward()
        optimizer.step()
        batch_loss_scalar = batch_loss.item()
    return model.module


class BenchmarkDDP(nn.Module):
    def __init__(self, rank, epochs, model, input, label, backend):
        super().__init__()
        self.rank = rank
        self.epochs = epochs
        self.model = model
        self.input = input
        self.label = label
        self.backend = backend
        torch.cuda.set_device(self.rank)
        torch.distributed.init_process_group(backend=self.backend, init_method='env://')
    
    def run(self):
        model = main(rank=self.rank, epochs=self.epochs, model=self.model, input=self.input, label=self.label)


if __name__ == '__main__':
    #USER need to modify this part --------
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    backend = 'nccl'
    batch_size = 128
    epochs = 10
    model=models.resnet50()
    input = torch.randn(32, 3, 224, 224)
    label = torch.randn(32, 1000)
    rank = args.local_rank
    # -------- END
    bench = BenchmarkDDP(rank, epochs, model, input, label, backend)
    bench.run()

