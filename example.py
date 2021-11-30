import os
import torch
import logging
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.nn.parallel import DistributedDataParallel as DDP
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'


class BenchmarkDDP(nn.Module):
    def __init__(self, backend, input, label, loss_fn, optim, lr, steps, model):
        super().__init__()
        self.backend = backend
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim
        self.lr = lr
        self.steps = steps
        self.input = input
        self.label = label
    
    def wrap_model(self, rank, world_size):
        dist.init_process_group(self.backend, rank=rank, world_size=world_size)
        model = self.model.to(rank)
        ddp_model = DDP(model)
        optimizer = self.optim(ddp_model.parameters(), lr=self.lr)
        for i in range(self.steps):
            outputs = ddp_model(self.input.to(rank))
            labels = self.label.to(rank)
            loss_fn = self.loss_fn
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

    def run(self):
        n_gpus = torch.cuda.device_count()
        assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
        world_size = n_gpus
        mp.spawn(self.wrap_model,
                args=(world_size,),
                nprocs=world_size,
                join=True)

def main():
    #USER need to modify this part --------
    my_model = models.resnet50()
    for param in my_model.parameters():
        param.requires_grad = True
    loss_fn = nn.CrossEntropyLoss()
    backend = 'nccl'
    optim = torch.optim.SGD
    lr = 0.001
    steps = 100
    input = torch.randn(32, 3, 224, 224)
    label = torch.randn(32, 1000)
    # -------- END
    bench = BenchmarkDDP(backend, input, label, loss_fn, optim, lr, steps, model=my_model)
    bench.run()


if __name__=="__main__":
    main()
