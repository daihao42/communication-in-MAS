#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from utils.distributed import DistributedComm

import torch

import torch.multiprocessing as mp

import torchvision
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                                transforms.ToPILImage(),
                                transforms.Resize((28*2,28*2)),
                                transforms.ToTensor(),
                              ])

data_train = datasets.MNIST(root = "./datasets/MNIST/",
                            transform=transform,
                            train = True,
                            download = True)

data_test = datasets.MNIST(root="./datasets/MNIST/",
                           transform = transform,
                           train = False)

import torchvision.models as models
alexnet = models.alexnet(pretrained=True)

from torch.utils.tensorboard import SummaryWriter       
writer = SummaryWriter("./logs/alexnet_logs/")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

alexnet.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(2, 2), stride=1, padding=(2, 2))

alexnet.classifier[6] = torch.nn.Linear(in_features=4096,out_features=10,bias=True)

trainloader = torch.utils.data.DataLoader(data_train, batch_size=64,shuffle=True)


def train(dist_comm, world_size):
    #损失函数:这里用交叉熵
    criterion = torch.nn.CrossEntropyLoss()

    #优化器 这里用SGD
    optimizer = torch.optim.SGD(alexnet.parameters(),lr=1e-3, momentum=0.9)

    num_epochs = 1 #训练次数

    alexnet.to(device)

    for epoch in range(num_epochs):
        running_loss = 0
        batch_size = 100

        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = alexnet(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()

            for v in alexnet.parameters():
                print(v.grad)

            print(inputs.grad)
            print( inputs.grad.shape)

            optimizer.step()

            writer.add_scalar('Train/Loss\\', loss.item(), epoch)
            writer.flush()

        print('[%d, %5d] loss:%.4f'%(epoch+1, (i+1)*100, loss.item()))


def _multi_processes_distributed_wrapper(master_ip, master_port, tcp_store_ip, tcp_store_port, world_size, rank, func):
    """
    run func on multiple local distributed commucation processes
    """
    dist_comm = DistributedComm(master_ip, master_port, tcp_store_ip, tcp_store_port, rank, world_size)
    func(dist_comm, world_size)

def _multi_processes_wrapper(world_size, func):
    """
    multiple processes managenment
    """
    #mp.set_start_method("spawn")
    mp.set_start_method('forkserver', force=True)
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=_multi_processes_distributed_wrapper, args=("127.0.0.1", "29700","127.0.0.1","29701", world_size, rank, func))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    #for p in processes:
    #    p.close()

def test_train():
    _multi_processes_wrapper(world_size=4, func = train)


if __name__ == '__main__':
    train(None,None)

