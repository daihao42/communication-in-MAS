#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from utils.distributed import DistributedComm
from utils.model_parallel import ModelParallel

import torch

import torch.nn.functional as F

import torch.multiprocessing as mp

import torchvision
from torchvision import datasets, transforms

class TModel(torch.nn.Module):
    def __init__(self):
        super(TModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, kernel_size=(3,3), stride=3)
        #self.conv2 = torch.nn.Conv2d(3, 1, kernel_size=(3,3))
        self.flatten = torch.nn.Flatten(1,-1)
        #self.fc = torch.nn.Linear(625, 10)
        self.fc = torch.nn.Linear(972, 10)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.conv2(x)
        x = self.flatten(x)
        return self.fc(x)

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

def _sink_and_recv_tensor(dist_comm, world_size):
    dist_comm.process_wait()
    device = "cuda:0"
    #device = "cpu"
    tmodel = TModel()
    p_tmodel = ModelParallel(tmodel, device)
    p_shape = [x.shape for x in tmodel.parameters()]

    # fedavg update
    dst_ranks = list(range(world_size))
    dst_ranks.remove(dist_comm._rank)

    for epoch in range(8):

        ps = p_tmodel._parameter_decode()
        #print(f"{dist_comm._rank} before sink {p_shape} size msg to {dst_ranks}")

        # !!!!! " hds = " is extremely necessary, otherwise the processes will block!!
        hds = dist_comm.sink_p2p_message_batch_async(dst_ranks, ps)

        dist_comm.process_wait()

        #print(f"{dist_comm._rank} after sink ")

        res_comm_group =  dist_comm.get_p2p_comm_group()

        #print(f"rank {dist_comm._rank} read p2p group {res_comm_group}")

        msgs,handle = dist_comm.read_p2p_message_batch_async(per_msg_size=len(p_shape),
                                               per_msg_shape=p_shape)

        #print(f"{dist_comm._rank} : ",msgs[0][0])

        dist_comm.process_wait()

def rtest_sink_and_recv_tensor():
    _multi_processes_wrapper(world_size=8, func = _sink_and_recv_tensor)

def _sink_and_recv_parameters(dist_comm, world_size):
    dist_comm.process_wait()
    device = "cuda:0"
    #device = "cpu"
    tmodel = TModel()
    p_tmodel = ModelParallel(tmodel, device)
    p_shape = [x.shape for x in tmodel.parameters()]

    # fedavg update
    dst_ranks = list(range(world_size))
    dst_ranks.remove(dist_comm._rank)

    for epoch in range(8):

        # !!!!! " hds = " is extremely necessary, otherwise the processes will block!!
        hds = p_tmodel.sink_parameter(dist_comm, dst_ranks)

        dist_comm.process_wait()

        #msgs,handle = dist_comm.read_p2p_message_batch_async(per_msg_size=len(p_shape),
        #                                       per_msg_shape=p_shape)

        msgs,handles = p_tmodel.recv_parameter(dist_comm=dist_comm)

        print(f"{dist_comm._rank} : ",msgs[0][0])

        dist_comm.process_wait()

def rtest_sink_and_recv_parameters():
    _multi_processes_wrapper(world_size=8, func = _sink_and_recv_parameters)


def _fed_avg(dist_comm, world_size):
    dist_comm.process_wait()
    device = "cuda:0"
    #device = "cpu"
    tmodel = TModel()
    p_tmodel = ModelParallel(tmodel, device)
    p_shape = [x.shape for x in tmodel.parameters()]

    # fedavg update
    dst_ranks = list(range(world_size))
    dst_ranks.remove(dist_comm._rank)

    for epoch in range(8):

        msgs,hds = p_tmodel.fed_avg(dist_comm=dist_comm,dst_ranks=dst_ranks)

        print(msgs[0][0])

    #dist_comm.process_wait()

def rtest_fed_avg():
    _multi_processes_wrapper(world_size=4, func = _fed_avg)

def train(dist_comm, world_size):

    seed = dist_comm._rank
    # set random seed for CPU
    torch.manual_seed(seed)      
    # set random seed for current GPU
    torch.cuda.manual_seed(seed)   

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
    resnet18 = models.resnet18(pretrained=False)

    from torch.utils.tensorboard import SummaryWriter       
    writer = SummaryWriter("./logs/resnet18/")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    resnet18.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    resnet18.fc = torch.nn.Linear(in_features=512,out_features=10,bias=True)

    resnet18.to(device)

    #tmodel = TModel()
    tmodel = resnet18

    tmodel.to(device)

    batch_size = 8

    trainloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size,shuffle=True)

    #损失函数:这里用交叉熵
    criterion = torch.nn.CrossEntropyLoss()

    #优化器 这里用SGD
    optimizer = torch.optim.SGD(tmodel.parameters(),lr=1e-3, momentum=0.9)

    num_epochs = 10 #训练次数

    p_tmodel = ModelParallel(tmodel, device)

    for epoch in range(num_epochs):

        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = p_tmodel.model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()

            #for v in alexnet.parameters():
            #    print(v.grad)

            #print(inputs.grad)
            #print( inputs.grad.shape)

            optimizer.step()

            writer.add_scalar(f"Train/Loss-{dist_comm._rank}\\", loss.item(), epoch)
            writer.flush()

        print('[%d, %5d] loss:%.4f'%(epoch+1, (i+1)*batch_size, loss.item()))

        # fedavg update
        dst_ranks = list(range(world_size))
        dst_ranks.remove(dist_comm._rank)
        p_tmodel.fed_avg(dist_comm=dist_comm, dst_ranks=dst_ranks)


def test_train():
    _multi_processes_wrapper(world_size=4, func = train)


if __name__ == '__main__':
    pytest.main(["-s","-v","test_distributed.py"])


