#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append(".")

from utils.log_utils import Logger

from utils.distributed import DistributedComm
from utils.model_parallel import ModelParallel

import torch

import torch.nn.functional as F

import torch.multiprocessing as mp

import numpy as np

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

def get_device():
    if torch.cuda.is_available():
        local_rank = int(os.environ["LOCAL_RANK"])
        device_id = local_rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        return f"cuda:{device_id}"
    else:
        return "cpu"

def train(dist_comm, world_size, logger:Logger):

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

    device = get_device()

    logger.device = device

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

    p_tmodel = ModelParallel(tmodel, device, dist_comm)

    first_flag = True

    losses = []

    for epoch in range(num_epochs):

        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            if first_flag:
                input_size = inputs.shape
                logger.model_summary(p_tmodel.model, input_size)
                logger.add_graph(p_tmodel.model, inputs.to(device))
                first_flag = False

            outputs = p_tmodel.model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()

            #for v in alexnet.parameters():
            #    print(v.grad)

            #print(inputs.grad)
            #print( inputs.grad.shape)

            optimizer.step()

            logger.add_scalar(f"Train/Loss/{epoch}", loss.item(), (i+1)*batch_size)
            losses.append(loss.item())

        logger.add_scalar(f"Train/GLoss", np.average(losses), epoch)
        logger.add_histogram(p_tmodel.model, epoch)
        logger.INFO(f'rank-{dist_comm._rank} : [%d, %5d] loss:%.4f'%(epoch+1, (i+1)*batch_size, np.average(losses)))
        losses = []

        # fedavg update
        dst_ranks = list(range(world_size))
        dst_ranks.remove(dist_comm._rank)
        p_tmodel.fed_avg(dst_ranks=dst_ranks)


if __name__ == '__main__':
    import sys
    import os
    dist_comm = DistributedComm.launch_init()
    world_size = int(os.environ["WORLD_SIZE"])
    logger = Logger("test_logs", "actor", dist_comm._rank)
    train(dist_comm, world_size, logger)
