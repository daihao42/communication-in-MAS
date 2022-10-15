#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNet(nn.Module):

    def __init__(self, observation_shape:tuple, num_actions:int, nonlinear=F.relu, hidden = 128):
        super(ActorNet, self).__init__()
        
        self.fc1 = nn.Linear(observation_shape[0], hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden,num_actions)
        self.nonlinear = nonlinear
        #self.out_fn = nn.Softmax(num_actions)

    def forward(self, inputs):
        x = self.nonlinear(self.fc1(inputs))
        x = self.nonlinear(self.fc2(x))
        x = self.fc3(x)
        #x = self.out_fn(self.fc3(x))
        #return torch.max(x,1)[1]
        return x

if __name__ == '__main__':
    import numpy as np
    net = ActorNet((5,),5)
    input_feature = np.array([[1,2,3,4,5]])
