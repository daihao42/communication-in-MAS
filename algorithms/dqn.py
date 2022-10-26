#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import deque
import numpy as np
import torch as th
import torch.nn as nn

import random

from torch.autograd import Variable


__author__ = 'dai'

class DNet(nn.Module):                                                                   
                                                                                             
    def __init__(self, observation_shape:tuple, output_size:int,
				 nonlinear=F.relu, hidden = 128):
        super(DNet, self).__init__()                                                     
                                                                                             
        self.fc1 = nn.Linear(observation_shape[0], hidden)                                   
        self.fc2 = nn.Linear(hidden, hidden)                                                 
        self.fc3 = nn.Linear(hidden,output_size)                                             
        self.nonlinear = nonlinear                                                           
        #self.out_fn = nn.Softmax(num_actions)                                               
                                                                                             
    def forward(self, inputs):                                                               
        x = self.nonlinear(self.fc1(inputs))                                                 
        x = self.nonlinear(self.fc2(x))                                                      
        x = self.fc3(x)                                                                      
        #x = self.out_fn(self.fc3(x))                                                        
        #return torch.max(x,1)[1]                                                            
        return x                                                                             
 

class DQN():
    def __init__(self, env, initial_epsilon, epsilon_decremental, memory_capacity, target_replace_iter, learning_rate, observation_shape, num_actions, num_agents, logger) -> None:

        self.num_actions = num_actions

        self.num_agents = num_agents

        self.logger = logger

        self.eval_net = DNet(observation_shape, num_actions)
        self.target_net = DNet(observation_shape, num_actions)
        
        # epsilon greedy
        self.epsilon = initial_epsilon

        self.epsilon_decremental = epsilon_decremental

        # update iteration
        self.target_replace_iter = target_replace_iter

        self.target_replace_iter_count = 0

        # learning rate
        self.learning_rate = learning_rate

        # memory queue capacity
        self.memory_capacity = memory_capacity
        self.memory = deque()
        self.memory_counter = 0

        # GPU training
        self.loss = []
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.eval_net.to(self.device)
        self.target_net.to(self.device)
        self.optimizer = th.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

    def store_transition(self, s, a, r, s_):
        self.memory_counter = self.memory_counter + 1
        transition = self.memory.append((s,a,r,s_))
        # pop old data
        if(len(self.memory) > self.memory_capacity):
            self.memory.popleft()  

    def choose_action(self,x, exploration = True):
        if not exploration:
            x = th.tensor(x, dtype = th.float).to(self.device)
            return self.eval_net(x).cpu().detach().numpy()

        if np.random.uniform() >= self.epsilon:
            #print("------ dqn action ------")
            x = th.tensor(x, dtype = th.float).to(self.device)
            return self.eval_net(x).cpu().detach().numpy()
        else:
            #print("------ random action ------")
            return np.random.rand(1,self.num_actions)[0]
    
    def replace_parameters(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def learn(self, gamma = 0.9, batch_size = 64):

        # sample data
        if(len(self.memory) < batch_size):
            batch_size = len(self.memory)
        b_memory = random.sample(self.memory,batch_size)

        b_s = th.FloatTensor([x[0] for x in b_memory])
        b_a = th.LongTensor([x[1] for x in b_memory])
        b_r = th.FloatTensor([x[2] for x in b_memory])
        #b_r = th.FloatTensor([np.repeat(x[2],self.num_agents) for x in b_memory]).reshape(-1))
        b_s_ = th.FloatTensor([x[3] for x in b_memory])

        # train
        #q_eval = self.eval_net(b_s.to(self.device)).max(1)[0] # shape (batch, 1)
        #q_next = self.target_net(b_s_.to(self.device)).detach().max(1)[0] # detach from graph, don't backpropagate

        q_eval = self.eval_net(b_s.to(self.device)).reshape(-1,self.num_agents,int(self.num_actions/self.num_agents)).max(2)[0] # shape (batch, 1)
        q_next = self.target_net(b_s_.to(self.device)).detach().reshape(-1,self.num_agents,int(self.num_actions/self.num_agents)).max(2)[0] # detach from graph, don't backpropagate
        q_target = b_r.to(self.device) + gamma * q_next   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        
        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.target_replace_iter_count += 1

        self.logger.add_scalar('Global/Loss\\', loss.detach().cpu().numpy(),self.target_replace_iter_count)
        self.logger.add_scalar('Global/Epsilon\\', self.epsilon, self.target_replace_iter_count)
        #self.loss.append(loss.detach().cpu().numpy())

        if(self.target_replace_iter_count % self.target_replace_iter == 0):
            self.replace_parameters()
            self.epsilon -= self.epsilon_decremental
            #print("---- replace_parameters and decrease epsilon to {} !!".format(self.epsilon))

    def saveModel(self, path):
        th.save(self.eval_net,path)

