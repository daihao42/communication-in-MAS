#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import deque
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import random

from torch.autograd import Variable

from models.ActorNet import ActorNet

class CommModel(nn.Module):

    def __init__(self, env,  observation_shape, num_actions, num_agents, hidden_shape, device) -> None:

        self.num_actions = num_actions

        self.num_agents = num_agents

        self.hidden_shape = hidden_shape

        super(CommModel,self).__init__()

        self.encode_net = self._create_encoder(observation_shape=observation_shape[0], hidden=self.hidden_shape)

        self.action_net = self._create_action_net(hidden_shape=hidden_shape, num_actions=num_actions)

        self.matrix = self._create_comm_matrix().to(device)

    def _create_encoder(self, observation_shape, hidden):

        return nn.Linear(observation_shape, hidden)

    def _create_comm_matrix(self):
        hi = Variable(th.Tensor(1), requires_grad = True)
        ci = Variable(th.Tensor(1), requires_grad = True)
        Variable(th.Tensor(1), requires_grad = True)
        return th.zeros(self.num_agents, self.num_agents) + ci/(self.num_agents - 1) - th.diag(th.repeat_interleave(ci/(self.num_agents - 1),self.num_agents)) + th.diag(th.repeat_interleave(hi,self.num_agents))

    def _create_action_net(self, hidden_shape, num_actions):
        return nn.Linear(hidden_shape, num_actions)

    def forward(self, x):

        x = th.stack([self.encode_net(i) for i in x])

        x = th.mm(self.matrix, x)

        return [self.action_net(i) for i in x]

class CommNet():

    def __init__(self, env, learning_rate, observation_shape, num_actions, num_agents) -> None:

        self.num_actions = num_actions

        self.num_agents = num_agents

        self.train_step = 0

        self.hidden_shape = 128

        # GPU training
        self.loss = []

        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

        self.commModel = CommModel(env,observation_shape,num_actions,num_agents,self.hidden_shape, self.device)

        self.commModel.to(self.device)

        self.optimizer = th.optim.Adam(self.commModel.parameters(), lr=learning_rate)

        self.clear_transition()

    def store_transition(self, s, a, r):
        self.ep_observations.append(s)
        self.ep_actions.append(a)
        self.ep_rewards.append(r)

    def clear_transition(self):
        self.ep_observations = []
        self.ep_actions = []
        self.ep_rewards = []

    def choose_action(self,x, greedy=False):
        #x = th.tensor(np.array(x), dtype = th.float).to(self.device)
        x = x.to(self.device)
        _logits = th.stack(self.commModel(x))
        _probs = th.softmax(_logits, dim=1)
        #return _probs.cpu().detach().numpy()
        return th.max(_probs,dim=1)[1].cpu().detach().numpy()

    def _discount_and_norm_rewards(self, gamma):
        discounted_ep_rs = np.zeros_like(self.ep_rewards)
        running_add = 0
        for t in reversed(range(0,len(self.ep_rewards))):
            running_add = running_add * gamma + self.ep_rewards[t]
            discounted_ep_rs[t] = running_add
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        #return list(map(lambda x:np.repeat(x,self.num_agents),discounted_ep_rs))
        return discounted_ep_rs

    def learn(self, gamma):

        # prediction
        _logits = [th.max(th.stack(self.commModel(th.tensor(x, dtype=th.float).to(self.device))), dim=1)[1].float() for x in self.ep_observations]

        # cross-entropy basicly include the one-hot encode function, tensor([[0.1,0.5,0.4],[0.2,0.6,0.2]]) v.s. tensor([1,1])

        #prediction = th.stack([Categorical(F.softmax(_logit)) for _logit in _logits])
        #prediction = th.stack([F.softmax(_logit) for _logit in _logits])
        prediction = th.stack(_logits)

        ploss = -th.binary_cross_entropy_with_logits(prediction, th.tensor(self.ep_actions, dtype=th.float).to(self.device)) # "-" because it was built to work with gradient descent, but we are using gradient ascent

        print(ploss)

        discounted_ep_rs = self._discount_and_norm_rewards(gamma=gamma)

        print(discounted_ep_rs)

        pseudo_loss = th.sum(ploss * th.FloatTensor(discounted_ep_rs).to(self.device))

        pseudo_loss.requires_grad_(True)

        # back propagation
        self.optimizer.zero_grad()
        pseudo_loss.backward()
        self.optimizer.step()

        self.train_step += 1


    def saveModel(self, path):
        th.save(self.commModel, path)

