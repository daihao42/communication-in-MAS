#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable

import torch                                                                                 
                                                                                             
                                                                                             
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
                                                                                             

class MyAlgorithm():

    def __init__(self, env, learning_rate, observation_shape, num_actions, num_agents, rank, device) -> None:

        self.observation_shape = observation_shape

        self.num_actions = num_actions
                                      
        self.num_agents = num_agents
                                      
        self.train_step = 0

        self.env = env

        self.lr = learning_rate

        self.device = th.device(f'cuda:{rank}' if th.cuda.is_available() else 'cpu')

        self.actor = DNet(observation_shape, num_actions)
        self.actor.to(self.device)

        self.critic = DNet(observation_shape, 1)
        self.critic.to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=learning_rate)
        self.actor_loss_func = F.cross_entropy

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=learning_rate)
        self.critic_loss_func = torch.nn.MSELoss()

    def choose_action(self, obs, greedy=False):
        x = th.tensor(obs, dtype = th.float).to(self.device)     
        _logits = self.actor(x)
        _probs = th.softmax(_logits, dim=1)
        if(greedy):                                                                   
            #return th.argmax(_probs,dim=1,keepdim=True).cpu().detach().numpy(), _probs.cpu().detach().numpy()
            return th.argmax(_probs,dim=1,keepdim=True), _probs.cpu().detach().numpy()
        #return th.multinomial(_probs, num_samples=1).reshape(-1)#.cpu().detach().numpy(), _probs.cpu().detach().numpy()
        return th.multinomial(_probs, num_samples=1).reshape(-1), _probs.cpu().detach().numpy()

    def train(self, sample, gamma = 0.99):

        n_obs = []
        n_action = []
        n_next_obs = []
        n_rew = []
        n_probs = []

        for (obs, action, _o_probs, reward, next_obs) in sample:
            n_obs.append(obs)
            n_action.append(action)
            n_next_obs.append(next_obs)
            n_rew.append(reward)
            n_probs.append(_o_probs)

        # for simple_spread
        #print(n_rew)

        discounted_ep_rs = torch.Tensor(self._discount_and_norm_rewards(n_rew, gamma)).reshape((-1,1)).to(self.device)
        #print(discounted_ep_rs)

        ### update critic ###
        value = self.critic(torch.Tensor(n_obs).reshape((-1,self.observation_shape[0])).to(self.device))
        next_value = self.critic(torch.Tensor(n_next_obs).reshape((-1,self.observation_shape[0])).to(self.device))

        critic_loss = self.critic_loss_func(discounted_ep_rs + gamma * value, next_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #print("value", value)
        #print("next_value", next_value)
        #print("discounted_ep_rs", discounted_ep_rs)

        with torch.no_grad():
            td_error = discounted_ep_rs + gamma * next_value - value

        ### update actor ###
        #new_action = self.actor(torch.Tensor(n_obs).reshape(-1,self.observation_shape[0])).reshape(-1)
        new_action, new_probs = self.choose_action(torch.Tensor(n_obs).reshape(-1,self.observation_shape[0]))
        new_probs = torch.Tensor(new_probs)
        #print("new_probs",new_probs)
        #print("n_probs", n_probs)
        actor_log_prob = self.actor_loss_func(input=new_probs.reshape((-1,self.num_actions)), target=torch.Tensor(n_probs).reshape((-1,self.num_actions)))
        #print("actor_log_prob",actor_log_prob)
        #print("td-error",td_error)

        actor_loss = -torch.mean(actor_log_prob * td_error)

        actor_loss.requires_grad = True

        self.actor_optimizer.zero_grad()
        #actor_loss.backward(torch.ones_like(actor_loss))
        actor_loss.backward()
        self.actor_optimizer.step()
        #print("actor_loss",actor_loss)
        '''

        _logits = self.actor(torch.Tensor(n_obs).reshape(-1,self.observation_shape[0])).reshape(-1)
        prediction = Categorical(F.softmax(_logits))                                   

        print("dicounted reward:",discounted_ep_rs)
        #print("logits:",_logits)
        #print("prediction:",prediction)
                                                                               
        ploss = prediction.log_prob(torch.Tensor(n_action).reshape((-1,1)).to(self.device)) # 

        #print("ploss:",ploss)
                                                                               
        actor_loss = th.mean(ploss * th.FloatTensor(discounted_ep_rs).to(self.device)) 

        self.actor_optimizer.zero_grad()
        actor_loss.backward(torch.ones_like(actor_loss))
        self.actor_optimizer.step()
        print("actor_loss",actor_loss)
        '''

        return critic_loss, actor_loss


    def _discount_and_norm_rewards(self, ep_rewards, gamma=0.99):
        discounted_ep_rs = np.zeros_like(ep_rewards)
        running_add = 0
        for t in reversed(range(0,len(ep_rewards))):
            running_add = running_add * gamma + ep_rewards[t]
            discounted_ep_rs[t] = running_add
        #discounted_ep_rs -= np.mean(discounted_ep_rs)
        #discounted_ep_rs /= np.std(discounted_ep_rs)
        #return list(map(lambda x:np.repeat(x,self.num_agents),discounted_ep_rs))
        return discounted_ep_rs 


