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
                                                                                             
    def __init__(self, observation_shape, output_size:int,
				 nonlinear=F.relu, hidden = 128):
        super(DNet, self).__init__()                                                     
                                                                                             
        self.fc1 = nn.Linear(observation_shape, hidden)                                   
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
                                                                                             

class NOAC():

    def __init__(self, env, learning_rate, observation_shape, num_actions, num_agents, rank, device, neighbors = 3, message_size=12) -> None:

        self.observation_shape = observation_shape[0]

        self.message_size = message_size

        self.num_actions = num_actions
                                      
        self.num_agents = num_agents
                                      
        self.train_step = 0

        self.env = env

        self.lr = learning_rate

        self.device = th.device(f'cuda:{rank}' if th.cuda.is_available() else 'cpu')

        self.actor = DNet(observation_shape=self.observation_shape+(message_size+num_actions)*(neighbors+1),
                          output_size= num_actions)
        self.actor.to(self.device)

        self.critic = DNet(observation_shape=self.observation_shape+(message_size+num_actions)*(neighbors+1), output_size=1)
        self.critic.to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=learning_rate)
        self.actor_loss_func = F.cross_entropy

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=learning_rate)
        self.critic_loss_func = torch.nn.MSELoss()

        self.encoder = DNet(observation_shape=self.observation_shape, output_size=message_size+num_actions)
        self.encoder.to(self.device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(),lr=learning_rate)


    def _encoder(self, obs):
        x = th.tensor(obs, dtype = th.float).to(self.device)     
        return self.encoder(x)

    def _kl(self,x,y):
        kl = F.kl_div(x.softmax(dim=-1).log(), y.softmax(dim=-1), reduction='sum')
        return kl

    def choose_action(self, obs, msg, neighbors_msgs, greedy=False):
        x = th.tensor(obs, dtype = th.float).to(self.device)     
        neighbors_msgs = torch.concat(neighbors_msgs)
        x = torch.concat([x, msg, neighbors_msgs])
        _logits = self.actor(x)
        _probs = th.softmax(_logits, dim=0)

        if(greedy):                                                                   
            #return th.argmax(_probs,dim=1,keepdim=True).cpu().detach().numpy(), _probs.cpu().detach().numpy()
            return th.argmax(_probs,dim=0,keepdim=True), _probs.cpu().detach().numpy()
        #return th.multinomial(_probs, num_samples=1).reshape(-1)#.cpu().detach().numpy(), _probs.cpu().detach().numpy()
        return th.multinomial(_probs, num_samples=1).reshape(-1), _probs.cpu().detach().numpy()

    def train(self, sample, gamma = 0.99):

        n_obs = []
        n_action = []
        n_next_obs = []
        n_rew = []
        n_probs = []
        n_msg = []
        n_next_msg = []

        n_neighbor_msgs = []
        n_next_neighbor_msgs = []

        for (obs, action, _o_probs, msg, neighbor_msgs, reward, next_obs, next_msg, next_neighbor_msgs) in sample:
            n_obs.append(obs)
            n_action.append(action)
            n_next_obs.append(next_obs)
            n_rew.append(reward)
            n_probs.append(_o_probs)
            n_msg.append(msg)
            n_next_msg.append(next_msg)
            n_neighbor_msgs.append(neighbor_msgs)
            n_next_neighbor_msgs.append(next_neighbor_msgs)

        # for simple_spread
        #print(n_rew)

        discounted_ep_rs = torch.Tensor(self._discount_and_norm_rewards(n_rew, gamma)).reshape((-1,1)).to(self.device)
        #print(discounted_ep_rs)

        ### update critic ###
        critic_input = []
        for i,obs in enumerate(n_obs):
            #print(f"msg_shape: {n_msg[i].detach().cpu().numpy().shape}")
            l = np.concatenate([obs , n_msg[i].detach().cpu().numpy()])
            for neighbor_msg in n_neighbor_msgs[i]:
                l = np.concatenate([l , neighbor_msg.detach().cpu().numpy()]) 
            critic_input.append(l)
        critic_input = torch.Tensor(critic_input).to(self.device)
        value = self.critic(critic_input)

        next_critic_input = []
        for i,obs in enumerate(n_next_obs):
            l = np.concatenate([obs , n_next_msg[i].detach().cpu().numpy() ])
            for neighbor_msg in n_next_neighbor_msgs[i]:
                l = np.concatenate([l , neighbor_msg.detach().cpu().numpy() ])
            next_critic_input.append(l)
        next_critic_input = torch.Tensor(next_critic_input).to(self.device)

        next_value = self.critic(next_critic_input)

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
        new_action = []
        new_probs = []
        for i,obs in enumerate(n_obs):
            b_action, b_probs = self.choose_action(n_obs[i],n_msg[i], n_neighbor_msgs[i])
            new_probs.append(b_probs)
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

        encoder_loss = []

        for i,obs in enumerate(n_obs):
            new_msg = self._encoder(obs)
            #la_kl = self._kl(new_msg[-self.num_actions:], torch.Tensor(n_probs[i]).to(self.device))
            #print(f"action_kl:{la_kl}")

            la_kl = torch.tensor(0)
            for neighbor_id in range(len(n_neighbor_msgs[i])):
                neighbor_msg_action = torch.cat([n_neighbor_msgs[i][neighbor_id][:self.message_size],torch.Tensor(n_probs[i]).to(self.device)]).detach()
                # no pesudo action
                # neighbor_msg_action = n_neighbor_msgs[i][neighbor_id].detach()
                # no encoder msg
                # neighbor_msg_action = torch.cat([new_msg[:self.message_size],torch.Tensor(n_probs[i]).to(self.device)]).detach()

                #print(f"neighbor_msg_action:{neighbor_msg_action}")

                la_kl = la_kl + self._kl(new_msg, neighbor_msg_action)
                # no encoder msg
                #break

            encoder_loss.append(la_kl.clone())
            self.encoder_optimizer.zero_grad()
            #encoder_loss = torch.cat(encoder_loss)
            la_kl.backward()
            self.encoder_optimizer.step()
        # no encoder pesudo 
        #encoder_loss.append(0)

        return critic_loss, actor_loss, encoder_loss[-1]


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


