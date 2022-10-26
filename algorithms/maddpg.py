#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from models.ddpgagent import DDPGAgent
from collections import deque
import random
import numpy as np

'''
DDPG -> MADDPG ??
'''
class MADDPG():
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, env, initial_epsilon, epsilon_decremental, memory_capacity, target_replace_iter, learning_rate, observation_shape, num_actions, num_agents, logger, discrete_action=True, tau=0.01) -> None:
        """
        Inputs:
        tau (float): Target update rate
        discrete_action (bool): Whether or not to use discrete action space
        """
        
        self.num_actions = num_actions

        self.num_agents = num_agents

        self.logger = logger

        self.discrete_action = discrete_action

        self.tau = tau

        self.niter = 0

        # epsilon greedy
        self.epsilon = initial_epsilon

        self.epsilon_decremental = epsilon_decremental

        self.agents = [DDPGAgent(lr=learning_rate,num_in_pol= observation_shape[0], num_out_pol=num_actions, num_in_critic=num_agents * (self.num_actions+observation_shape[0]), USE_CUDA=True)
                        for i in range(num_agents)]
        
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
        #self.loss = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.optimizer = th.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        #self.loss_func = nn.MSELoss()

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    """
    Take a step forward in environment with all agents
    Inputs:
	observations: List of observations for each agent
	explore (boolean): Whether or not to add exploration noise
    Outputs:
	actions: List of actions for each agent
    """
    def choose_action(self, observations, exploration=True):
        if not exploration:
            return [a.step(torch.tensor(obs, dtype = torch.float).to(self.device)) for a, obs in zip(self.agents, observations)]

        if np.random.uniform() >= self.epsilon:
            return [a.step(torch.tensor(obs, dtype = torch.float).to(self.device)) for a, obs in zip(self.agents, observations)]
        else:
            return [np.random.rand(1,self.num_actions)[0] for i in range(self.num_agents)]

    def update(self, sample, agent_i, gamma, parallel=False, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]

        curr_agent.critic_optimizer.zero_grad()

        #if self.discrete_action: # one-hot encode action
        #    all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
        #                    zip(self.target_policies, next_obs)]
        #else:
        all_trgt_acs = torch.cat([torch.cat([pi(ob) for ob,pi in zip(nobs,self.target_policies)]) for nobs in next_obs]).reshape(-1,self.num_actions*self.num_agents)
        trgt_vf_in = torch.cat((next_obs.reshape(next_obs.shape[0],-1), all_trgt_acs), dim=1)
   
        target_value = (rews[:,[agent_i]] + gamma *
                        curr_agent.target_critic(trgt_vf_in) *
                        (1 - dones[:,[agent_i]]))

        vf_in = torch.cat((obs.reshape(next_obs.shape[0],-1), acs.reshape(acs.shape[0],-1)), dim=1)
        actual_value = curr_agent.critic(vf_in)
        mse_loss_func = torch.nn.MSELoss()
        vf_loss = mse_loss_func(actual_value, target_value.detach())
        vf_loss.backward()
        if parallel:
            average_gradients(curr_agent.critic)
        torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        curr_agent.policy_optimizer.zero_grad()

        #if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            #curr_pol_out = curr_agent.policy(obs[agent_i])
            #curr_pol_vf_in = gumbel_softmax(curr_pol_out.cpu(), hard=True)
        #else:
        curr_pol_out = curr_agent.policy(obs[:,[agent_i]])

        all_pol_acs = torch.cat([torch.cat([pi(ob) for ob,pi in zip(sobs,self.policies)]) for sobs in obs]).reshape(-1,self.num_actions*self.num_agents)
        vf_in = torch.cat((obs.reshape(obs.shape[0],-1), all_pol_acs), dim=1)

        pol_loss = -curr_agent.critic(vf_in).mean()
        pol_loss += (curr_pol_out**2).mean() * 1e-3
        pol_loss.backward()
        if parallel:
            average_gradients(curr_agent.policy)
        torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss},
                               self.niter)


    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)

        self.epsilon -= self.epsilon_decremental
        self.niter += 1
        self.logger.add_scalar('Global/Epsilon\\', self.epsilon, self.niter)

    def store_transition(self, s, a, r, s_, d):
        self.memory_counter = self.memory_counter + 1
        transition = self.memory.append((s,a,r,s_,d))
        # pop old data
        if(len(self.memory) > self.memory_capacity):
            self.memory.popleft()  

    def learn(self, gamma = 0.95, batch_size = 64):
        # sample data
        if(len(self.memory) < batch_size):
            batch_size = len(self.memory)
        b_memory = random.sample(self.memory,batch_size)
        b_s = torch.FloatTensor([x[0] for x in b_memory]).to(self.device)
        b_a = torch.LongTensor([x[1] for x in b_memory]).to(self.device)
        b_r = torch.FloatTensor([x[2] for x in b_memory]).to(self.device)
        b_s_ = torch.FloatTensor([x[3] for x in b_memory]).to(self.device)
        b_d = torch.FloatTensor([x[4] for x in b_memory]).to(self.device)

        for a_i in range(len(self.agents)):                   
            self.update((b_s, b_a, b_r, b_s_, b_d), a_i, gamma=gamma, logger=self.logger)       
        self.update_all_targets()                         

        #self.logger.add_scalar('Global/Loss\\', loss.detach().cpu().numpy(),self.target_replace_iter_count)
        #self.logger.add_scalar('Global/Epsilon\\', self.epsilon, self.niter)
        #self.loss.append(loss.detach().cpu().numpy())

        #if(self.target_replace_iter_count % self.target_replace_iter == 0):
            #self.replace_parameters()
            #self.epsilon -= self.epsilon_decremental
            #print("---- replace_parameters and decrease epsilon to {} !!".format(self.epsilon))

    def saveModel(self, path):
        [torch.save(a.policy,path+"agent_{}".format(i)) for i,a in enumerate(self.agents)]

