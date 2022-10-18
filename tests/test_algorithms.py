#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from component.learner import Learner
from algorithms.commnet import CommNet
from algorithms.proposed import MyAlgorithm
import argparse
import time
import numpy as np

from utils.log_utils import Logger

import torch

def test_algorithm():

    rank = 0

    logger = Logger("test_algorithm", f"ac_network", rank)

    env = Learner._make_env("simple_spread", num_agents=7, max_episode_len=40, display=False)

    obs_n = env.reset()

    algorithm = CommNet(env,                                  
                     learning_rate=1e-4,                                  
                     observation_shape=env.env.observe(env.env.agents[0]).shape,
                     num_actions=env.action_space,                              
                     num_agents = env.n_agents) 

    myalg = MyAlgorithm(env,
                        learning_rate=1e-4,                                  
                        observation_shape=env.env.observe(env.env.agents[0]).shape,
                        num_actions=env.action_space,                              
                        num_agents = env.n_agents,
                        rank=rank) 

    buffer = []

    epoch = 0
    for epoch in range(1000000):
        #print("obs shape",np.array(obs_n).shape)
        action, _probs = myalg.choose_action(obs_n)
        action2, _probs2 = myalg.choose_action(obs_n)
        #print("action / action2",action, action2)
        stable = torch.nn.functional.cross_entropy(torch.Tensor(_probs).reshape((-1,5)), torch.Tensor(_probs2).reshape((-1,5)))
        logger.add_scalar("Inference/Stable", stable, epoch)
        #print("action", action)

        buffer.append((obs_n, action, _probs))

        obs_n, reward_n, done_n, info_n, delta_reward_n = env.step(action)

        buffer[-1] = buffer[-1] + (delta_reward_n, obs_n)

        #env.render()
        #time.sleep(0.2)
        #print("reward_n : ",reward_n)
        #print("global_reward : ",env.global_reward())
        epoch = epoch + 1
        logger.add_scalar("Train/Reward", env.global_reward(), epoch)
        if any(done_n):
            critic_loss, actor_loss = myalg.train(buffer)
            buffer = []
            logger.add_scalar("Train/Final_Reward", env.global_reward(), epoch)
            logger.add_scalar("Train/Critic_Loss", critic_loss, epoch)
            logger.add_scalar("Train/Actor_Loss", actor_loss, epoch)

            env.reset()



    env.close()



