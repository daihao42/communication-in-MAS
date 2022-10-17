#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from component.learner import Learner
from algorithms.commnet import CommNet
from algorithms.proposed import MyAlgorithm
import argparse
import time
import numpy as np

def test_algorithm():

    rank = 0

    env = Learner._make_env("simple_spread", num_agents=7, max_episode_len=40, display=False)

    obs_n = env.reset()

    reward_n = env.init_reward

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
    for epoch in range(10):
        print("obs shape",np.array(obs_n).shape)
        action = myalg.choose_action(obs_n)
        print("partial action:",myalg.choose_action(obs_n[:4]))
        action = myalg.choose_action(obs_n)
        print("action", action)

        buffer.append((obs_n, action))

        obs_n, reward_n, done_n, info_n, reward_n = env.step(action)

        buffer[-1] = buffer[-1] + (reward_n, obs_n)

        env.render()
        time.sleep(0.2)
        print("reward_n : ",reward_n)
        print("global_reward : ",env.global_reward())
        if any(done_n):
            break

    myalg.train(buffer)

    env.close()



