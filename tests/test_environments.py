#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import time
import environment as envs

def _make_env(env_name, num_agents, max_episode_len, continuous_actions=False,display=False):                                                                       
    env = envs.load(env_name + ".py").Scenario(num_agent=num_agents, max_cycles=max_episode_len, continuous_actions=continuous_actions, display=display)
    return env                                                                               

def test_simple_spread():
    env = _make_env("simple_spread", num_agents=7, max_episode_len=40)
    env.reset()

    print(env.state())
    for epoch in range(100):
        env.step([])
        env.render()
        time.sleep(0.2)

    env.close()




