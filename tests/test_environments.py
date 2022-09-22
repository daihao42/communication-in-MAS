#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import time
import environment as envs
import numpy as np
import torch.multiprocessing as mp


def _multi_processes_wrapper(procs_size, func):
    """
    multiple processes managenment
    """
    #mp.set_start_method("spawn")
    mp.set_start_method('forkserver', force=True)
    processes = []
    for rank in range(procs_size):
        p = mp.Process(target=func, args=())
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    #for p in processes:
    #    p.close()



def _make_env(env_name, num_agents, max_episode_len, continuous_actions=False,display=False):                                                                       
    env = envs.load(env_name + ".py").Scenario(num_agent=num_agents, max_cycles=max_episode_len, continuous_actions=continuous_actions, display=display)
    return env                                                                               

def _random_action(action_space, n_agents):
    n_action = [np.random.randint(0,action_space) for i in range(n_agents)]
    return n_action

def rtest_simple_spread(display=False):
    env = _make_env("simple_spread", num_agents=7, max_episode_len=40, display=display)
    env.reset()

    for epoch in range(100):
        obs_n, reward_n, done_n, info_n, reward_n = env.step(_random_action(env.action_space, env.n_agents))
        env.render()
        time.sleep(0.2)
        print("reward_n : ",reward_n)
        print("global_reward : ",env.global_reward())
        if any(done_n):
            break

    env.close()



def rtest_traffic_junction(display=False):
    env = envs.load("traffic_junction" + ".py").Scenario(display=display)
    env.reset()

    for round in range(5):
        for epoch in range(100):
            obs_n, reward_n, done_n, info_n, reward_n = env.step(_random_action(env.action_space, env.n_agents))
            env.render()
            time.sleep(0.2)
            print("reward_n : ",reward_n)
            if any(done_n):
                break

        env.reset()

    env.close()

def rtest_tiger_deer(display=True):
    env = envs.load("tiger_deer" + ".py").Scenario(display=display)
    env.reset()

    for round in range(5):
        for epoch in range(100):
            deer_action = _random_action(env.action_space["deer"], env.n_agents["deer"])
            tiger_action = _random_action(env.action_space["tiger"], env.n_agents["tiger"])
            obs_n, reward_n, done_n, info_n, reward_n = env.step(deer_action + tiger_action)
            env.render()
            time.sleep(0.2)
            print("reward_n : ",reward_n)
            if any(done_n):
                break

        env.reset()

    env.close()


def test_multi_process_simple_spread():
    #_multi_processes_wrapper(3, rtest_simple_spread)

    #_multi_processes_wrapper(8, rtest_traffic_junction)

    _multi_processes_wrapper(4, rtest_tiger_deer)

if __name__ == '__main__':
    pytest.main(["-s","-v","test_environments.py"])


