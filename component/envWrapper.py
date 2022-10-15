#!/usr/bin/env python
# -*- coding: utf-8 -*-

import environment as envs
import numpy as np
import torch.multiprocessing as mp
from torch.multiprocessing import Pipe
import time

class BaseAgent:
    def __init__(self) -> None:
        pass

    def main(self, child_pipe):
        env = self._make_env("simple_spread", num_agents=7, max_episode_len=40, display=False)
        obs_n = env.reset()

        while True:
            child_pipe.send(obs_n)
            print("test")
            action = child_pipe.recv()
            print("test2", action)
            obs_n, reward_n, done_n, info_n, reward_n = env.step(action)

    def _make_env(self, env_name, num_agents, max_episode_len, continuous_actions=False,display=False):                                                                       
        env = envs.load(env_name + ".py").Scenario(num_agent=num_agents, max_cycles=max_episode_len, continuous_actions=continuous_actions, display=display)
        return env                                                                               



class ParallelizedEnvWrapper:

    def __init__(self, parallelism = 1, display = False):
        self.parallelism = parallelism
        self.display = display

        self.pipes = []

    def run(self):

        ts = BaseAgent()
        self.pipes = self._multi_processes_wrapper(self.parallelism, ts.main)
        obs_n_p = []
        for i in range(10):
            for pi in self.pipes:
                obs_n_p.append(pi.recv())
            #actions = self.test_random_action(self.envs[0].action_space, self.envs[0].n_agents)
            actions = [self.test_random_action() for i in range(self.parallelism)]
            #self.take_action(actions)
            print("call send")
            for pi, action in zip(self.pipes, actions):
                pi.send(action)


    #def test_random_action(self, action_space, n_agents):
    def test_random_action(self):
        #n_action = [np.random.randint(0,action_space) for i in range(n_agents)]
        n_action = [np.random.randint(0,5) for i in range(7)]
        return n_action


    def main(self, child_pipe):
        env = self._make_env("simple_spread", num_agents=7, max_episode_len=40, display=self.display)
        obs_n = env.reset()

        while True:
            child_pipe.send(obs_n)
            print("test")
            action = child_pipe.recv()
            print("test2", action)
            obs_n, reward_n, done_n, info_n, reward_n = env.step(action)


    def _multi_processes_wrapper(self, procs_size, func):
        """
        multiple processes managenment
        """
        mp.set_start_method("spawn")
        #mp.set_start_method('forkserver', force=True)
        processes = []
        pipes = []
        for rank in range(procs_size):
            (parent_pipe, child_pipe) = Pipe()
            p = mp.Process(target=func, args=(child_pipe,))
            p.start()
            processes.append(p)
            pipes.append(parent_pipe)

        #for p in processes:
        #    p.join()

        return pipes

    def _make_env(self, env_name, num_agents, max_episode_len, continuous_actions=False,display=False):                                                                       
        env = envs.load(env_name + ".py").Scenario(num_agent=num_agents, max_cycles=max_episode_len, continuous_actions=continuous_actions, display=display)
        return env                                                                               

    #def take_action(self, actions):
        #for env, action in zip(self.envs, actions):
            #env.step(action)

    @property
    def action_space(self):
        return 0
