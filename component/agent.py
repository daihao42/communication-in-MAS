#!/usr/bin/env python
# -*- coding: utf-8 -*-

import environment as envs
from typing import List
import numpy as np
import torch.multiprocessing as mp
from torch.multiprocessing import Pipe
import time, os

from utils.distributed import DistributedComm

import torch

class BaseAgent:
    def __init__(self) -> None:
        pass

    def main(self, child_pipe):
        env = self._make_env("simple_spread", num_agents=7, max_episode_len=40, display=False)
        obs_n = env.reset()
        reward_n = env.init_reward

        while True:
            #child_pipe.send([obs_n, reward_n])
            child_pipe.send(obs_n)
            print("child agent sent observation")
            action = child_pipe.recv()
            print("child action get action", action)
            obs_n, reward_n, done_n, info_n, reward_n = env.step(action)
            print(os.getpid(), reward_n)
            if any(done_n):
                env.reset()

    def _make_env(self, env_name, num_agents, max_episode_len, continuous_actions=False,display=False):                                                                       
        env = envs.load(env_name + ".py").Scenario(num_agent=num_agents, max_cycles=max_episode_len, continuous_actions=continuous_actions, display=display)
        return env                                                                               


class ParallelizedAgent():
    """
    these agents take actions follow a same policy
    while they can be worked in different (parallel) envs 
    """

    def __init__(self, learner_rank: int, dist_comm:DistributedComm, parallelism = 1, display = False) -> None:
        self.envs = envs
        self.agent_ids = []
        self.learner_rank = learner_rank
        self.dist_comm = dist_comm

        self.parallelism = parallelism

        self.display = display

        self.pipes = []
    
    def _remote_batch_inference(self, input_tensors):
        print("input_tensors", input_tensors.shape)
        hds = self.dist_comm.write_p2p_message([self.learner_rank], [input_tensors])
        #hds = self.dist_comm.write_p2p_message_batch_async([self.learner_rank], [input_tensors])

        print("p2p group:",self.dist_comm.get_p2p_comm_group())
        res = self.dist_comm.read_p2p_message(msg_shape=(3,7))
        print("get from learner", res)
        if len(res) == 0:
            print("nothing get from learner")
            return [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]
        src_list = list(map(lambda x:x[0], res))
        actions = list(map(lambda x:x[1], res))[0].numpy()
        actions = list(map(lambda x: [int(y) for y in x], actions))

        print("read_p2p_msg : ", src_list, actions)
        return actions

    def run(self):
        env = BaseAgent()
        self.pipes = self._multi_processes_wrapper(self.parallelism, env.main)
        obs_rew_p = []
        for i in range(100):
            for pi in self.pipes:
                obs_rew_p.append(pi.recv())
            #actions = self.test_random_action(self.envs[0].action_space, self.envs[0].n_agents)
            #obs_n_p_t = torch.Tensor(np.array(list(map(lambda x:x[0], obs_rew_p))))
            obs_n_p_t = torch.Tensor(np.array(obs_rew_p))
            #print(obs_n_p_t)
            actions = self._remote_batch_inference(obs_n_p_t)
            #self.take_action(actions)
            print("parent agent get actions", actions)
            for pi, action in zip(self.pipes, actions):
                pi.send(action)

            obs_n_p = []

            time.sleep(1)

    
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

    @property
    def action_space(self):
        return 0

    def _get_reward(self, a_id):
        pass

    def _get_obs(self, env, a_id):
        pass

    def _get_comm(self, a_id):
        pass

    def save_rollout(self):
        pass
