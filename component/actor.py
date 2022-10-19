#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import Error
import environment as envs
from typing import List
import numpy as np
import torch.multiprocessing as mp
from torch.multiprocessing import Pipe
import time, os

from utils.distributed import DistributedComm

import torch

class BaseActor:
    def __init__(self) -> None:
        pass

    def main(self, child_pipe):
        num_agents = 7
        max_episode_len=40
        env = self._make_env("simple_spread", num_agents=num_agents, max_episode_len=max_episode_len, display=False)
        obs_n = env.reset()
        reward_n = env.init_reward
        delta_reward_n = [0 for i in range(num_agents)]

        while True:
            child_pipe.send([obs_n, reward_n, delta_reward_n])
            print("child agent sent observation")
            action = child_pipe.recv()
            print("child action get action", action)
            obs_n, reward_n, done_n, info_n, delta_reward_n = env.step(action)
            print(os.getpid(), reward_n)
            if any(done_n):
                env.reset()

    def _make_env(self, env_name, num_agents, max_episode_len, continuous_actions=False,display=False):                                                                       
        env = envs.load(env_name + ".py").Scenario(num_agent=num_agents, max_cycles=max_episode_len, continuous_actions=continuous_actions, display=display)
        return env                                                                               


class ParallelizedActor():
    """
    these agents take actions follow a same policy
    while they can be worked in different (parallel) envs 
    """

    def __init__(self, learner_rank: int, dist_comm:DistributedComm,
                 parallelism = 1, num_agents = 7, display = False) -> None:
        self.agent_ids = []
        self.learner_rank = learner_rank
        self.dist_comm = dist_comm

        self.parallelism = parallelism

        self.display = display

        self.num_agents = num_agents

        self.pipes = []
    
    def _remote_batch_inference(self):
        #print("p2p group:",self.dist_comm.get_p2p_comm_group())
        res = self.dist_comm.read_p2p_message(msg_shape=(self.parallelism,self.num_agents))
        print("get from learner", res)
        if len(res) == 0:
            print("nothing get from learner")
            return []
        src_list = list(map(lambda x:x[0], res))
        actions = list(map(lambda x:x[1], res))[0].numpy()
        actions = list(map(lambda x: [int(y) for y in x], actions))

        print("read_p2p_msg : ", src_list, actions)
        return actions

    def run(self, max_epochs):
        base_actor = BaseActor()
        self.pipes = self._multi_processes_wrapper(self.parallelism, base_actor.main)
        obs_p = []
        rew_p = []
        delta_rew_p = []
        for i in range(max_epochs):
            for p_i in self.pipes:
                temp = p_i.recv()
                obs_p.append(temp[0])
                rew_p.append(temp[1])
                delta_rew_p.append(temp[2]) 
            obs_t = torch.Tensor(np.array(obs_p))
            rew_t = torch.Tensor(np.array(rew_p)).reshape(self.parallelism, self.num_agents,-1)
            delta_rew_t = torch.Tensor(np.array(delta_rew_p)).reshape(self.parallelism, self.num_agents,-1)
            print(rew_t.shape)
            input_t = torch.concat((obs_t, rew_t, delta_rew_t), dim=2)
            #print(obs_n_p_t)

            hds = self.dist_comm.write_p2p_message([self.learner_rank], [input_t])
            #hds = self.dist_comm.write_p2p_message_batch_async([self.learner_rank], [input_tensors])

            actions = []
            #self.take_action(actions)
            while(len(actions) == 0):
                actions = self._remote_batch_inference()
                print("parent agent get actions", actions)
                if(len(actions) == 0):
                    time.sleep(1)

            for pi, action in zip(self.pipes, actions):
                pi.send(action)

            obs_p = []
            rew_p = []
            delta_rew_p = []

            #time.sleep(1)

    
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

