#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils.distributed import DistributedComm

from algorithms.commnet import CommNet

import environment as envs

import torch

import time

import numpy as np

class Learner:

    def __init__(self, algorithm, master_ip, master_port,
                 tcp_store_ip, tcp_store_port, rank, world_size, backend) -> None:
        self.dist_comm = DistributedComm(master_ip, master_port, tcp_store_ip, tcp_store_port, rank, world_size, backend=backend)
        self.algorithm = algorithm

    def inference(self):
        while True:
            print("read")
            #print("p2p group:",self.dist_comm.get_p2p_comm_group())
            #if len(self.dist_comm.get_p2p_comm_group()) == 0:
                #time.sleep(1)
            #    print("call continue")
            #    continue

            #res, _, _ = self.dist_comm.read_p2p_message_batch_async(per_msg_size=1,per_msg_shape=[(3,7,42)])
            res = self.dist_comm.read_p2p_message(msg_shape=(3,7,42))

            #self.dist_comm.reset_p2p_comm_group()

            actor_list = list(map(lambda x:x[0], res))
            obs = list(map(lambda x:x[1], res))
            #print("inference", obs)

            if len(obs) == 0:
                continue

            #actions = self.algorithm.choose_action(obs[0])
            actions = [torch.Tensor([Learner.test_random_action() for i in range(3)]) for j in range(len(actor_list))]

            print("before write", actor_list, actions)
            hds = self.dist_comm.write_p2p_message(actor_list, actions)
            #hds = self.dist_comm.write_p2p_message_batch_async(actor_list, actions)
            print("finish")


    @staticmethod
    def _make_env(env_name, num_agents, max_episode_len, continuous_actions=False,display=False):                                                                       
        env = envs.load(env_name + ".py").Scenario(num_agent=num_agents, max_cycles=max_episode_len, continuous_actions=continuous_actions, display=display)
        return env                                                                               

    @staticmethod
    def test_random_action():
        #n_action = [np.random.randint(0,action_space) for i in range(n_agents)]
        n_action = [np.random.randint(0,5) for i in range(7)]
        return n_action


    @staticmethod
    def test(rank, world_size):
        env = Learner._make_env("simple_spread", num_agents=7, max_episode_len=40, display=False)

        env.reset()

        algorithm = CommNet(env,                                  
                         learning_rate=1e-4,                                  
                         observation_shape=env.env.observe(env.env.agents[0]).shape,
                         num_actions=env.action_space,                              
                         num_agents = env.n_agents) 
        master_ip = "localhost"
        master_port = "29500"
        tcp_store_ip = "localhost"
        tcp_store_port = "29501"
        world_size = world_size
        rank = rank
        backend = 'gloo'
        learner = Learner(algorithm, master_ip, master_port, tcp_store_ip, tcp_store_port, rank, world_size, backend)
        
        learner.inference()

