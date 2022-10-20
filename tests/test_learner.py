#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from component.learner import Learner
from algorithms.commnet import CommNet
from algorithms.proposed import MyAlgorithm
import argparse,time

from utils.log_utils import Logger

def test_learner():
    parse = argparse.ArgumentParser("Communication for MAS")
    parse.add_argument("--rank", type=int, default=0, help="dist rank")
    parse.add_argument("--world-size", type=int, default=1, help="dist world_size")
    parse.add_argument("--log-time", type=str, default=str(int(time.time())), help="log dir time")
    arglist = parse.parse_args()
    rank = arglist.rank
    world_size = arglist.world_size

    env = Learner._make_env("simple_spread", num_agents=7, max_episode_len=40, display=False)

    env.reset()

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

    master_ip = "localhost"
    master_port = "29700"
    tcp_store_ip = "localhost"
    tcp_store_port = "29701"
    world_size = world_size
    rank = rank
    backend = 'gloo'
    logger = Logger("parallel_learner", f"ac_network", rank, log_time=arglist.log_time)
    learner = Learner(myalg, master_ip, master_port, tcp_store_ip, tcp_store_port, rank, world_size, backend, logger)
    
    learner.inference()

