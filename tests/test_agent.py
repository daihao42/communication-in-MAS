#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from component.agent import ParallelizedAgent
from utils.distributed import DistributedComm

import sys

import argparse

def test_parallelizedAgent():
    parse = argparse.ArgumentParser("Communication for MAS")
    parse.add_argument("--rank", type=int, default=0, help="dist rank")
    parse.add_argument("--world-size", type=int, default=1, help="dist world_size")
    arglist = parse.parse_args()
    rank = arglist.rank
    world_size = arglist.world_size

    master_ip = "127.0.0.1"
    master_port = "29700"
    tcp_store_ip = "127.0.0.1"
    tcp_store_port = "29701"
    backend = 'gloo'
    dist_comm = DistributedComm(master_ip, master_port, tcp_store_ip, tcp_store_port, rank, world_size, backend=backend)
    agent = ParallelizedAgent(learner_rank=0, dist_comm=dist_comm, parallelism=3)
    agent.run()
    #env_w = ParallelizedEnvWrapper(parallelism=3)
    #env_w.run()

