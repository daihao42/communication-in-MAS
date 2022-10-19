#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from component.actor import ParallelizedActor
from utils.distributed import DistributedComm

import sys

import argparse

def test_parallelizedAgent():
    parse = argparse.ArgumentParser("Communication for MAS")
    parse.add_argument("--rank", type=int, default=0, help="dist rank")
    parse.add_argument("--world-size", type=int, default=1, help="dist world_size")
    parse.add_argument("--max-epochs", type=int, default=1000, help="max epochs")
    parse.add_argument("--learner-rank", type=int, default=0, help="learner rank")
    parse.add_argument("--parallelism", type=int, default=3, help="parallelism")
    arglist = parse.parse_args()
    rank = arglist.rank
    world_size = arglist.world_size

    master_ip = "localhost"
    master_port = "29700"
    tcp_store_ip = "localhost"
    tcp_store_port = "29701"
    backend = 'gloo'
    dist_comm = DistributedComm(master_ip, master_port, tcp_store_ip, tcp_store_port, rank, world_size, backend=backend)
    agent = ParallelizedActor(learner_rank=arglist.learner_rank, dist_comm=dist_comm, parallelism=arglist.parallelism)
    agent.run(arglist.max_epochs)
    #env_w = ParallelizedEnvWrapper(parallelism=3)
    #env_w.run()

