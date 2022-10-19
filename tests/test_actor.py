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
    arglist = parse.parse_args()
    rank = arglist.rank
    world_size = arglist.world_size

    master_ip = "localhost"
    master_port = "29500"
    tcp_store_ip = "localhost"
    tcp_store_port = "29501"
    backend = 'gloo'
    dist_comm = DistributedComm(master_ip, master_port, tcp_store_ip, tcp_store_port, rank, world_size, backend=backend)
    agent = ParallelizedActor(learner_rank=0, dist_comm=dist_comm, parallelism=3)
    agent.run(arglist.max_epochs)
    #env_w = ParallelizedEnvWrapper(parallelism=3)
    #env_w.run()

