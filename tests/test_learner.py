#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from component.learner import Learner

import argparse

def test_learner():
    parse = argparse.ArgumentParser("Communication for MAS")
    parse.add_argument("--rank", type=int, default=0, help="dist rank")
    parse.add_argument("--world-size", type=int, default=1, help="dist world_size")
    arglist = parse.parse_args()
    rank = arglist.rank
    world_size = arglist.world_size

    Learner.test(rank, world_size)

