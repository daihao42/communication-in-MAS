#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

class Analyzer():

    def __init__(self, events_path):
        self._ea = event_accumulator.EventAccumulator(events_path)
        self._ea.Reload()

    def list_scalar_keys(self):
        return self._ea.scalars.Keys()

    def get_scalar(self, key):
        return pd.DataFrame(self._ea.scalars.Items("Train/Loss/0/rank_0\\"))

if __name__ == '__main__':
    e_path = "/home/dai/Data/Project/MARL/communication_in_MAS/communication-in-MAS/logs/test_logs/actor_with_estimate/1662691061/tensorboard" 
    ana = Analyzer(e_path)
    keys = ana.list_scalar_keys()
    print(ana.get_scalar(keys[0]))
