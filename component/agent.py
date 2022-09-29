#!/usr/bin/env python
# -*- coding: utf-8 -*-

import environment as envs

class BaseAgent():

    def __init__(self, env_name, algorithm) -> None:
        self.env_name = env_name
        self.algorithm = algorithm

    def _make_env(self):
        env = envs.load(self.env_name + ".py").Scenario()
        return env                                                                               

    def _do_action(self):
        pass

    def _get_reward(self):
        pass

    def _get_obs(self):
        pass

    def _get_comm(self):
        pass

    def save_rollout(self):
        pass
