#!/usr/bin/env python
# -*- coding: utf-8 -*-

import environment as envs
from typing import List

class BaseAgent():

    def __init__(self, env, algorithm, agent_ids:List[str]) -> None:
        self.algorithm = algorithm
        self.env = env
        self.agent_ids = agent_ids

    def _batch_take_action(self):
        actions = []
        for a_id in self.agent_ids:
            obs = self._get_obs(a_id)
            actions.append(self.algorithm.take_action(obs))

    def _get_reward(self, a_id):
        pass

    def _get_obs(self, a_id):
        pass

    def _get_comm(self, a_id):
        pass

    def save_rollout(self):
        pass
