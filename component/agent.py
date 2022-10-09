#!/usr/bin/env python
# -*- coding: utf-8 -*-

import environment as envs
from typing import List
from threading import RLock

from utils.distributed import DistributedComm
from component.envWrapper import ParallelizedEnvWrapper

class ParallelizedAgent():
    """
    these agents take actions follow a same policy
    while they can be worked in different (parallel) envs 
    """

    def __init__(self, env_list: List[ParallelizedEnvWrapper], learner_rank: int, dist_comm:DistributedComm) -> None:
        self.lock =RLock()
        self.envs = env_list
        self.agent_ids = []
        self.learner_rank = learner_rank
        self.dist_comm = dist_comm
        self.action_space = envs[0].action_space

    def add_hook(self, env:ParallelizedEnvWrapper, agent_id:str):
        self.lock.acquire()
        try:
            self.envs.append(env)
            self.agent_ids.append(agent_id)
        finally:
            self.lock.release()

    def _batch_take_action(self):
        obs_batch = []
        for env,a_id in zip(self.envs,self.agent_ids):
            obs_batch.append(self._get_obs(env,a_id))
        return self._remote_batch_inference(obs_batch)

    def _remote_batch_inference(self, input_tensors):
        self.dist_comm.write_p2p_message([self.learner_rank], [input_tensors])
        return self.dist_comm.read_p2p_message(msg_shape=(len(self.agent_ids),self.action_space))

    def _get_reward(self, a_id):
        pass

    def _get_obs(self, env, a_id):
        pass

    def _get_comm(self, a_id):
        pass

    def save_rollout(self):
        pass
