#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'dai'

import time
import gym


class Scenario():

    def __init__(self, grid_shape=(14, 14), step_cost=-0.01, n_max=10, collision_reward=-10,
                 arrive_prob=0.5, full_observable: bool = False, max_steps: int = 100, continuous_actions = False, display = False) -> None:
        self.continuous_actions = continuous_actions
        self.env = self.make_env(grid_shape, step_cost, n_max, collision_reward, arrive_prob, full_observable, max_steps)
        self.action_space = self.env.action_space[0].n
        self._num_agent = self.env.n_agents
        self.display = display
        self.observation_shape = self.env.observation_space[0].shape
        self.ep_reward = 0

    @property
    def n_agents(self):
        return self._num_agent

    def make_env(self, grid_shape, step_cost, n_max, collision_reward, arrive_prob, full_observable, max_steps):
        return gym.make('ma_gym:TrafficJunction10-v0',grid_shape=grid_shape, step_cost=step_cost, n_max=n_max, collision_reward=collision_reward, arrive_prob=arrive_prob, full_observable=full_observable, max_steps=max_steps)

    def reset(self):
        self.ep_reward = 0
        return self.env.reset()

    def step(self, actions):
        obs_n, reward_n, done_n, info_n = self.env.step(actions)
        self.ep_reward += sum(reward_n)
        return obs_n, [self.ep_reward for i in range(self._num_agent)], done_n, info_n, reward_n
        #return obs_n, reward_n, done_n, info_n, reward_n

    def close(self):
        self.env.close()

    def render(self):
        if self.display:
            self.env.render()

