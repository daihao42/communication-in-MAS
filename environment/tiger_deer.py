#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'dai'

from pettingzoo.magent import tiger_deer_v4
import time


class Scenario():

    def __init__(self,num_agent = 3, max_cycles = 25, continuous_actions = False, display = False) -> None:
        self._num_agent = {"deer":101, "tiger":20}
        self.max_cycles = max_cycles
        self.continuous_actions = continuous_actions
        self.env = self.make_env()
        self.action_space = {"deer":self.env.action_space(agent="deer_0").n,
                             "tiger":self.env.action_space(agent="tiger_0").n}
        self.display = display

    @property
    def n_agents(self):
        return self._num_agent


    def make_env(self):
        return tiger_deer_v4.env(max_cycles=self.max_cycles)

    def reset(self):
        self.env.reset()
        obs_n = []
        for i,agent in enumerate(self.env.agents):
            obs_n.append(self.env.observe(agent=agent))

        return obs_n

    '''
        bug fixed: the rewards and dones are make sense only after all agents take steps
    '''
    def step(self, actions):
        obs_n = []
        done_n = []
        info_n = []
        reward_n = []
        for i,agent in enumerate(self.env.agents):
            self.env.step(actions[i])
        for i,agent in enumerate(self.env.agents):
            reward_n.append(self.env.rewards[agent])
            obs_n.append(self.env.observe(agent=agent))
            done_n.append(self.env.dones[agent])
        return obs_n, reward_n, done_n, info_n, reward_n

    def state(self):
        return self.env.state()

    def close(self):
        if self.display:
            self.env.close()

    def render(self):
        if self.display:
            self.env.render()


