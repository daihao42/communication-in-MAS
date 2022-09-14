#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'dai'

from pettingzoo.magent import tiger_deer_v3
import time


class Scenario():

    def __init__(self,num_agent = 3, max_cycles = 25, continuous_actions = False, display = False) -> None:
        self.num_agent = num_agent
        self.max_cycles = max_cycles
        self.continuous_actions = continuous_actions
        self.env = self.make_env()
        self.action_space = self.env.action_space(agent="agent_0").n
        self.display = display

    def make_env(self):
        return tiger_deer_v3.env(N=self.num_agent, max_cycles=self.max_cycles, continuous_actions=self.continuous_actions)

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
        return obs_n, reward_n, done_n, info_n

    def state(self):
        return self.env.state()

    def close(self):
        if self.display:
            self.env.close()

    def render(self):
        if self.display:
            self.env.render()


