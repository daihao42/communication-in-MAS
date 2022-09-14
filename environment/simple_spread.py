#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'dai'

from pettingzoo.mpe import simple_spread_v2
import time
import numpy as np


class Scenario():

    def __init__(self,num_agent = 3, max_cycles = 25, continuous_actions = False, display = False) -> None:
        self.num_agent = num_agent
        self.max_cycles = max_cycles
        self.continuous_actions = continuous_actions
        self.env = self.make_env()
        self.action_space = self.env.action_space(agent="agent_0").n
        self.display = display
        self.last_reward = []

    def make_env(self):
        return simple_spread_v2.env(N=self.num_agent, max_cycles=self.max_cycles, continuous_actions=self.continuous_actions)

    def reset(self):
        self.env.reset()
        obs_n = []
        rew_n = []
        for i,agent in enumerate(self.env.agents):
            obs_n.append(self.env.observe(agent=agent))
            rew_n.append(self.env.rewards[agent])

        self.last_reward = rew_n

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
            #reward_n.append(self.env.rewards[agent])
            obs_n.append(self.env.observe(agent=agent))
            done_n.append(self.env.dones[agent])

        reward_n = np.repeat(self.global_reward(),self.num_agent)
        delta_reward_n = []
        for i,agent in enumerate(self.env.agents):
            delta_reward_n.append(reward_n[i] - self.last_reward[i])

        self.last_reward = reward_n
        #return obs_n, delta_reward_n, done_n, info_n, reward_n
        return obs_n, reward_n, done_n, info_n, reward_n

    def bound_env(self, actions):
        pass
        #for i,agent in enumerate(self.env.agents):
        #    if 


    '''
        min distance used in petting, i changed it to max
    '''
    def global_reward(self):
        world = self.env.env.env.world
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= max(dists)
        return rew

    def state(self):
        return self.env.state()

    def close(self):
        if self.display:
            self.env.close()

    def render(self):
        if self.display:
            self.env.render()


