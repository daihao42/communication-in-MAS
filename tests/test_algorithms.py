#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from component.learner import Learner
from algorithms.commnet import CommNet
from algorithms.proposed import MyAlgorithm
from algorithms.noac import NOAC
import argparse
import time
import numpy as np

from utils.log_utils import Logger

import torch

def get_device(rank):
    if torch.cuda.is_available():
        device_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        return f"cuda:{device_id}"
    else:
        return "cpu"


def test_noac_train():
    rank = 0

    env = Learner._make_env("simple_spread", num_agents=7, max_episode_len=40, display=False)

    device = get_device(0)

    obs_n = env.reset()


    logger = Logger("noac", f"noac", rank)
    neighbors = 3
    noacs = [NOAC(env,
                learning_rate=5e-4,
                observation_shape=env.env.observe(env.env.agents[0]).shape,
                num_actions=env.action_space,
                num_agents = env.n_agents,
                rank=rank,
                device=device,
                  neighbors=neighbors,
                message_size=12)
             for i in range(env.n_agents)
             ]

    def _get_neighbor_ids(a_id, ls):
        res = []
        for i in range(neighbors):
            res.append(ls[(a_id+i)% env.n_agents])
        return res

    buffer = [[] for i in range(env.n_agents)]

    epoch = 0
    for epoch in range(100000):
        print(f"training in epoch : {epoch}")
        msgs = []
        for a_id in range(env.n_agents):
            msg = noacs[a_id]._encoder(obs=obs_n[a_id])
            msgs.append(msg)

        for a_id in range(env.n_agents):
            if(len(buffer[a_id]) != 0):
                buffer[a_id][-1] = buffer[a_id][-1] + (msgs[a_id], _get_neighbor_ids(a_id, msgs))

        actions = []
        _probs_s = []
        for a_id in range(env.n_agents):
            action,_probs = noacs[a_id].choose_action(obs_n[a_id] ,msg=msgs[a_id], neighbors_msgs=_get_neighbor_ids(a_id, msgs), greedy=True)
            actions.append(action.cpu().detach().numpy()[0])
            _probs_s.append(_probs)
            action2, _probs2 = noacs[a_id].choose_action(obs_n[a_id] ,msg=msgs[a_id], neighbors_msgs=_get_neighbor_ids(a_id, msgs), greedy=True)
            #print("action / action2",action, action2)
            stable = torch.nn.functional.cross_entropy(torch.Tensor(_probs).reshape((-1,5)), torch.Tensor(_probs2).reshape((-1,5)))
            logger.add_scalar("Inference/{aid}/Stable", stable, epoch)
            #print("action", action)

            buffer[a_id].append((obs_n[a_id], actions[a_id], _probs_s[a_id], msgs[a_id], _get_neighbor_ids(a_id, msgs)))

        print(f"actions:{actions}")

        obs_n, reward_n, done_n, info_n, delta_reward_n = env.step(actions)

        for a_id in range(env.n_agents):

            buffer[a_id][-1] = buffer[a_id][-1] + (delta_reward_n[a_id], obs_n[a_id])

            #env.render()
            #time.sleep(0.2)
            #print("reward_n : ",reward_n)
            #print("global_reward : ",env.global_reward())
        epoch = epoch + 1
        logger.add_scalar("Train/Reward", env.global_reward(), epoch)

        if any(done_n):
            logger.add_scalar("Train/Final_Reward", env.global_reward(), epoch)
            for a_id in range(env.n_agents):   
                critic_loss, actor_loss, encoder_loss = noacs[a_id].train(buffer[a_id][:-1])
                buffer[a_id] = []
                logger.add_scalar(f"Train/{a_id}/Critic_Loss", critic_loss, epoch)
                logger.add_scalar(f"Train/{a_id}/Actor_Loss", actor_loss, epoch)
                logger.add_scalar(f"Train/{a_id}/encoder_Loss", encoder_loss, epoch)

            env.reset()

    for eval_epoch in range(200):

        msgs = []
        for a_id in range(env.n_agents):
            msg = noacs[a_id]._encoder(obs=obs_n[a_id])
            msgs.append(msg)

        actions = []
        _probs_s = []
        for a_id in range(env.n_agents):
            action,_probs = noacs[a_id].choose_action(obs_n[a_id] ,msg=msgs[a_id], neighbors_msgs=_get_neighbor_ids(a_id, msgs), greedy=True)
            actions.append(action.cpu().detach().numpy()[0])
            _probs_s.append(_probs)
            #print("action / action2",action, action2)

        obs_n, reward_n, done_n, info_n, delta_reward_n = env.step(actions)

        env.render()

        if any(done_n):
            env.reset()

        time.sleep(1)

    env.close()

