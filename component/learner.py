#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils.distributed import DistributedComm

import threading

import environment as envs

import torch

import time

import numpy as np

class ReplayBuffer :

    def __init__(self, num_actors, parallelism, num_agents, buffer_size = 10000) -> None:
        """
        rank : parallelism : agent : deque
        """
        self.num_actors = num_actors
        self.parallelism = parallelism
        self.num_agents = num_agents

        self.replay_buffer = {}
        self.buffer_size = buffer_size

        self.temp_last_obs_action = {}

    def _construct_buffer(self):
        """
        actor_id is rank
        """
        return [[ [] for agent in range(self.num_agents)] for para in range(self.parallelism)]

    def store_transition(self, actor_list, obss, actions, _probs):
        '''
        obs, action, reward, next_obs
        '''
        for i,actor_id in enumerate(actor_list):
            if actor_id not in self.replay_buffer:
                self.replay_buffer[actor_id] = self._construct_buffer()
            for para in range(self.parallelism):
                for agent in range(self.num_agents):
                    self._check_over_length(self.replay_buffer[actor_id][para][agent])
                    self.replay_buffer[actor_id][para][agent].append((obss[i][para][agent],
                                                                      actions[i][para][agent], _probs[i][para][agent])
                                                                     + self.temp_last_obs_action[actor_id][para][agent][0])
        self.temp_last_obs_action = {}

    def store_reward(self, actor_list, reward_n, next_obs):
        for i,actor_id in enumerate(actor_list):
            self.temp_last_obs_action[actor_id] = self._construct_buffer()
            for para in range(self.parallelism):
                for agent in range(self.num_agents):
                    self.temp_last_obs_action[actor_id][para][agent].append((reward_n[i][para][agent],next_obs[i][para][agent]))

    def _check_over_length(self, buf):
        if len(buf) > self.buffer_size:
            buf.remove(buf[0])
            
    def sample(self, actor_list, batch_size=64):
        sample_data = []
        m_batch_size = batch_size
        for actor_id in actor_list:
            for para in range(self.parallelism):
                batch_size = m_batch_size
                for agent in range(self.num_agents):
                    if(batch_size > len(self.replay_buffer[actor_id][para][agent]) - 1):
                        batch_size = len(self.replay_buffer[actor_id][para][agent]) - 1
                    sample_data.append(self.replay_buffer[actor_id][para][agent][:batch_size])
                    self.replay_buffer[actor_id][para][agent] = self.replay_buffer[actor_id][para][agent][batch_size:]
        return sample_data

class Learner:

    def __init__(self, algorithm, master_ip, master_port,
                 tcp_store_ip, tcp_store_port, rank, world_size, backend, logger,
                 num_actors = 2, parallelism = 3 , num_agents = 7,obs_shape = 42, num_actions=5) -> None:
        self.dist_comm = DistributedComm(master_ip, master_port, tcp_store_ip, tcp_store_port, rank, world_size, backend=backend)
        self.algorithm = algorithm
        self.num_actors = num_actors
        self.parallelism = parallelism
        self.num_agents = num_agents
        self.obs_shape = obs_shape
        self.num_actions = num_actions

        self.replay_buffer = ReplayBuffer(num_actors, parallelism, num_agents)

        self.logger = logger

        self.rank = rank

    def inference(self):
        epochs = 1
        while True:
            print("read")
            #print("p2p group:",self.dist_comm.get_p2p_comm_group())
            #if len(self.dist_comm.get_p2p_comm_group()) == 0:
                #time.sleep(1)
            #    print("call continue")
            #    continue

            #res, _, _ = self.dist_comm.read_p2p_message_batch_async(per_msg_size=1,per_msg_shape=[(3,7,42)])
            res = self.dist_comm.read_p2p_message(msg_shape=(self.parallelism,self.num_agents,self.obs_shape+2))

            #self.dist_comm.reset_p2p_comm_group()

            # [actor (random) , parallelism, agents]
            actor_list = list(map(lambda x:x[0], res))
            obs_rew = torch.Tensor(np.array(list(map(lambda x:x[1].numpy(), res)))) # shape = (actors, parallelism, agents, obs+rew shape)
            #print("inference", obs)

            if len(obs_rew) == 0:
                continue

            obs = obs_rew[:,:,:,:self.obs_shape]
            last_distance = obs_rew[:,:,:,self.obs_shape:self.obs_shape+1]
            last_rew = obs_rew[:,:,:,self.obs_shape+1:]

            print("obs", obs.shape)
            print("last_rew", last_rew.shape)

            actions,_probs = self.algorithm.choose_action(obs.reshape((-1,self.obs_shape)))
            #actions = [torch.Tensor([Learner.test_random_action() for i in range(3)]) for j in range(len(actor_list))]

            actions = actions.reshape((len(actor_list),self.parallelism,self.num_agents)).cpu().detach().numpy()
            _probs = _probs.reshape((len(actor_list),self.parallelism,self.num_agents,self.num_actions))

            print("before write", actor_list, actions)
            hds = self.dist_comm.write_p2p_message(actor_list, [torch.Tensor(a) for a in actions])
            #hds = self.dist_comm.write_p2p_message_batch_async(actor_list, actions)

            #print(actions.shape)
            self.replay_buffer.store_reward(actor_list, last_rew.cpu().numpy(), obs.cpu().numpy())
            self.replay_buffer.store_transition(actor_list,obs.cpu().numpy(),actions, _probs)


            self.logger.add_scalar(f"Train/{self.rank}/Reward", torch.mean(last_rew), epochs)
            self.logger.add_scalar(f"Train/{self.rank}/Distance", torch.mean(last_distance), epochs)

            #print(self.replay_buffer.replay_buffer)

            print("finish")

            #parent_pipe.send(self.replay_buffer)

            epochs = epochs + 1
            print("epochs : ", epochs)
            if (epochs % 40) == 0:
                print("up training processing")
                t1 = threading.Thread(target=self.train,args=(actor_list,epochs))
                t1.start()
                self.logger.add_scalar(f"Train/{self.rank}/Final_Reward", torch.mean(last_rew), epochs)
                self.logger.add_scalar(f"Train/{self.rank}/Final_Distance", torch.mean(last_distance), epochs)
                #self.train(actor_list, epochs)


    def train(self, actor_list, epochs):
        print("training whihin data")
        sample_data = self.replay_buffer.sample(actor_list=actor_list)
        for data in sample_data:
            critic_loss, actor_loss = self.algorithm.train(data)
            self.logger.add_scalar(f"Train/{self.rank}/Critic_Loss", critic_loss, epochs)
            self.logger.add_scalar(f"Train/{self.rank}/Actor_Loss", actor_loss, epochs)

        print("training finished")

    @staticmethod
    def _make_env(env_name, num_agents, max_episode_len, continuous_actions=False,display=False):                                                                       
        env = envs.load(env_name + ".py").Scenario(num_agent=num_agents, max_cycles=max_episode_len, continuous_actions=continuous_actions, display=display)
        return env                                                                               

    @staticmethod
    def test_random_action():
        #np.random.seed(1)
        #n_action = [np.random.randint(0,action_space) for i in range(n_agents)]
        n_action = [np.random.randint(0,5) for i in range(7)]
        return n_action


