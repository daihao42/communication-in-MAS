#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch as th
from torch import Tensor
import torch.distributed as dist
import numpy as np

import os
import copy
import time, datetime
from typing import List, Tuple

__author__ = 'dai'

'''
distributed communication through pytorch.distributed
'''

class DistributedComm():

    def __init__(self, master_ip:str, master_port:str, tcp_store_ip:str, tcp_store_port:str, rank:int, world_size:int, backend='gloo', tcp_store_timeout=30) -> None:
        os.environ['MASTER_ADDR'] = master_ip
        os.environ['MASTER_PORT'] = master_port
        dist.init_process_group(backend = backend, rank = rank, world_size = world_size)

        self._world_size = world_size

        self._rank = rank

        #self._p2p_term = 0
        self._dist = dist

        self._tcp_store_ip = tcp_store_ip
        self._tcp_store_port = tcp_store_port

        if rank == 0:
            self._tcp_store = dist.TCPStore(tcp_store_ip, int(tcp_store_port), world_size, True, datetime.timedelta(seconds=tcp_store_timeout))
            self.init_comm_channel()
            #self._tcp_store_logic_p2p_term_init()
        else : 
            time.sleep(1)
            self._tcp_store = dist.TCPStore(tcp_store_ip, int(tcp_store_port), world_size, False, datetime.timedelta(seconds=tcp_store_timeout))

    def _destory_distributed_group():
        dist.destroy_process_group()

    def _group_broadcast_send(self, msg, src, group = None, async_op = False):
        dist.broadcast(msg,src = src, group=group, async_op=async_op)

    def _group_broadcast_receive(self, msg, src):
        dist.broadcast(msg,src = src)
        return msg

    def _sync_send(self, msg, dst):
        dist.send(msg, dst = dst)

    def _sync_recv(self, msg, src):
        dist.recv(msg, src = src)
        return msg

    def _async_broadcast_send(self, msg, src):
        handle = dist.broadcast(msg,src = src, async_op=True)
        return handle

    def _async_send(self, msg, dst):
        handle = dist.isend(msg, dst = dst)
        return handle

    def _async_recv(self, msg, src):
        handle = dist.irecv(msg, src = src)
        return handle, msg

    def _p2p_batch_op(self, func, tensor, b_rank):
        return dist.P2POp(func, tensor, b_rank)

    def _p2p_batch_op_execute(self, op_list):
        return dist.batch_isend_irecv(op_list)

    def _async_wrapper_is_completed(self, handle):
        return handle.is_completed()

    def _async_wrapper_wait(self, handle):
        handle.wait()

    def _tcp_store_set(self, key, value):
        return self._tcp_store.set(key, value)

    def _tcp_store_get(self, key):
        return self._tcp_store.get(key).decode(encoding="utf-8")

    '''
    def _tcp_store_logic_p2p_term_init(self):
        self._tcp_store.set("p2p_term", "0")

    def _tcp_store_logic_p2p_term_inc(self):
        self._tcp_store.add("p2p_term", 1)
    '''

    def _tcp_store_wait(self, key_list:List[str], timeout = 30):
        self._tcp_store.wait(key_list, datetime.timedelta(seconds=timeout))

    def _tcp_store_delete(self, key):
        self._tcp_store.delete_key(key)

    '''
        init communication receive channels
        peer_to_peer_rank : 1,2,3
        broadcast_rank : 2,3,4
    '''
    def init_comm_channel(self):
        for i in range(self._world_size):
            self._tcp_store_set("p2p_{rank}".format(rank = i),"")
            self._tcp_store_set("broadcast_{rank}".format(rank = i),"")

    #################################################
    '''
    Peer to peer
    '''
    #################################################

    '''
    @deprecated append ur own rank_id to the one who u need to communicate with.
    '''
    def set_p2p_comm_group(self, comm_list):
        self._tcp_store_set("p2p_{rank}".format(rank = self._rank),",".join(map(lambda x: str(x),comm_list)))
        #self._tcp_store_set("p2p_{rank}".format(rank = self._rank),",")
        #for i in comm_list:
        #    temp = self._tcp_store_get("p2p_{rank}".format(rank = i))
        #    self._tcp_store_set("p2p_{rank}".format(rank = i),temp+",{rank}".format(rank=self._rank))

    '''
    call it after received all msg from other agents.
    '''
    def reset_p2p_comm_group(self):
        self._tcp_store_set("p2p_{rank}".format(rank = self._rank),"")

    def get_p2p_comm_group(self):
        src_list = []
        for src_rank in range(self._world_size):
            dst_list = self._tcp_store_get("p2p_{rank}".format(rank = src_rank)).split(",")
            if dst_list == [""]:
                continue
            dst_list = map(lambda x:int(x), dst_list)
            if self._rank in dst_list:
                src_list.append(src_rank)
        return src_list

    def write_p2p_message(self, comm_list:List[int], msgs:List[Tensor]):
        """
        write msg to dst by async send - isend()
        """
        self.set_p2p_comm_group(comm_list)
        handles = [self._async_send(msg, dst_rank)  
                for dst_rank, msg in zip(comm_list, msgs)]
        return handles

    def read_p2p_message(self, msg_shape=(1,)):
        """
        receive msg from src by sync recv - recv()
        """
        src_list = self.get_p2p_comm_group()
        msgs = []
        temp = th.zeros(msg_shape)
        for src_rank in src_list:
            self._sync_recv(temp, src_rank)
            # deepcopy is necessary cause recv() changes tmep every calls,
            # leading msgs to be [last_recv, last_recv, ...]
            msgs.append((src_rank, copy.deepcopy(temp)))
        return msgs

    def write_p2p_message_batch_async(self, comm_list:List[int], msgs:List[Tensor]):
        """
        batch async send msg - P2POps and batch_isend_irecv()
        send msgs to comm_list respectively
        """
        self.set_p2p_comm_group(comm_list)
        op_list = []

        for dst_rank, msg in zip(comm_list, msgs):
            op_list.append(self._p2p_batch_op(dist.isend, msg, dst_rank))
        handles = self._p2p_batch_op_execute(op_list=op_list)
        return handles

    def sink_p2p_message_batch_async(self, comm_list:List[int], msgs:List[Tensor]):
        """
        batch async send msg - P2POps and batch_isend_irecv()
        sink msgs to comm_list
        """
        self.set_p2p_comm_group(comm_list)
        op_list = []

        for dst_rank in comm_list:
            for msg in msgs:
                op_list.append(self._p2p_batch_op(dist.isend, msg, dst_rank))
        handles = self._p2p_batch_op_execute(op_list=op_list)
        return handles

    def read_p2p_message_batch_async(self, per_msg_size = 1, per_msg_shape = [(1,)]) -> Tuple[List[Tuple[int, List[th.Tensor]]], None]:
        """
        batch async recv msg - P2POps and batch_isend_irecv()
        """
        src_list = self.get_p2p_comm_group()
        msgs = {}
        r_msgs = []
        op_list = []
        for src_rank in src_list:
            msgs[src_rank] = [th.zeros(a_shape) for a_shape in per_msg_shape]
            for i in range(per_msg_size):
                op_list.append(self._p2p_batch_op(dist.irecv, msgs[src_rank][i], src_rank))
            r_msgs.append((src_rank, msgs[src_rank]))
        handles = self._p2p_batch_op_execute(op_list=op_list)
        return r_msgs, handles

    # @deprecated use barrier() instead.
    #def wait_p2p_comm_notify(self, all_rank_list:List[int]):
    #    key_list = ["p2p_done_{rank}".format(rank = rank) for rank in all_rank_list]
    #    self._tcp_store_wait(key_list)

    #def get_p2p_comm_notify(self, all_rank_list:List[int]):
    #    key_list = ["p2p_done_{rank}".format(rank = rank) for rank in all_rank_list]
    #    for key in key_list:
    #        self._tcp_store_get(key)

    #def set_p2p_comm_notify(self):
    #    self._tcp_store_set("p2p_done_{rank}".format(rank = self._rank), "done")

    #def reset_p2p_comm_notify(self):
    #    self._tcp_store_delete("p2p_done_{rank}".format(rank = self._rank))


    #################################################
    '''
    Broadcast
    '''
    #################################################
    def set_broadcast_comm_group(self, comm_list):
        self._tcp_store_set("broadcast_{rank}".format(rank = self._rank),",".join(map(lambda x: str(x),comm_list)))

    def get_broadcast_comm_group(self):
        src_list = []
        for src_rank in range(self._world_size):
            dst_list = self._tcp_store_get("broadcast_{rank}".format(rank = src_rank)).split(",")
            if dst_list == [""]:
                continue
            dst_list = map(lambda x:int(x), dst_list)
            if self._rank in dst_list:
                src_list.append(src_rank)
        return src_list

    def reset_broadcast_comm_group(self):
        self._tcp_store_set("broadcast_{rank}".format(rank = self._rank),"")


    # collective comm functions such as broadcast() dont meet my requiremnts on group comm
    # so i implement broadcast by isend() and recv()
    def write_group_message(self, comm_list:List[int], msg:Tensor):
        """
        write msg to broadcast by async send - isend()
        """
        self.set_broadcast_comm_group(comm_list)
        handles = [self._async_send(msg, dst_rank)  
                for dst_rank in comm_list]
        return handles

    def read_group_message(self):
        """
        receive msg from src by sync recv - recv()
        """
        src_list = self.get_broadcast_comm_group()
        msgs = []
        temp = th.zeros(1)
        for src_rank in src_list:
            self._sync_recv(temp, src_rank)
            # deepcopy is necessary cause recv() changes tmep every calls,
            # leading msgs to be [last_recv, last_recv, ...]
            msgs.append((src_rank, copy.deepcopy(temp)))
        return msgs

    def process_wait(self):
        self._dist.barrier()


if __name__ == '__main__':
    dist_comm = DistributedComm("127.0.0.1", "29600","127.0.0.1","29601",0,1)
