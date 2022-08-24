#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from utils.distributed import DistributedComm


import torch

import torch.multiprocessing as mp

import os, datetime, time

import copy

def mtest_init():
    """
    test if one process initial will be ok.
    """
    dist_comm = DistributedComm("127.0.0.1", "29700","127.0.0.1","29701",0,1)
    assert dist_comm._tcp_store_get("p2p_{rank}".format(rank = 0)) == "" , \
        "initial p2p commucation_channel failed with single process."
    assert dist_comm._tcp_store_get("broadcast_{rank}".format(rank = 0)) == "" , \
        "initial broadcast commucation_channel failed with single process."
    dist_comm._dist.destroy_process_group()

def _multi_processes_distributed_wrapper(master_ip, master_port, tcp_store_ip, tcp_store_port, world_size, rank, func):
    """
    run func on multiple local distributed commucation processes
    """
    dist_comm = DistributedComm(master_ip, master_port, tcp_store_ip, tcp_store_port, rank, world_size)
    func(dist_comm, world_size)

def _multi_processes_wrapper(world_size, func):
    """
    multiple processes managenment
    """
    #mp.set_start_method("spawn")
    mp.set_start_method('forkserver', force=True)
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=_multi_processes_distributed_wrapper, args=("127.0.0.1", "29700","127.0.0.1","29701", world_size, rank, func))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    #for p in processes:
    #    p.close()

def _check_channel_init(dist_comm, world_size):
    """
    chech if channels have been initialed with ""
    """
    assert dist_comm._tcp_store_get("p2p_{rank}".format(rank = 0)) == "" , \
        "initial p2p commucation_channel failed on rank = 0."
    assert dist_comm._tcp_store_get("p2p_{rank}".format(rank = world_size-1)) == "" , \
        "initial p2p commucation_channel failed on rank world_size - 1."
    assert dist_comm._tcp_store_get("broadcast_{rank}".format(rank = 0)) == "" , \
        "initial broadcast commucation_channel failed on rank = 0."
    assert dist_comm._tcp_store_get("broadcast_{rank}".format(rank = world_size-1)) == "" , \
        "initial broadcast commucation_channel failed on rank world_size - 1."

def test_multi_process_init():
    """
    call _check_channel_init
    """
    _multi_processes_wrapper(world_size=8, func = _check_channel_init)

def _p2p_set_and_get_comm_group(dist_comm, world_size):
    """
    test on p2p set commucation group, by storing list into tcpstore
    p2p_dst: src1,src2,...
    """
    #comm_list = [i for i in range(dist_comm._rank)]
    comm_list = list(range(world_size))
    dist_comm.set_p2p_comm_group(comm_list)

    # wait for all setting finished
    #dist_comm._dist.barrier()
    #time.sleep(2)

    res =  dist_comm.get_p2p_comm_group()
    print(dist_comm._rank, comm_list, res)
    #assert res == comm_list, \
    #    "p2p set commucation group failed"

    #dist_comm._dist.barrier()
    dist_comm.reset_p2p_comm_group()

    assert dist_comm.get_p2p_comm_group() == [], \
        "p2p reset commucation group failed"

def error_test_p2p_group_set_and_get():
    """
    call _set_and_get
    error because no barrier() to gurrante the process sync
    """
    _multi_processes_wrapper(world_size=8, func = _p2p_set_and_get_comm_group)


def _p2p_set_and_barrier(dist_comm, world_size):
    """
    write - notify_set - read - notify_reset
    """
    # before set, it's necessary to call barrier(), 
    # because rank 0 will do reset on all p2p group,
    # this gonna happened after other rank != 0 process call set_p2p_comm_group()
    #dist_comm._dist.barrier()
    dist_comm.process_wait()

    comm_list = list(range(world_size))
    #all_rank_list = list(range(world_size))
    dist_comm.set_p2p_comm_group(comm_list)

    #dist_comm._dist.barrier()
    dist_comm.process_wait()

    # set notify done
    #dist_comm.set_p2p_comm_notify()

    # wait for all notify done
    #dist_comm.wait_p2p_comm_notify(all_rank_list)

    res =  dist_comm.get_p2p_comm_group()
    print(dist_comm._rank, comm_list, res)
    assert res == comm_list, \
        "p2p set commucation group failed"

    #dist_comm.reset_p2p_comm_notify()

    #dist_comm._dist.barrier()
    dist_comm.process_wait()

    '''
    0 -> 1,3,5
    1 -> 2,5,6
    2 -> 3
    3 -> 1,4
    4 -> 6
    5 -> 2,3
    6 -> 0,1,2,3,4,5,7
    7 -> 3,4,5

    convert to

    0 <- 6
    1 <- 0,3,6
    2 <- 1,5,6
    3 <- 0,2,5,6,7
    4 <- 3,6,7
    5 <- 0,1,6,7
    6 <- 1,4
    7 <- 6
    '''
    dst_dict = {
        0 : [1,3,6],
        1 : [2,5,6],
        2 : [3],
        3 : [1,4],
        4 : [6],
        5 : [2,3],
        6 : [0,1,2,3,4,5,7],
        7 : [3,4,5],
    }

    dist_comm.set_p2p_comm_group(dst_dict[dist_comm._rank])

    #dist_comm._dist.barrier()
    dist_comm.process_wait()

    src_dict = {
        0 : [6],
        1 : [0,3,6],
        2 : [1,5,6],
        3 : [0,2,5,6,7],
        4 : [3,6,7],
        5 : [1,6,7],
        6 : [0,1,4],
        7 : [6],
    }

    res =  dist_comm.get_p2p_comm_group()
    print(dist_comm._rank, src_dict[dist_comm._rank], res)
    assert res == src_dict[dist_comm._rank], \
        "p2p set random commucation group failed"

    #dist_comm._dist.barrier()
    dist_comm.process_wait()

    dist_comm.reset_p2p_comm_group()

    #dist_comm._dist.barrier()
    dist_comm.process_wait()

    assert dist_comm.get_p2p_comm_group() == [], \
        "p2p reset commucation group failed"


def test_p2p_group_set_and_wait():
    """
    call _p2p_set_and_wait
    """
    _multi_processes_wrapper(world_size=8, func = _p2p_set_and_barrier)

def _broadcast_set_and_get(dist_comm, world_size):
    """
    test on broadcast set commucation group, by storing list into tcpstore
    broadcast_dst: src1,src2,...
    """
    #dist_comm._dist.barrier()
    dist_comm.process_wait()

    comm_list = list(range(world_size))
    dist_comm.set_broadcast_comm_group(comm_list)

    dist_comm.process_wait()
    res =  dist_comm.get_broadcast_comm_group()
    print(dist_comm._rank, comm_list, res)
    assert res == comm_list, \
        "broadcast set commucation group failed"

    #dist_comm._dist.barrier()
    dist_comm.process_wait()

    dist_comm.reset_broadcast_comm_group()

    #dist_comm._dist.barrier()
    dist_comm.process_wait()

    assert dist_comm.get_broadcast_comm_group() == [], \
        "broadcast reset commucation group failed"

def test_broadcast_notify_set_and_wait():
    """
    call _broadcast_set_and_wait
    """
    _multi_processes_wrapper(world_size=8, func = _broadcast_set_and_get)


def _dist_send_and_recv(dist_comm, world_size):
    """
    test on the dist.send() and dist.recv()
    """
    #dist_comm._dist.barrier()
    dist_comm.process_wait()

    send = torch.zeros(2)

    recv = torch.zeros(2)

    hds = [dist_comm._async_send(send + dist_comm._rank + i, dst = i) 
           if i != dist_comm._rank else None for i in range(world_size)]
    print("rank {} call 2".format(dist_comm._rank))

    #dist_comm._dist.barrier()
    dist_comm.process_wait()

    print("rank {} call 3".format(dist_comm._rank))
    for i in range(world_size):
        if i != dist_comm._rank:
            #src_i = copy.deepcopy(i)
            #print("rank {} recv from {}".format(dist_comm._rank, i))
            dist_comm._sync_recv(recv, src = i)
            print("rank {} receive recv={} from {}".format(dist_comm._rank, recv, i))

    #dist_comm._dist.barrier()
    dist_comm.process_wait()


def test_send_and_recv():
    """
    call _dist_send_and_recv
    """
    _multi_processes_wrapper(world_size=16, func = _dist_send_and_recv)


def _p2p_write_and_read(dist_comm, world_size):
    """
    test on the distributed read and write through p2p isend/recv
    """
    #dist_comm._dist.barrier()
    dist_comm.process_wait()

    '''
    0 -> 1,3,5
    1 -> 2,5,6
    2 -> 3
    3 -> 1,4
    4 -> 6
    5 -> 2,3
    6 -> 0,1,2,3,4,5,7
    7 -> 3,4,5

    convert to

    0 <- 6
    1 <- 0,3,6
    2 <- 1,5,6
    3 <- 0,2,5,6,7
    4 <- 3,6,7
    5 <- 0,1,6,7
    6 <- 1,4
    7 <- 6
    '''
    dst_dict = {
        0 : [1,3,6],
        1 : [2,5,6],
        2 : [3],
        3 : [1,4],
        4 : [6],
        5 : [2,3],
        6 : [0,1,2,3,4,5,7],
        7 : [3,4,5],
    }

    hds = dist_comm.write_p2p_message(dst_dict[dist_comm._rank],
                                list(map(lambda x:torch.zeros(1)+dist_comm._rank,
                                         dst_dict[dist_comm._rank])))
    #dist_comm._dist.barrier()
    dist_comm.process_wait()

    print("=======call after write =========", dist_comm._rank)

    src_dict = {
        0 : [6],
        1 : [0,3,6],
        2 : [1,5,6],
        3 : [0,2,5,6,7],
        4 : [3,6,7],
        5 : [1,6,7],
        6 : [0,1,4],
        7 : [6],
    }

    res =  dist_comm.get_p2p_comm_group()
    print(dist_comm._rank, src_dict[dist_comm._rank], res)
    assert res == src_dict[dist_comm._rank], \
        "p2p set random commucation group failed"

    print("=======call before read =========", dist_comm._rank)

    res =  dist_comm.read_p2p_message()

    src_res = list(map(lambda x:(x,torch.tensor(x)), src_dict[dist_comm._rank]))
    print(dist_comm._rank, src_res, res)
    assert res == src_res, \
        "p2p random write and read failed"

    #dist_comm._dist.barrier()
    dist_comm.process_wait()

def test_p2p_write_and_read():
    """
    call _broadcast_set_and_wait
    """
    _multi_processes_wrapper(world_size=8, func = _p2p_write_and_read)

def _group_write_and_read(dist_comm, world_size):
    """
    test on the distributed read and write through isend/recv
    """
    #dist_comm._dist.barrier()
    dist_comm.process_wait()

    '''
    0 -> 1,3,5
    1 -> 2,5,6
    2 -> 3
    3 -> 1,4
    4 -> 6
    5 -> 2,3
    6 -> 0,1,2,3,4,5,7
    7 -> 3,4,5

    convert to

    0 <- 6
    1 <- 0,3,6
    2 <- 1,5,6
    3 <- 0,2,5,6,7
    4 <- 3,6,7
    5 <- 0,1,6,7
    6 <- 1,4
    7 <- 6
    '''
    dst_dict = {
        0 : [1,3,6],
        1 : [2,5,6],
        2 : [3],
        3 : [1,4],
        4 : [6],
        5 : [2,3],
        6 : [0,1,2,3,4,5,7],
        7 : [3,4,5],
    }

    hds = dist_comm.write_group_message(dst_dict[dist_comm._rank],torch.zeros(1)+dist_comm._rank)
    #dist_comm._dist.barrier()
    dist_comm.process_wait()

    print("=======call after write =========", dist_comm._rank)

    src_dict = {
        0 : [6],
        1 : [0,3,6],
        2 : [1,5,6],
        3 : [0,2,5,6,7],
        4 : [3,6,7],
        5 : [1,6,7],
        6 : [0,1,4],
        7 : [6],
    }

    res =  dist_comm.get_broadcast_comm_group()
    print(dist_comm._rank, src_dict[dist_comm._rank], res)
    assert res == src_dict[dist_comm._rank], \
        "group set random commucation group failed"

    print("=======call before read =========", dist_comm._rank)

    res =  dist_comm.read_group_message()

    src_res = list(map(lambda x:(x,torch.tensor(x)), src_dict[dist_comm._rank]))
    print(dist_comm._rank, src_res, res)
    assert res == src_res, \
        "group random write and read failed"

    #dist_comm._dist.barrier()
    dist_comm.process_wait()

def test_group_write_and_read():
    """
    call _broadcast_set_and_wait
    """
    _multi_processes_wrapper(world_size=8, func = _group_write_and_read)


if __name__ == '__main__':
    pytest.main(["-s","-v","test_distributed.py"])

