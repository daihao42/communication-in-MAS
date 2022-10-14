#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from utils.distributed import DistributedComm


import torch

import torch.multiprocessing as mp

import os, datetime, time

import copy

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# cuda out of memory
#device_backend = [("cpu", "gloo"), ("cuda:0", "nccl")]
device_backend = [("cpu", "gloo")]

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

def _multi_processes_distributed_wrapper(master_ip, master_port, tcp_store_ip, tcp_store_port, world_size, rank, func, device, backend):
    """
    run func on multiple local distributed commucation processes
    """
    dist_comm = DistributedComm(master_ip, master_port, tcp_store_ip, tcp_store_port, rank, world_size, backend=backend)
    func(dist_comm, world_size, device=device)

def _multi_processes_wrapper(world_size, func, device="cpu", backend="gloo"):
    """
    multiple processes managenment
    """
    #mp.set_start_method("spawn")
    mp.set_start_method('forkserver', force=True)
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=_multi_processes_distributed_wrapper, args=("127.0.0.1", "29700","127.0.0.1","29701", world_size, rank, func, device, backend))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    #for p in processes:
    #    p.close()


def _check_channel_init(dist_comm, world_size, device):
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

@pytest.mark.parametrize("device, backend", device_backend)
def test_multi_process_init(device, backend):
    """
    call _check_channel_init
    """
    if(backend == "gloo"):
        _multi_processes_wrapper(world_size=8, func = _check_channel_init, device=device, backend=backend)
    elif(backend == "nccl"):
        _multi_processes_wrapper(world_size=3, func = _check_channel_init, device=device, backend=backend)

def _p2p_set_and_get_comm_group(dist_comm, world_size, device):
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


def _p2p_set_and_barrier(dist_comm, world_size, device):
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

@pytest.mark.parametrize("device, backend", device_backend)
def test_p2p_group_set_and_wait(device, backend):
    """
    call _p2p_set_and_wait
    """
    if backend == "gloo":
        _multi_processes_wrapper(world_size=8, func = _p2p_set_and_barrier, device=device, backend=backend)
    elif backend == "nccl":
        _multi_processes_wrapper(world_size=3, func = _p2p_set_and_barrier, device=device, backend=backend)

def _broadcast_set_and_get(dist_comm, world_size, device):
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

@pytest.mark.parametrize("device, backend", device_backend)
def test_broadcast_notify_set_and_wait(device, backend):
    """
    call _broadcast_set_and_wait
    """
    if backend == "gloo":
        _multi_processes_wrapper(world_size=8, func = _broadcast_set_and_get, device=device, backend=backend)
    elif backend == "nccl":
        _multi_processes_wrapper(world_size=3, func = _broadcast_set_and_get, device=device, backend=backend)


def _dist_send_and_recv(dist_comm, world_size, device):
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

@pytest.mark.parametrize("device, backend", device_backend)
def test_send_and_recv(device, backend):
    """
    call _dist_send_and_recv
    """
    if backend == "gloo":
        _multi_processes_wrapper(world_size=8, func = _dist_send_and_recv, device=device, backend=backend)
    elif backend == "nccl":
        _multi_processes_wrapper(world_size=3, func = _dist_send_and_recv, device=device, backend=backend)

def _p2p_write_and_read(dist_comm, world_size, device):
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

@pytest.mark.parametrize("device, backend", device_backend)
def test_p2p_write_and_read(device, backend):
    """
    call _broadcast_set_and_wait
    """
    if backend == "gloo":
        _multi_processes_wrapper(world_size=8, func = _p2p_write_and_read, device=device, backend=backend)
    elif backend == "nccl":
        _multi_processes_wrapper(world_size=3, func = _p2p_write_and_read, device=device, backend=backend)

def _p2p_write_and_read_batch(dist_comm, world_size, device):
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

    hds = dist_comm.write_p2p_message_batch_async(dst_dict[dist_comm._rank],
                                list(map(lambda x:torch.zeros(1)+dist_comm._rank,
                                         dst_dict[dist_comm._rank])))
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

    res,_,_ =  dist_comm.read_p2p_message_batch_async()

    src_res = list(map(lambda x:(x,[torch.tensor(x)]), src_dict[dist_comm._rank]))
    print(dist_comm._rank, src_res, res)
    assert res == src_res, \
        "p2p random write and read failed"

    #dist_comm._dist.barrier()
    dist_comm.process_wait()

@pytest.mark.parametrize("device, backend", device_backend)
def test_p2p_write_and_read_batch(device, backend):
    """
    call _broadcast_set_and_wait
    """
    if backend == "gloo":
        _multi_processes_wrapper(world_size=8, func = _p2p_write_and_read_batch, device=device, backend=backend)
    elif backend == "nccl":
        _multi_processes_wrapper(world_size=3, func = _p2p_write_and_read_batch, device=device, backend=backend)

def _p2p_sink_and_read_batch(dist_comm, world_size, device):
    """
    test on the distributed read and write through p2p isend/recv
    """
    #dist_comm._dist.barrier()
    dist_comm.process_wait()

    '''
    0 -> [[10,20,30], [100,200,300], [1000,2000,3000]]
    1 -> [[11,21,31], [101,201,301], [1001,2001,3001]]
    2 -> [[12,22,32], [102,202,302], [1002,2002,3002]]
    3 -> [[13,23,33], [103,203,303], [1003,2003,3003]]
    4 -> [[14,24,34], [104,204,304], [1004,2004,3004]]
    5 -> [[15,25,35], [105,205,305], [1005,2005,3005]]
    6 -> [[16,26,36], [106,206,306], [1006,2006,3006]]
    7 -> [[17,27,37], [107,207,307], [1007,2007,3007]]

    '''
    param_dict = {
            0 : [torch.Tensor([10,20,30]), torch.Tensor([100,200,300]), torch.Tensor([1000,2000,3000])],
            1 : [torch.Tensor([11,21,31]), torch.Tensor([101,201,301]), torch.Tensor([1001,2001,3001])],
            2 : [torch.Tensor([12,22,32]), torch.Tensor([102,202,302]), torch.Tensor([1002,2002,3002])],
            3 : [torch.Tensor([13,23,33]), torch.Tensor([103,203,303]), torch.Tensor([1003,2003,3003])],
            4 : [torch.Tensor([14,24,34]), torch.Tensor([104,204,304]), torch.Tensor([1004,2004,3004])],
            5 : [torch.Tensor([15,25,35]), torch.Tensor([105,205,305]), torch.Tensor([1005,2005,3005])],
            6 : [torch.Tensor([16,26,36]), torch.Tensor([106,206,306]), torch.Tensor([1006,2006,3006])],
            7 : [torch.Tensor([17,27,37]), torch.Tensor([107,207,307]), torch.Tensor([1007,2007,3007])]
    }

    dst_list = [(dist_comm._rank + x) % world_size for x in range(1,4)]

    hds = dist_comm.sink_p2p_message_batch_async(dst_list, param_dict[dist_comm._rank])

    dist_comm.process_wait()

    src_list = [(dist_comm._rank - x + world_size) % world_size for x in range(1,4)]

    res_comm_group =  dist_comm.get_p2p_comm_group()

    print(f"rank {dist_comm._rank} read p2p group {res_comm_group}, src_list {src_list}")

    assert set(res_comm_group) == set(src_list), \
        "sink set comm group failed"

    res,_,handle = dist_comm.read_p2p_message_batch_async(per_msg_size=len(param_dict[dist_comm._rank]),
                                                 per_msg_shape=[(3,),(3,),(3,)])

    src_res = list(map(lambda x:(x,param_dict[x]), src_list))
    print(dist_comm._rank, src_res, res)

    assert res.sort() == src_res.sort(), \
        "sink and read failed"

    #dist_comm._dist.barrier()
    dist_comm.process_wait()

@pytest.mark.parametrize("device, backend", device_backend)
def test_p2p_sink_and_read_batch(device, backend):
    """
    call _broadcast_set_and_wait
    """
    if backend == "gloo":
        _multi_processes_wrapper(world_size=8, func = _p2p_sink_and_read_batch, device=device, backend=backend)
    elif backend == "nccl":
        _multi_processes_wrapper(world_size=3, func = _p2p_sink_and_read_batch, device=device, backend=backend)

def _p2p_sink_and_read_batch_repeat(dist_comm, world_size, device):
    """
    test on the distributed read and write through p2p isend/recv
    """
    #dist_comm._dist.barrier()
    dist_comm.process_wait()

    '''
    0 -> [[10,20,30], [100,200,300], [1000,2000,3000]]
    1 -> [[11,21,31], [101,201,301], [1001,2001,3001]]
    2 -> [[12,22,32], [102,202,302], [1002,2002,3002]]
    3 -> [[13,23,33], [103,203,303], [1003,2003,3003]]
    4 -> [[14,24,34], [104,204,304], [1004,2004,3004]]
    5 -> [[15,25,35], [105,205,305], [1005,2005,3005]]
    6 -> [[16,26,36], [106,206,306], [1006,2006,3006]]
    7 -> [[17,27,37], [107,207,307], [1007,2007,3007]]

    '''
    param_dict = {
            0 : [torch.Tensor([10,20,30]), torch.Tensor([100,200,300]), torch.Tensor([1000,2000,3000])],
            1 : [torch.Tensor([11,21,31]), torch.Tensor([101,201,301]), torch.Tensor([1001,2001,3001])],
            2 : [torch.Tensor([12,22,32]), torch.Tensor([102,202,302]), torch.Tensor([1002,2002,3002])],
            3 : [torch.Tensor([13,23,33]), torch.Tensor([103,203,303]), torch.Tensor([1003,2003,3003])],
            4 : [torch.Tensor([14,24,34]), torch.Tensor([104,204,304]), torch.Tensor([1004,2004,3004])],
            5 : [torch.Tensor([15,25,35]), torch.Tensor([105,205,305]), torch.Tensor([1005,2005,3005])],
            6 : [torch.Tensor([16,26,36]), torch.Tensor([106,206,306]), torch.Tensor([1006,2006,3006])],
            7 : [torch.Tensor([17,27,37]), torch.Tensor([107,207,307]), torch.Tensor([1007,2007,3007])]
    }

    dst_list = [(dist_comm._rank + x) % world_size for x in range(1,4)]

    for i in range(3):

        hds = dist_comm.sink_p2p_message_batch_async(dst_list, param_dict[dist_comm._rank])

        dist_comm.process_wait()

        src_list = [(dist_comm._rank - x + world_size) % world_size for x in range(1,4)]

        res_comm_group =  dist_comm.get_p2p_comm_group()

        print(f"rank {dist_comm._rank} read p2p group {res_comm_group}, src_list {src_list}")

        assert set(res_comm_group) == set(src_list), \
            "sink set comm group failed"

        res,_,handle = dist_comm.read_p2p_message_batch_async(per_msg_size=len(param_dict[dist_comm._rank]),
                                                     per_msg_shape=[(3,),(3,),(3,)])

        src_res = list(map(lambda x:(x,param_dict[x]), src_list))
        print(dist_comm._rank, src_res, res)

        assert res.sort() == src_res.sort(), \
            "sink and read failed"

        #dist_comm._dist.barrier()
        dist_comm.process_wait()

@pytest.mark.parametrize("device, backend", device_backend)
def test_p2p_sink_and_read_batch_repeat(device, backend):
    """
    call _broadcast_set_and_wait
    """
    if backend == "gloo":
        _multi_processes_wrapper(world_size=8, func = _p2p_sink_and_read_batch_repeat, device=device, backend=backend)
    elif backend == "nccl":
        _multi_processes_wrapper(world_size=3, func = _p2p_sink_and_read_batch_repeat, device=device, backend=backend)

def _group_write_and_read(dist_comm, world_size, device):
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

@pytest.mark.parametrize("device, backend", device_backend)
def test_group_write_and_read(device, backend):
    """
    call _broadcast_set_and_wait
    """
    if backend == "gloo":
        _multi_processes_wrapper(world_size=8, func = _group_write_and_read, device=device, backend=backend)
    elif backend == "nccl":
        _multi_processes_wrapper(world_size=3, func = _group_write_and_read, device=device, backend=backend)

if __name__ == '__main__':
    pytest.main(["-s","-v","test_distributed.py"])

