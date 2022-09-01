#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn

from typing import List

from utils.distributed import DistributedComm

import numpy as np

class ModelParallel():

    def __init__(self, model:nn.Module, device) -> None:
        self._model = model
        self.p_shape = [x.shape for x in self._model.parameters()]
        self.device = device

    @property
    def model(self):
        return self._model

    def _parameter_decode(self):
        self.parameters = [x.cpu().detach() for x in self._model.parameters()]
        return self.parameters

    def _parameter_update(self, new_params : List[torch.Tensor]):
        for param, new_param in zip(self._model.parameters(), new_params):
            param.data = nn.parameter.Parameter(new_param)
        self._model.to(self.device)

    def _parameter_avg(self, parameters: List[List[torch.Tensor]]):
        r"""
            @parameters : [[t0_0, t0_1, ...], [t1_0, t1_1, ...]]
            t0 : device 0
            t0_1 : parameter of 1-th layer of DNN on device 0
        """
        t_param = [[parameters[j][i] for j in range(len(parameters))] 
                   for i in range(len(parameters[0]))]
        return [torch.mean(torch.stack(tp,dim=0),dim=0) for tp in t_param]

    def sink_parameter(self, dist_comm : DistributedComm, dst_ranks : List[int]):
        r"""
        return handle is extremely necessary, otherwise process will block!
        """
        ps = self._parameter_decode()
        hds = dist_comm.sink_p2p_message_batch_async(dst_ranks, ps)
        return hds

    def recv_parameter(self, dist_comm : DistributedComm):
        r"""
        return handle is extremely necessary, otherwise process will block!
        """
        msgs,handle = dist_comm.read_p2p_message_batch_async(per_msg_size=len(self.p_shape),
                                               per_msg_shape=self.p_shape)
        return msgs,handle

    def fed_avg(self, dist_comm : DistributedComm, dst_ranks : List[int]):
        dist_comm.process_wait()
        #print(f"before {dist_comm._rank} call sink")
        hds = self.sink_parameter(dist_comm, dst_ranks)
        #print(f"{dist_comm._rank} have called sink and before call barrier")
        dist_comm.process_wait()
        #print(f"{dist_comm._rank} have called barrier and before call recv")
        msgs,handles = self.recv_parameter(dist_comm)
        #print(f"{dist_comm._rank} have called recv and before call barrier 2")
        #dist_comm.process_wait()
        pss = list(map(lambda x:x[1], msgs))
        self._parameter_update(self._parameter_avg(pss))
        return msgs,hds
