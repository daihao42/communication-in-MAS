#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn

from typing import List

from utils.distributed import DistributedComm

import numpy as np

class ModelParallel():

    def __init__(self, model:nn.Module) -> None:
        self.model = model
        self.p_shape = [x.shape for x in self.model.parameters()]

    def _parameter_decode(self):
        self.parameters = [x.cpu().detach() for x in self.model.parameters()]
        return self.parameters

    def _parameter_update(self, new_params : List[torch.Tensor]):
        for param, new_param in zip(self.model.parameters(), new_params):
            param.data = nn.parameter.Parameter(new_param)

    def _parameter_avg(self, parameters: List[List[torch.Tensor]]):
        return np.average(list(map(lambda x:[y.numpy() for y in x], parameters)),axis=0)

    def sink_parameter(self, dist_comm : DistributedComm, dst_ranks : List[int]):
        ps = self._parameter_decode()
        return dist_comm.sink_p2p_message_batch_async(dst_ranks, ps)

    def recv_parameter(self, dist_comm : DistributedComm):
        msgs,_ = dist_comm.read_p2p_message_batch_async(per_msg_size=len(self.p_shape),
                                               per_msg_shape=self.p_shape)
        return msgs

    def fed_avg(self, dist_comm : DistributedComm, dst_ranks : List[int]):
        self.sink_parameter(dist_comm, dst_ranks)
        dist_comm.process_wait()
        msgs = self.recv_parameter(dist_comm)
        pss = list(map(lambda x:x[1], msgs))
        self._parameter_update(self._parameter_avg(pss))
