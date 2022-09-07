#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'dai'

import logging

import torch

from torch import nn

from torch.utils.tensorboard.writer import SummaryWriter

import os,time

from torchsummary import summary

from typing import Tuple

class Logger():

    def __init__(self, experimental_name:str, model_name:str, rank_id:int, log_dir:str="./logs") -> None:
        self.rank_id = rank_id

        log_dir = os.path.join(log_dir, experimental_name, model_name, str(int(time.time())))

        os.makedirs(log_dir, exist_ok=True)

        os.makedirs(os.path.join(log_dir,"logs"), exist_ok=True)

        self.writer = SummaryWriter(os.path.join(log_dir,"tensorboard"), str(rank_id))

        logging.basicConfig(level=logging.DEBUG #set logging level, all log beyond level will output, DEBUG < INFO < WARN < ERROR. 
                             ,filename=os.path.join(log_dir,"logs", f"{rank_id}.log") #output filename
                             ,filemode="w" #write mode, w denotes rewrite, a denotes apppend
                             ,format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s" #output format
                             # -8 denotes left alignment with 8 space
                             ,datefmt="%Y-%m-%d %H:%M:%S" #output time format
                             )

    def INFO(self, content):
        logging.info(content)

    def WARN(self, content):
        logging.warning(content)

    def ERROR(self, content):
        logging.error(content)

    def DEBUG(self, content):
        logging.debug(content)

    def model_summary(self, model:nn.Module, input_size:Tuple):
        self.INFO("=============================\n")
        self.INFO(str(model.__class__)[8:-2] + "\n")
        self.INFO(summary(model,input_size))

    def add_scalar(self, tag, value, global_step):
        self.writer.add_scalar(f"{tag}/rank_{self.rank_id}\\", value, global_step)
        self.writer.flush()

    def add_graph(self, model:nn.Module, input_size:Tuple):
        fake = torch.randn(input_size)
        self.writer.add_graph(model, fake)
        self.writer.flush()


if __name__ == '__main__':
    logger = Logger("test_logs","test_actor",0)
