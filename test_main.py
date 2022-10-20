#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import os, sys
import torch

if __name__ == '__main__':
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #pytest.main(["-s","-v","tests/test_distributed.py"])
    #pytest.main(["-s","-v","tests/test_model_parallel.py"])
    #pytest.main(["-s","-v","tests/test_environments.py"])
    pytest.main(["-s","-v","tests/test_logs.py"])
    #pytest.main(["-s","-v","tests/test_agent.py"])
    #pytest.main(["-s","-v","tests/test_learner.py"])
    #pytest.main(["-s","-v","tests/test_algorithms.py"])
 
