#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import os

if __name__ == '__main__':
    #pytest.main(["-s","-v","tests/test_distributed.py"])
    pytest.main(["-s","-v","tests/test_model_parallel.py"])
    #os.system("python tests/test_train.py 0")
    #os.system("python tests/test_train.py 1")
    #os.system("python tests/test_train.py 2")
 
