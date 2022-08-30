#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

if __name__ == '__main__':
    #pytest.main(["-s","-v","tests/test_distributed.py"])
    pytest.main(["-s","-v","tests/test_model_parallel.py"])
 
