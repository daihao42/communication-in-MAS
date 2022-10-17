#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import os, sys
import torch

if __name__ == '__main__':
    pytest.main(["-s","-v","tests/test_actor.py"])
 
