#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 16:05:10 2020

@author: muhammad
"""

import PDE
import torch
from torch import tensor

A = torch.eye(2)
B = tensor([[1.],[1.]])
C = tensor([[1., 1.]])

Q = torch.eye(2)
R = tensor([[1.]])

kalman = PDE.KalmanFilter(A,B,C,Q,R)
kalman.propagate(torch.randn(1,1),torch.randn(1))