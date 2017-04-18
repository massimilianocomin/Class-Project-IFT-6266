#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:48:32 2017

@author: mcomin
"""
from ConvAE import *

M = Model(bs=5,n=100)

M.Train(epochs=30)
