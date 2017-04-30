#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:48:32 2017

@author: mcomin
"""
from LSGAN import *

M = Model(bs=50,n=82500)

M.Train(epochs=20)
