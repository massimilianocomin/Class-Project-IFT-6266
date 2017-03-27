#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:46:32 2017

@author: mcomin
"""
import sys
sys.path.append('../lib')

import numpy as np
import pickle
import os
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from Img import *
from Layers import *

theano.config.floatX = 'float32'
theano.config.intX = 'int32'

class ConvolutionalAutoEncoder(Loss):
    """
    Implementation of a ConvAE.
    """
    def __init__(self):

        print('Building computational graph...')
        
        self.ks = (5,5) # Kernel size
        self.bs = 50 # Batch size
        
        x = T.tensor4('inputs') # Cropped
        y = T.tensor4('targets') # Full
        
        loss = self.build(x,y)

        updat = Tool.adam(loss, self.params)

        self.train = theano.function([x,y],loss,updates=updat)
        self.predict = theano.function([x],self.Y)
        
        print('Computational graph built.')

    def build(self,x,y):

        L = {}
        x = x.reshape((x.shape[0],3,64,64))
        y = y.reshape((y.shape[0],3,64,64))
        self.params = []
        
        args = lambda x,y : {'nchan':x, 'nkernels':y, 'kernelsize':self.ks,
                              'activation':'sigmoid', 'pad':2,'pmode':'average_inc_pad'}
        
        L[1] = ConvLayer(x, **args(3,10), poolsize = (1,1))
        L[2] = ConvLayer(L[1].output, **args(10,12))
        L[3] = ConvLayer(L[2].output, **args(12,14), poolsize = (1,1))
        L[4] = ConvLayer(L[3].output, **args(14,16))
        L[5] = ConvLayer(L[4].output, **args(16,18), poolsize = (1,1))
        L[6] = ConvLayer(L[5].output, **args(18,20))
        
        sizeout = L[6].shape(L[5].shape(L[4].shape(L[3].shape(L[2].shape(L[1].shape(64))))))

        L[7] = DenseLayer(L[6].output, 20*sizeout**2, 10*sizeout**2, 'tanh')
        L[8] = DenseLayer(L[7].output, 10*sizeout**2, 5*sizeout**2, 'tanh')
        L[9] = DenseLayer(L[8].output, 5*sizeout**2, 10*sizeout**2, 'tanh')
        L[10] = DenseLayer(L[9].output, 10*sizeout**2, 20*sizeout**2, 'tanh')

        sizein = (L[10].output.shape[0],20,sizeout,sizeout)

        L[11] = TConvLayer(L[10].output.reshape(sizein), **L[6].arguments, shape = (16,16), batch=self.bs, tied=False)
        L[12] = TConvLayer(L[11].output, **L[5].arguments, shape = (16,16), batch=self.bs)
        L[13] = TConvLayer(L[12].output, **L[4].arguments, shape = (32,32), batch=self.bs, tied=False)
        L[14] = TConvLayer(L[13].output, **L[3].arguments, shape = (32,32), batch=self.bs)
        L[15] = TConvLayer(L[14].output, **L[2].arguments, shape = (64,64), batch=self.bs, tied=False)
        L[16] = TConvLayer(L[15].output, **L[1].arguments, shape = (64,64), batch=self.bs)

        self.Y = L[16].output

        for i in L.keys():
            self.params += L[i].params

        Y_flat = self.Y.reshape((y.shape[0],3,64*64))
        y_flat = y.reshape((y.shape[0],3,64*64))

        return T.mean(T.sum( T.pow( y_flat - Y_flat, 2), axis=[1,2])) 
    
    def Train(self,inputs,targets,epochs=1):

        for i in range(epochs):
            
            with Tool.Timer() as t :
                for j in range(int(inputs.shape[0]/self.bs)):
                    out = self.train(inputs[j*self.bs:(j+1)*self.bs], targets[j*self.bs:(j+1)*self.bs])
            print('Epoch {0} ## Loss : {1:.4} ## Time : {2:.3} s'.format(i+1,float(out),t.interval))

    def EarlyStopTrain(self,train_inp,train_tar,valid_inp,valid_tar,max_epoch=2000):
        i,j,k = 0,0,0 # Iterators
        accuracy = np.zeros(2) # Initial accuracy
        step = 3 # Number of steps between evaluations
        patience = 15 # Number of bad steps before giving up
        self.best_time = 0
        self.best_params = []
        print('Training with Early Stop...')

        while i < patience and k < max_epoch:
            k += 1
            j += step

            self.Train(train_inp, train_tar, epochs = step)
            self.preds = self.predict(valid_inp[0:self.bs])

            if accuracy[1] > accuracy[0]:
                i = 0
                self.best_params = self.params
                self.best_time = j
                accuracy[0] = accuracy[1]
            else:
                i += 1
        
        print('Early Stop ! Saving best parameters...')
        self.__save__(name = 'BP_')
        print('Done.')

    def __save__(self,name):
        with open(Tool.getpath() + str(name) + 'CAE_1','wb') as file:
            pickle.dump(self.params.get_value(borrow=True),file,-1)

    def __load__(self,name):
        with open(Tool.getpath() + str(name) + 'CAE_1') as file:
            self.params.set_value(pickle.load(file),borrow=True)

    def onetest(self,n):

        oneimg = trainy[0].reshape(1,3,64,64)
        for i in range(n):
            out = self.train(oneimg,oneimg)
            print('Epoch {0} ## Loss : {1:.4}'.format(i+1,float(out)))
        onepred = M.predict(oneimg)
        I.plot(onepred[0])



I = Img()

try:
    trainset, validset

except NameError:
    trainset, validset = I.load(100) # Load x images
    trainx = trainset[0] # Inputs are cropped images
    trainy = trainset[1] # Targets are full images
    validx = validset[0]
    validy = validset[1]

    
M = ConvolutionalAutoEncoder()

M.EarlyStopTrain(trainx,trainy,validx,validy)
