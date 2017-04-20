#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:56:03 2017

@author: mcomin
"""
import sys
import os

if os.environ['LOC'] == 'local':
    libpath = '../lib'
    filepath = os.getcwd()
elif os.environ['LOC'] == 'hades':
    libpath = '/home2/ift6ed13/lib'
    filepath = '/home2/ift6ed13/results'
sys.path.append(libpath)

import numpy as np
import pickle
import theano
import theano.tensor as T
from Img import *
from Layers import *

theano.config.floatX = 'float32'
theano.config.intX = 'int32'

class Model:
    """
    Implementation of a Convolutional image generator without pooling.
    """
    def __init__(self,bs,n=None,entrainement=True):
    
        print('Building computational graph...')

        self.bs = bs
        self.n = n
        self.entrainement = entrainement
        
        self.name = 'Conv2'
        self.path = filepath
        self.I = Img()
        self.I.trainlist = self.I.trainlist[0:n]

        x = T.tensor4('inputs') 
        y = T.tensor4('targets')
        
        loss = self.build(x,y)

        updat = Tool.adam(loss, self.params)
        print(self.params)
        self.train = theano.function([x,y],loss,updates=updat)
        self.predict = theano.function([x],self.Y)
        
        print('Computational graph built.')


    def build(self,x,y): # No pooling / unpooling
    #previous conv : 4 conv blocks, 10 to 80 kernels, tanh activations only, stride only. Divide by 256

        L = {}
        x = x.reshape((x.shape[0],3,64,64))
        y = y.reshape((y.shape[0],3,32,32))

        # s = 1 : Same size || s = 2 : size/2
        Conv = lambda x,y,z,s,a : ConvLayer(x, nchan = y, nkernels = z, kernelsize = (3,3),
                                    activation = a, pad = 1, stride=(s,s))

        TConv = lambda x,shp,args : TConvLayer(x, batch = self.bs, shape = shp, tied=False, **args)

        BN = lambda x,y : BatchNorm(x, y, activation = 'sigmoid', dims = 4)


        L[0] = InputLayer(x)
        L[1] = Conv(L[0], 3, 15, 1, 'relux')
        L[2] = Conv(L[1], 15, 30, 2, None)
        L[3] = BN(L[2], 30)
        L[4] = Conv(L[3], 30, 40, 1, 'relux')
        L[5] = Conv(L[4], 40, 50, 2, None)
        L[6] = BN(L[5], 50)
        L[7] = Conv(L[6], 50, 60, 1, 'relux')
        L[8] = Conv(L[7], 60, 70, 2, None)
        L[9] = BN(L[8], 70)
        L[10] = Conv(L[9], 70, 80, 1, 'relux')
        L[11] = Conv(L[10], 80, 90, 2, None)
        L[12] = BN(L[11], 90)
        L[13] = Conv(L[12], 90, 100, 1, 'relux')
        L[14] = Conv(L[13], 100, 110, 2, None)
        L[15] = BN(L[14], 110)
        
        L[16] = TConv(L[15], (4,4), L[14].arguments)
        L[17] = TConv(L[16], (4,4), L[13].arguments)
        L[18] = BN(L[17], 90)
        L[19] = TConv(L[18], (8,8), L[11].arguments)
        L[20] = TConv(L[19], (8,8), L[10].arguments)
        L[21] = BN(L[20], 70)
        L[22] = TConv(L[21], (16,16), L[8].arguments)
        L[23] = TConv(L[22], (16,16), L[7].arguments)
        L[24] = BN(L[23], 50)
        L[25] = TConv(L[24], (32,32), L[5].arguments)
        L[26] = TConv(L[25], (32,32), L[4].arguments)
        L[27] = BN(L[26], 30)
        
        mod = L[2].arguments.copy()
        mod['stride'] = (1,1)
        
        L[28] = TConv(L[27], (32,32), mod)
        L[29] = TConv(L[28], (32,32), L[1].arguments)
        
        self.Y = L[29].output

        self.params = [x for i in L.keys() for x in L[i].params]

        return Tool.Mae(self.Y, y)/64. + Tool.DSSIM(self.Y, y)
    
    def Train(self,epochs=1,save=True):

        for i in range(epochs):
            
            with Tool.Timer() as t :
                
                for j in range(self.n // self.bs):
                    
                    crop,center = self.I.load_batch(self.bs,j,mode='train')
                    out = self.train(crop,center)
                    
            print('Epoch {0} ## Loss : {1:.4} ## Time : {2:.3} s'.format(i+1,float(out),t.interval))

            if ((i+1)%5 == 0 or i+1 == epochs) and save:
                self.Generate('train')
                self.Generate('valid')
                self.__save__(str(i+1))
            
    def Generate(self,mode,nbatch=1):

        base_img = []
        center_pred = []
        
        for j in range(nbatch):
            crop,center = self.I.load_batch(self.bs,j,mode=mode)
            base_img += [crop]
            center_pred += [self.predict(crop)]
        
        pred = np.concatenate(center_pred)
        recon = np.concatenate(base_img)
        recon[:,:,16:48,16:48] += pred
 
        if mode == 'train':
            names = self.I.trainlist[0:self.bs]
            self.train_recon = recon,names
        elif mode == 'valid':
            names = self.I.validlist[0:self.bs]
            self.valid_recon = recon, names

    def __save__(self,epoch):
        
        directory = self.path+'/'+self.name
        
        numpy_params = [self.params[k].get_value() for k in range(len(self.params))]
        
        with open(directory + '_params_' + epoch,'wb') as file:
            pickle.dump(numpy_params,file, 2)

        for i in range(10):
            self.I.save(self.train_recon[0][i],directory + '_train/' + epoch + '_' + str(i))
            self.I.save(self.valid_recon[0][i],directory + '_valid/' + epoch + '_' + str(i))
        
        with open(directory + '_train/' + 'train_names.txt','wb') as file:
            pickle.dump(self.train_recon[1][0:10], file)
        with open(directory + '_valid/' + 'valid_names.txt','wb') as file:
            pickle.dump(self.valid_recon[1][0:10], file)


    def __load__(self,epoch):
        
        with open(self.path+'/'+self.name+'_params_' + epoch,'rb') as file:
            loaded_params = pickle.load(file)
            
        for k in range(len(self.params)):
            self.params[k].set_value(loaded_params[k])
        
