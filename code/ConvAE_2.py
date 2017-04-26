#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 13:11:10 2017

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
    Implementation of a ConvAE
    
    *** During Validation, Rebuild Model ***
    BatchNorm Layers with train_batch's mean and std
    Dropout Layers with train = False
    """
    def __init__(self,bs,n=None,entrainement=True):
    
        print('Building computational graph...')

        self.bs = bs
        self.n = n
        self.entrainement = entrainement
        
        self.path = filepath
        self.I = Img()
        self.I.trainlist = self.I.trainlist[0:n]

        x = T.tensor4('inputs') 
        y = T.tensor4('targets')
        
        loss = self.build(x,y)

        updat = Tool.adam(loss, self.params)

        self.train = theano.function([x,y],loss,updates=updat)
        self.predict = theano.function([x],self.Y)
        
        print('Computational graph built.')

    def build(self,x,y):

        L = {}
        x = x.reshape((x.shape[0],3,64,64))
        y = y.reshape((y.shape[0],3,32,32))
        self.params = []
        sizeout = 4 # 4 pooling layers over a 64x64 image = 4x4 image
        

        Conv = lambda x,y,z,p : ConvLayer(x, nchan = y, nkernels = z, kernelsize = (3,3),
                                    activation = None, pad = 'half', pmode = 'average_inc_pad',
                                    poolsize=(p,p))
        
        TConv = lambda x,shp,args,tied : TConvLayer(x, batch = self.bs, tied=tied, shape = shp, **args)

        BN = lambda x,y : BatchNorm(x, y, activation = 'tanh', dims = 4)


        L[0] = InputLayer(x)

        L[1] = Conv(L[0], 3, 15, 1)
        L[2] = Conv(L[1], 15, 30, 2)

        L[3] = BN(L[2], 30)

        L[4] = Conv(L[3], 30, 40, 1)
        L[5] = Conv(L[4], 40, 50, 2)

        L[6] = BN(L[5], 50)

        L[7] = Conv(L[6], 50, 60, 1)
        L[8] = Conv(L[7], 60, 70, 2)

        L[9] = BN(L[8], 70)
        
        L[10] = Conv(L[9], 70, 80, 1)
        L[11] = Conv(L[10], 80, 90, 2)
        
        L[12] = BN(L[11], 90)


        L[13] = DenseLayer(L[12], 90*sizeout**2, 45*sizeout**2, 'elu')
        L[14] = DenseLayer(L[13], 45*sizeout**2, 90*sizeout**2, 'elu')
        
        reshaped = ReshapeLayer(L[14], (self.bs,90,sizeout,sizeout))


        L[15] = SumLayer(reshaped,L[11])
        L[16] = BN(L[15], 90)
        
        L[17] = TConv(L[16], (8,8), L[11].arguments, tied=False)
        L[18] = TConv(L[17], (8,8), L[10].arguments, tied=True)
        
        L[19] = SumLayer(L[18], L[8])
        L[20] = BN(L[19], 70)

        L[21] = TConv(L[20], (16,16), L[8].arguments, tied=False)
        L[22] = TConv(L[21], (16,16), L[7].arguments, tied=True)

        L[23] = SumLayer(L[22],L[5])
        L[24] = BN(L[23], 50)

        L[25] = TConv(L[24], (32,32), L[5].arguments, tied=False)
        L[26] = TConv(L[25], (32,32), L[4].arguments, tied=True)

        L[27] = SumLayer(L[26],L[2])
        L[28] = BN(L[27], 30)

        modified_args = L[2].arguments.copy()
        modified_args['poolsize'] = (1,1)

        L[29] = TConv(L[28], (32,32), modified_args, tied=False)
        L[30] = TConv(L[29], (32,32), L[1].arguments, tied=True)

        self.Y = L[30].output
        self.L = L

        self.params = [x for i in L.keys() for x in L[i].params]

        return Tool.Mae(self.Y, y)/(16.**2) + Tool.DSSIM(self.Y, y)

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

