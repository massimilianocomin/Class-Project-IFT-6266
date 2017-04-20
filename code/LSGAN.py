#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 13:26:13 2017

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
theano.config.exception_verbosity = 'low'
theano.config.optimizer = 'fast_run' # fast_run, fast_compile,None

class Model:
    """
    Implementation of a Conditional Least Squares Generative Adversarial Network.
    
    Generator : 
        - Takes High dimentional noise + cropped image as input
        - As in DCGAN : relu activations, no pooling, strides instead.
        - Add Skip Connections ?
        - Gaussian noise at last layer ?
        
    Discriminator :
        - Takes a full image (true or cropped + generated center) as input.
        - As in LSGAN : leaky relu activations, no pooling, strides instead.
    
    
    *** During Validation, Rebuild Model ***
    BatchNorm Layers with train_batch's mean and std
    Dropout Layers with train = False
    """
    def __init__(self,bs,n=None):
    
        print('Building computational graph...')
        
        self.bs = bs
        self.I = Img()
        self.n = n
        self.I.trainlist = self.I.trainlist[0:n]
        self.name = 'LSGAN'
        
        
        delta = 0.15 # Part of Image Metrics in Generator Loss
        teta = 0.4 # Part of SSIM in Image Metric
        l_rate = 1e-3 # Learning rate
        m_rate = 0.9 # Momentum parameter


        x = T.tensor4('crop')
        y = T.tensor4('center')
        z = T.tensor4('noise')
        
        g = self.G(x,z)
        d_real = self.D(y)
        d_gen = self.D(g)
        
        one = .9*np.ones(self.bs) # Label smoothing
        zero = .02*np.zeros(self.bs) # Label smoothing
        
        Dcost = .5*Tool.Mse( d_real, one, dims=2 ) + .5*Tool.Mse( d_gen, zero, dims=2 )
        Gcost = .5*Tool.Mse( d_gen, one, dims=2 ) + delta*( (1.-teta)*Tool.Mae(g,y) + teta*Tool.DSSIM(g,y) )

        D_update = Tool.rmsprop_nesterov(Dcost, self.D_params, eta = l_rate, alpha = m_rate)
        G_update = Tool.adam(Gcost, self.G_params)


        self.train_D = theano.function([x,y,z], Dcost, updates = D_update)
        self.train_G = theano.function([x,y,z], Gcost, updates = G_update)
        self.generator = theano.function([x,z], g)
        
        print('Computational graph built.')

    def D(self,y):
        
        D = {}
        y = y.reshape((y.shape[0],3,32,32))
        
        # s = 1 : Same size || s = 2 : size/2
        Conv = lambda x,y,z,s : ConvLayer(x, nchan = y, nkernels = z, kernelsize = (3,3),
                            activation = None, pad = 1, stride=(s,s))

        BN = lambda x,y : BatchNorm(x, y, activation = 'relux', dims = 4)

        
        D[0] = InputLayer(y)
        
        D[1] = Conv(D[0], 3, 64, s = 2) # Out = 16
        D[2] = BN(D[1], 64)
        
        D[3] = Conv(D[2], 64, 64, s = 1) # Out = 16
        D[4] = BN(D[3], 64)

        D[5] = Conv(D[4], 64, 128, s = 2) # Out = 8
        D[6] = BN(D[5], 128)
        
        D[7] = Conv(D[6], 128, 128, s = 1) # Out = 8
        D[8] = BN(D[7], 128)
        
        D[9] = Conv(D[8], 128, 256, s = 2) # Out = 4
        D[10] = BN(D[9], 256)
        
        D[11] = Conv(D[10], 256, 256, s = 1) # Out = 4
        D[12] = BN(D[11], 256)
        
        D[13] = Conv(D[12], 256, 512, s = 2) # Out = 2
        D[14] = BN(D[13], 512)
        
        D[15] = Conv(D[14], 512, 512, s = 1) # Out = 2
        D[16] = BN(D[15], 512)
        
        D[17] = DenseLayer(D[16], 512*2*2, 256, activation = 'relux') 
        
        D[18] = DenseLayer(D[17], 256, 1, activation = 'sigmoid')
        
        self.D_params = [x for i in D.keys() for x in D[i].params]

        return D[18].output

    def G(self,x,z): 

        G = {}
        x = x.reshape((x.shape[0],3,64,64)) # Cropped image
        z = z.reshape((z.shape[0],3,128,128)) # Gaussian noise

        # s = 1 : Same size || s = 2 : size/2
        Conv = lambda x,y,z,s : ConvLayer(x, nchan = y, nkernels = z, kernelsize = (3,3),
                                    activation = None, pad = 1, stride=(s,s))

        BN = lambda x,y : BatchNorm(x, y, activation = 'relu', dims = 4)

        # Cropped Image
        G['x0'] = InputLayer(x)
        
        G['x1'] = Conv(G['x0'], 3, 256, 1) # Out = 64
        G['x2'] = BN(G['x1'], 256)
        
        G['x3'] = Conv(G['x2'], 256, 128, 2) # Out = 32
        G['x4'] = BN(G['x3'], 128)
        
        G['x5'] = Conv(G['x4'], 128, 64, 1) # Out = 32


        # Gaussian Noise
        G['z0'] = InputLayer(z)
        
        G['z1'] = Conv(G['z0'], 3, 256, 2) # Out = 64
        G['z2'] = BN(G['z1'], 256)
        
        G['z3'] = Conv(G['z2'], 256, 128, 1) # Out = 64
        G['z4'] = BN(G['z3'], 128)
        
        G['z5'] = Conv(G['z4'], 128, 64, 2) # Out = 32


        # Junction, Out = 32

        G[-1] = SumLayer(G['x5'],G['z5'])
        G[0] = BN(G[-1], 64)
        
        G[1] = Conv(G[0], 64, 64, 1)
        G[2] = BN(G[1], 64)
        
        G[3] = Conv(G[2], 64, 32, 1) 
        G[4] = BN(G[3], 32)
        
        G[5] = Conv(G[4], 32, 16, 1) 
        G[6] = BN(G[5], 16)
        
        G[7] = Conv(G[6], 16, 8, 1)
        G[8] = BN(G[7], 8)
        
        G[9] = ConvLayer(G[8], nchan = 8, nkernels = 3, kernelsize = (3,3), activation = 'sigmoid', pad = 1)

        self.G_params = [x for i in G.keys() for x in G[i].params]

        return G[9].output


    def Train(self,epochs=1, save=True):

        for i in range(epochs):
            
            with Tool.Timer() as t:
                
                for j in range(self.n // self.bs):
                    
                    crop, center = self.I.load_batch(self.bs,j,mode='train')
                    z = self.Noise((self.bs,3,128,128))
                    
                    dloss = self.train_D(crop, center, z)
                    gloss = self.train_G(crop, center, z)
            
            string = 'Epoch {0} ## Discriminator Loss : {1:.4} ## Generator Loss : {2:.4} ## Time : {3:.2} s'
            print(string.format(i+1,float(dloss),float(gloss),t.interval))

            if ((i+1)%5 == 0 or i+1 == epochs) and save:
                self.Generate('train')
                self.Generate('valid')
                self.__save__(str(i+1))
            
    def Generate(self,mode,nbatch=1):

        base_img = []
        center_pred = []
        
        for j in range(nbatch):
            crop,_ = self.I.load_batch(self.bs,j,mode=mode)
            base_img += [crop]
            z = self.Noise((self.bs,3,128,128))
            center_pred += [self.generator(crop,z)]
        
        pred = np.concatenate(center_pred)
        recon = np.concatenate(base_img)
        recon[:,:,16:48,16:48] += pred
 
        if mode == 'train':
            names = self.I.trainlist[0:self.bs]
            self.train_recon = recon,names
        elif mode == 'valid':
            names = self.I.validlist[0:self.bs]
            self.valid_recon = recon, names

    def Noise(self,size):
        return np.random.normal(0.,1.,size = size).astype(theano.config.floatX)
    
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



