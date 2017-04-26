#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 13:26:13 2017

@author: mcomin
"""

import sys
import os
import glob
import pickle
import PIL.Image as Image
import re

if os.environ['LOC'] == 'local':
    datapath = '/Network/Servers/seguin.pmc.umontreal.ca/Users/mcomin/inpainting'
    libpath = '../lib'
    filepath = os.getcwd()
elif os.environ['LOC'] == 'hades':
    datapath = '/home2/ift6ed13/data'
    libpath = '/home2/ift6ed13/lib'
    filepath = '/home2/ift6ed13/results'
sys.path.append(libpath)

import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from Layers import *

theano.config.floatX = 'float32'
theano.config.intX = 'int32'


class Loader:
    """
    Images and caption loader.
    """
    
    def __init__(self):
        
        # Images
        
        with open(libpath+'/SKIP_NAMES','rb') as file:
            self.IMG_TO_SKIP = pickle.load(file)

        trainlist = glob.glob(datapath+'/train/*.jpg')
        validlist = glob.glob(datapath+'/valid/*.jpg')
        
        self.trainlist = np.array([x for x in trainlist if x not in self.IMG_TO_SKIP])
        self.validlist = np.array([x for x in validlist if x not in self.IMG_TO_SKIP])
        
        self.namelist = {'train':self.trainlist,'valid':self.validlist}
        
    
    def get_batch(self,batchsize,i,mode):
        
        names = self.namelist[mode]
        batch_list = names[i*batchsize:(i+1)*batchsize]
        capt_list = ['COCO' + re.search('COCO(.+?).jpg',i).group(1) for i in batch_list]
            
        img = [np.array(Image.open(fname)) for fname in batch_list]
        img = np.array(img,dtype=theano.config.floatX)/256.
        img = img.transpose(0,3,1,2)

        img_crop = np.copy(img)
        img_crop[:,:,16:48,16:48] = 0
                  
        return img_crop, img[:,:,16:48,16:48], np.array(capt_list)

    def plot(self,image,save=False):

        image = image.transpose(1,2,0)
        plt.axis("off")
        plt.imshow(image)
        
        if save:
            plt.savefig(save+'.png')

class Model(Loader):
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
        
        Loader.__init__(self)
        self.bs = bs
        self.n = n
        self.trainlist = self.trainlist[0:n]
        self.name = 'LSGAN'

        
        
        delta = 0.15 # Part of Image Metrics in Generator Loss
        teta = 0.4 # Part of SSIM in Image Metric
        l_rate = 1e-3 # Learning rate
        m_rate = 0.9 # Momentum parameter


        x = T.tensor4('crop')
        y = T.tensor4('center')
        z = T.tensor4('noise')
        s = T.scalar('smooth')
        
        g = self.G(x,z)
        d_real = self.D(y)
        d_gen = self.D(g)
        
        one = np.ones(self.bs) 
        zero = np.zeros(self.bs)
        
        Dcost = .5*Tool.Mse( d_real, s*one, dims=2 ) + .5*Tool.Mse( d_gen, zero, dims=2 ) # Label smoothing (One sided)
        Gcost = .5*Tool.Mse( d_gen, one, dims=2 ) + delta*( (1.-teta)*Tool.Mae(g,y) + teta*Tool.DSSIM(g,y) )

        D_update = Tool.rmsprop_nesterov(Dcost, self.D_params, eta = l_rate, alpha = m_rate, rho=0.8)
        G_update = Tool.rmsprop_momentum(Gcost, self.G_params, eta = l_rate/10., alpha = m_rate, rho=0.9)


        self.train_D = theano.function([x,y,z,s], Dcost, updates = D_update)
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
        
        Drop = lambda x : Dropout(x, drop = 0.3, train = True)

        
        D[0] = InputLayer(y)
        
        D[1] = Conv(Drop(D[0]), 3, 16, s = 2) # Out = 16
        D[2] = BN(D[1], 16)
        
        D[3] = Conv(Drop(D[2]), 16, 24, s = 2) # Out = 8
        D[4] = BN(D[3], 24)

        D[5] = Conv(Drop(D[4]), 24, 36, s = 2) # Out = 4
        D[6] = BN(D[5], 36)
        
        D[7] = Conv(Drop(D[6]), 36, 72, s = 2) # Out = 2
        D[8] = BN(D[7], 72)
        
        D[9] = DenseLayer(D[8], 72*2*2, 1, activation = 'sigmoid') 
        
        self.D_params = [x for i in D.keys() for x in D[i].params]

        return D[9].output

    def G(self,x,z): 

        G = {}
        x = x.reshape((x.shape[0],3,64,64)) # Cropped image
        z = z.reshape((z.shape[0],3,64,64)) # Gaussian noise

        # s = 1 : Same size || s = 2 : size/2
        Conv = lambda x,y,z,s : ConvLayer(x, nchan = y, nkernels = z, kernelsize = (3,3),
                                    activation = None, pad = 1, stride=(s,s))

        BN = lambda x,y : BatchNorm(x, y, activation = 'relu', dims = 4)


        G['image'] = InputLayer(x)
        G['noise'] = InputLayer(z)
        
        G[0] = JoinLayer(G['image'],G['noise'],axis = 1)
        
        G[1] = Conv(G[0], 6, 72, 2)
        G[2] = BN(G[1], 72)
        
        G[3] = Conv(G[2], 72, 36, 1)
        G[4] = BN(G[3], 36)
        
        G[5] = Conv(G[4], 36, 18, 1) 
        G[6] = BN(G[5], 18)
        
        G[7] = Conv(G[6], 18, 9, 1)
        G[8] = BN(G[7], 9)
        
        G[9] = ConvLayer(G[8], nchan = 9, nkernels = 3, kernelsize = (3,3), activation = 'sigmoid', pad = 1)

        self.G_params = [x for i in G.keys() for x in G[i].params]

        return G[9].output


    def Train(self,epochs=1, save=True):

        Dloss = np.zeros((epochs,self.n // self.bs))
        Gloss = np.zeros((epochs,self.n // self.bs))
        
        for i in range(epochs):
            
            with Tool.Timer() as t:
                
                for j in range(self.n // self.bs):
                    
                    # Inputs
                    crop, center, _ = self.get_batch(self.bs,j,'train')
                    z = self.Noise((self.bs,3,64,64))
                    
                    # Random smoothing
                    s = np.random.normal(0.9,0.05)
                    
                    # Training !
                    Dloss[i,j] = self.train_D(crop, center, z, s)
                    Gloss[i,j] = self.train_G(crop, center, z)
            
            string = 'Epoch {0} ## Discriminator Loss : {1:.6} ## Generator Loss : {2:.6} ## Time : {3:.2} s'
            print(string.format(i+1,float(Dloss[i,-1]),float(Gloss[i,-1]),t.interval))

            if ((i+1)%5 == 0 or i+1 == epochs) and save:
                self.Generate('train')
                self.Generate('valid')
                self.__save__(str(i+1))

        # Save all losses : 
        
        losses = {'G':Gloss,'D':Dloss}
        with open(filepath + '/' + self.name + '_LOSS','wb') as f:
            pickle.dump(losses, f, 2)

    def Generate(self,mode,n=0):
        
        crop,_,names = self.get_batch(self.bs,n,mode=mode)
        z = self.Noise((self.bs,3,64,64))
        
        base = np.copy(crop)
        pred = np.array(self.generator(crop,z))

        base[:,:,16:48,16:48] += pred

        if mode == 'train':
            self.train_recon = base, names
        elif mode == 'valid':
            self.valid_recon = base, names

    def Noise(self,size):
        return np.random.normal(0.,1.,size = size).astype(theano.config.floatX)
    
    def __save__(self,epoch):
        
        directory = filepath+'/'+self.name
        
        G_numpy_params = [self.G_params[k].get_value() for k in range(len(self.G_params))]
        D_numpy_params = [self.D_params[k].get_value() for k in range(len(self.D_params))]
        
        with open(directory + '_G_params_' + epoch,'wb') as file:
            pickle.dump(G_numpy_params,file, 2)

        with open(directory + '_D_params_' + epoch,'wb') as file:
            pickle.dump(D_numpy_params,file, 2)

        for i in range(15):
            
            with open(directory + '_train/' + epoch + '_' + str(i), 'wb') as file:
                pickle.dump(self.train_recon[0][i], file, 2)
                
            with open(directory + '_valid/' + epoch + '_' + str(i), 'wb') as file:
                pickle.dump(self.valid_recon[0][i], file, 2)
        
        with open(directory + '_train/' + 'train_names.txt','wb') as file:
            pickle.dump(self.train_recon[1][0:15], file)
        with open(directory + '_valid/' + 'valid_names.txt','wb') as file:
            pickle.dump(self.valid_recon[1][0:15], file)


    def __load__(self,epoch):
        
        with open(filepath+'/'+self.name+'_G_params_' + epoch,'rb') as file:
            G_loaded_params = pickle.load(file)
            
        for k in range(len(self.G_params)):
            self.G_params[k].set_value(G_loaded_params[k])

        with open(filepath+'/'+self.name+'_D_params_' + epoch,'rb') as file:
            D_loaded_params = pickle.load(file)
            
        for k in range(len(self.D_params)):
            self.D_params[k].set_value(D_loaded_params[k])

