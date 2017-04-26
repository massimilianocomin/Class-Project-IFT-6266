#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:44:58 2017

@author: mcomin
"""
import os
import sys
import numpy as np

if os.environ['LOC'] == 'local':
    path = '/Network/Servers/seguin.pmc.umontreal.ca/Users/mcomin/inpainting'
    libpath = '/Users/mcomin/Google Drive/Master/Codes/lib'

elif os.environ['LOC'] == 'hades':
    path = '/home2/ift6ed13/data'
    libpath = '/home2/ift6ed13/lib'
else:
    sys.exit('Environment variable LOC not found. Verify .bash_profile.')
    
    
import matplotlib.pyplot as plt
import PIL.Image as Image
import theano
import glob
import pickle

theano.config.floatX = 'float32'
theano.config.intX = 'int32'


class Img:
    
    def __init__(self):

        with open(libpath+'/SKIP_NAMES','rb') as file:
            self.IMG_TO_SKIP = pickle.load(file)

        trainlist = glob.glob(path+'/train/*.jpg')
        validlist = glob.glob(path+'/valid/*.jpg')
        
        self.trainlist = [x for x in trainlist if x not in self.IMG_TO_SKIP]
        self.validlist = [x for x in validlist if x not in self.IMG_TO_SKIP]
        
        self.namelist = {'train':self.trainlist,'valid':self.validlist}


    def load(self,n=None,shared=False,mode='train'):
        """
        Loads the whole dataset, or the first n examples.
        """
        
        print('Loading COCO dataset...')
        
        names = self.namelist[mode]
        
        dataset = [np.array(Image.open(fname)) for fname in names[0:n]]
        dataset = np.array(dataset)
        dataset = (dataset[0:n]/256).astype(theano.config.floatX)
        dataset = dataset.transpose(0,3,1,2)

        data_crop = np.copy(dataset)
        data_crop[:,:,16:48,16:48] = 0
        data_center = np.copy(dataset)
        data_center = data_center[:,:,16:48,16:48]
        
        if shared:
            data_crop = theano.tensor._shared(data_crop,borrow=True)
            data_center = theano.tensor._shared(data_center,borrow=True)
            
        print('Dataset loaded.')

        return data_crop, data_center


    def load_batch(self,batchsize,i,mode):
        """
        Loads successive minibatches of the dataset.
        """
        names = self.namelist[mode]
        batch_list = names[i*batchsize:(i+1)*batchsize]
            
        batch = [np.array(Image.open(fname)) for fname in batch_list]
        batch = np.array(batch,dtype=theano.config.floatX)/256.
        batch = batch.transpose(0,3,1,2)

        batch_crop = np.copy(batch)
        batch_crop[:,:,16:48,16:48] = 0
        
        return batch_crop, batch[:,:,16:48,16:48]


    def plot(self,inp,imgpath=None):
        
        image = inp.transpose(1,2,0)

        plt.axis("off")
        plt.imshow(image)
        if imgpath:
            plt.savefig(imgpath+'.png')
    
    def save(self,img, path):
        
        with open(path,'wb') as f:
            pickle.dump(img,f)

    def _DETECT_GRAYSCALE_IMG(self):
        
        skipnames = []
        
        for name in self.trainlist:
            op = Image.open(name)
            img = np.array(op)
            if img.shape != (64,64,3):
                skipnames += [name]
                print('Found Error')
            op.close()

        for name in self.validlist:
            op = Image.open(name)
            img = np.array(op)
            if img.shape != (64,64,3):
                skipnames += [name]
                print('Found Error')
            op.close()
        
        with open('SKIP_NAMES','wb') as file:
            pickle.dump(skipnames,file,-1)
