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
    import matplotlib
    matplotlib.use('Agg')
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


    def load(self,n=None):
        """
        Loads the whole dataset, or the first n examples.
        """
        
        print('Loading COCO dataset...')
        
        train = [np.array(Image.open(fname)) for fname in self.trainlist[0:n]]
        train = np.array([x for x in train if x.shape == (64,64,3)])
        train = (train[0:n]/256).astype(theano.config.floatX)
        train = train.transpose(0,3,1,2)

        valid = [np.array(Image.open(fname)) for fname in self.validlist[0:n]]
        valid = np.array([x for x in valid if x.shape == (64,64,3)])
        valid = (valid[0:n]/256).astype(theano.config.floatX)
        valid = valid.transpose(0,3,1,2)

        train_crop = np.copy(train)
        train_crop[:,:,16:48,16:48] = 0
        valid_crop = np.copy(valid)
        valid_crop[:,:,16:48,16:48] = 0

        print('Dataset loaded.')

        return train_crop, train, valid_crop, valid


    def load_batch(self,batchsize,i,mode):
        """
        Loads successive minibatches of the dataset.
        """
        
        if mode == 'train':
            batch_list = self.trainlist[i*batchsize:(i+1)*batchsize]
        elif mode == 'valid':
            batch_list = self.validlist[i*batchsize:(i+1)*batchsize]
        else:
            sys.exit('Img.load_batch Error: Please select a valid mode.')
            
        batch = [np.array(Image.open(fname)) for fname in batch_list]
        batch = np.array(batch[0:batchsize],dtype=theano.config.floatX)/256.
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
    
    def save(self,image,imgpath):
        
        image = image.transpose(1,2,0)
        
        with open(imgpath,'wb') as file:
            pickle.dump(image,file, 2)

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

I = Img()
#for i in range(80000//100):
#    test = I.load_batch(100,i,mode='train')
#    if test[0].shape != (100, 3, 64, 64):
#        print('ERROR 0')
#    if test[1].shape != (100, 3, 32, 32):
#        print('ERROR 1')
