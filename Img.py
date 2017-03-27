#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:44:58 2017

@author: mcomin
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import theano
import glob

theano.config.floatX = 'float32'
theano.config.intX = 'int32'


class Img:
    
    def __init__(self):
        
        if os.environ['LOC'] == 'local':
            self.path = '/Network/Servers/seguin.pmc.umontreal.ca/Users/mcomin/inpainting'
        elif os.environ['LOC'] == 'hades':
            self.path = '/home2/ift6ed13/data'
        else: 
            sys.exit('Environment variable LOC not found. Verify .bash_profile.')
    def load(self,n=None):
        
        m = None if n is None else 2*n
        
        print('Loading COCO dataset...')
        trainlist = glob.glob(self.path+'/train'+'/*.jpg')
        validlist = glob.glob(self.path+'/valid'+'/*.jpg')
        traincroplist = glob.glob(self.path+'/train_crop'+'/*.jpg')
        validcroplist = glob.glob(self.path+'/valid_crop'+'/*.jpg')
        
        train = [np.array(Image.open(fname)) for fname in trainlist[0:m]]
        train = np.array([x for x in train if x.shape == (64,64,3)])
        train = (train[0:n]/256).astype(theano.config.floatX)

        valid = [np.array(Image.open(fname)) for fname in validlist[0:m]]
        valid = np.array([x for x in valid if x.shape == (64,64,3)])
        valid = (valid[0:n]/256).astype(theano.config.floatX)

        train_crop = [np.array(Image.open(fname)) for fname in traincroplist[0:m]]
        train_crop = np.array([x for x in train_crop if x.shape == (64,64,3)])
        train_crop = (train_crop[0:n]/256).astype(theano.config.floatX)

        valid_crop = [np.array(Image.open(fname)) for fname in validcroplist[0:m]]
        valid_crop = np.array([x for x in valid_crop if x.shape == (64,64,3)])
        valid_crop = (valid_crop[0:n]/256).astype(theano.config.floatX)
        
        train = train.transpose(0,3,1,2)
        valid = valid.transpose(0,3,1,2)
        train_crop = train_crop.transpose(0,3,1,2)
        valid_crop = valid_crop.transpose(0,3,1,2)
        
        trainset = train_crop, train
        validset = valid_crop, valid
        print('Dataset loaded.')

        return trainset, validset
    
    def plot(self,inp,tar=None):
        
        if tar is not None:
            image = np.concatenate((inp.transpose(1,2,0),tar.transpose(1,2,0)),axis=1)
        else:
            image = inp.transpose(1,2,0)

        plt.axis("off")
        plt.imshow(image)

    def crop(self,img):

        tocrop = np.array(img)
        center = (int(np.floor(tocrop.shape[0] / 2.)), int(np.floor(tocrop.shape[1] / 2.)))
        
        if len(tocrop.shape) == 3:
            tocrop[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
        else:
            tocrop[center[0]-16:center[0]+16, center[1]-16:center[1]+16] = 0
        
        cropped_image = Image.fromarray(tocrop)
        
        return cropped_image

    
    def _GENERATE_CROPPED_IMG(self):
        
        print('Loading images from path...')
        train = glob.glob(self.path+'/train/*.jpg')
        valid = glob.glob(self.path+'/valid/*.jpg')
        train_crop = glob.glob(self.path+'/train_crop/*.jpg')
        valid_crop = glob.glob(self.path+'/valid_crop/*.jpg')
        
        print('Verifying already cropped images...')
        train_todo = [x for x in train if x not in set(train_crop)]
        valid_todo = [x for x in valid if x not in set(valid_crop)]
        print('Cropping images...')

        for image_path in valid_todo:
            try:
                img = Image.open(image_path)
                img = self.crop(img)
                name = os.path.basename(image_path)
                img.save( self.path + '/valid_crop/' + name)
            except:
                print('An error occured while cropping ',name,'. Image skipped.')

        for image_path in train_todo:
            try:
                img = Image.open(image_path)
                img = self.crop(img)
                name = os.path.basename(image_path)
                img.save( self.path + '/train_crop/' + name)
            except:
                print('An error occured while cropping ',name,'. Image skipped.')
