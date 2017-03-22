#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:44:58 2017

@author: mcomin
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from skimage.transform import resize
import glob

class ImageTools:
    
    def __init__(self,path=None):

        self.path = '/Network/Servers/seguin.pmc.umontreal.ca/Users/mcomin/inpainting' if path is None else path


    def load(self,n=None):
        
        m = None if n is None else 2*n
        
        print('Loading COCO dataset...')
        trainlist = glob.glob(self.path+'/train'+'/*.jpg')
        validlist = glob.glob(self.path+'/valid'+'/*.jpg')
        traincroplist = glob.glob(self.path+'/train_crop'+'/*.jpg')
        validcroplist = glob.glob(self.path+'/valid_crop'+'/*.jpg')
        
        train = [np.array(Image.open(fname)) for fname in trainlist[0:m]]
        train = np.array([x for x in train if x.shape == (64,64,3)])
        train = train[0:n]

        valid = [np.array(Image.open(fname)) for fname in validlist[0:m]]
        valid = np.array([x for x in valid if x.shape == (64,64,3)])
        valid = valid[0:n]

        train_crop = [np.array(Image.open(fname)) for fname in traincroplist[0:m]]
        train_crop = np.array([x for x in train_crop if x.shape == (64,64,3)])
        train_crop = train_crop[0:n]

        valid_crop = [np.array(Image.open(fname)) for fname in validcroplist[0:m]]
        valid_crop = np.array([x for x in valid_crop if x.shape == (64,64,3)])
        valid_crop = valid_crop[0:n]
        
        train = train.transpose(0,3,1,2)
        valid = valid.transpose(0,3,1,2)
        train_crop = train_crop.transpose(0,3,1,2)
        valid_crop = valid_crop.transpose(0,3,1,2)
        
        trainset = train,train_crop
        validset = valid,valid_crop
        print('Dataset loaded.')

        return trainset, validset
    
    def plot(self,inp):

        image = np.copy(inp)
        image = image.transpose(1,2,0)
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

    
    def _GEN_CROP_IMG(self):
        
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




