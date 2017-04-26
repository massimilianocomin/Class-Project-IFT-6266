#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 16:31:28 2017

@author: mcomin
"""

import sys
import os
import numpy as np
import _pickle as pickle
import gensim
import string
from collections import OrderedDict

class Caption:
    """
    Caption class.
    --------------
    
    Contains the necessary utilities to perform a word embedding of the captions.
    
    """
    def __init__(self):
        
        if os.environ['LOC'] == 'hades':
            self.datapath = '/home2/ift6ed13/data'
        elif os.environ['LOC'] == 'local':
            self.datapath = '/Network/Servers/seguin.pmc.umontreal.ca/Users/mcomin/inpainting'

        caption_path = self.datapath + '/dict_key_imgID_value_caps_train_and_valid.pkl'

        with open(caption_path,'rb') as f :
            self.caption = pickle.load(f)
            self.caption_keys = list(self.caption.keys()) # Name of image
            self.caption_vals = list(self.caption.values()) # 5 captions per image

    
    def convert(self,phrase):
        
        remove_punct = phrase.translate(str.maketrans('','',string.punctuation))
        return remove_punct.split()

    def cosine_dist(self,x,y):
        return np.dot(x,y) / (np.linalg.norm(x) * np.linalg.norm(y))

    def build(self):

        print('Training Word2Vec on captions...')
        
        sentences = [self.convert(x) for y in self.caption_vals for x in y]
        self.model = gensim.models.Word2Vec(sentences, size=200, window=10, workers=4, min_count = 1, iter = 20)

        print('Model built.')

        # Google: gensim.models.KeyedVectors.load_word2vec_format(self.datapath+'/GoogleNews-vectors-negative300.bin.gz',binary=True)

        with open(self.datapath + '/worddict.pkl','rb') as f:
            words = pickle.load(f)
            words = list(words.keys())


        print('Creating word dictionary...')

        self.wordict = OrderedDict({})

        for i in words:
            try:
                self.wordict[i] = self.model[i]
            except:
                print(i)
                continue

        print('Vector word embedding created.')
    
    def save(self,filename):
        with open(os.getcwd()+'/'+filename,'wb') as file:
            pickle.dump(self.caption_vec, file, 2)
 
    
    
