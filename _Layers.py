#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:01:33 2017

@author: mcomin
"""
import sys
import time
import numpy as np
import numpy.random as rng
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

theano.config.floatX = 'float32'
theano.config.intX = 'int32'

class Loss:

    @staticmethod
    def crossE(p,y):
        return T.mean(T.nnet.categorical_crossentropy(p,y))

    @staticmethod
    def crossent(p,y):
        return -T.mean(T.log(p)[T.arange(y.shape[0]), y])

    @staticmethod
    def accuracy(p,y):
        return T.mean(T.eq(T.argmax(p, axis = 1),y))

    @staticmethod
    def nll(p,y,n):
        return -T.sum(T.log(p)*T.extra_ops.to_one_hot(y, n))

    @staticmethod
    def L1(param):
        return T.sum(np.abs(param))

    @staticmethod
    def L2(param):
        return T.sum(param**2)

class Optimizer:

    @staticmethod
    def rmsprop():
        raise NotImplementedError

    @staticmethod
    def adam():
        raise NotImplementedError

    @staticmethod
    def sgd():
        raise NotImplementedError

class Tool:

    functions = {None:lambda x: x,
                    'sigmoid':T.nnet.sigmoid,
                    'tanh':T.tanh,
                    'softmax':T.nnet.softmax,
                    'relu':T.nnet.relu,
                    'softplus':T.nnet.softplus}

    @staticmethod
    def setseed():
        raise NotImplementedError

class DenseLayer:
    """
    Fully-connected dense layer class.
    # Arguments :
        : inputs : ndarray or T.tensor : Previous layer

        : n_in : int : Number of units in the previous layer

        : n_out : int : Number of output units

        : activation : string : Activation function (None,sigmoid,tanh,relu,softmax)

    # Attributes :
        : params : list : List of all the parameters of the layer

        : output : ndarray or T.tensor : Layer's output
    """
    c = 0
    def __init__(self,inputs,n_in,n_out,activation=None):

        assert activation in Tool.functions.keys()

        DenseLayer.c += 1
        c = DenseLayer.c

        inputs = inputs.flatten(2)

        x = np.sqrt(6. / (n_in + n_out)) * (4. if activation == 'sigmoid' else 1.)
        w = rng.uniform(-x,x,size=(n_in,n_out)).astype(theano.config.floatX)
        b = np.zeros(n_out,dtype=theano.config.floatX)

        self.W = theano.shared(w,'W'+str(c))
        self.B = theano.shared(b,'B'+str(c))
        self.params = [self.W,self.B]
        self.output = Tool.functions[activation](T.dot(inputs,self.W) + self.B)

class ConvLayer:
    """
    Convolutional + Maxpooling layer class.
    # Arguments :
        : inputs : Matrix :

        : filter_shape : tuple :

        : poolsize : tuple :

        : activation : string : Activation function (None,sigmoid,tanh,relu,softmax)

    # Returns :
    """
    c = 0
    def __init__(self,inputs,filter_shape,poolsize=(2,2),activation='relu',alpha=.02):

        assert activation in Tool.functions.keys()

        ConvLayer.c += 1
        c = ConvLayer.c

        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) // np.prod(poolsize))
        x = np.sqrt(6./(fan_in+fan_out)) * (4. if activation == 'sigmoid' else 1.)

        k = rng.uniform(-x,x,size=filter_shape).astype(theano.config.floatX)
        b = np.zeros(filter_shape[0],dtype=theano.config.floatX)

        self.K = theano.shared(k,'K'+str(c))
        self.B = theano.shared(b,'C'+str(c))
        self.params = [self.K,self.B]

        convolution = T.nnet.conv.conv2d(inputs,self.K)
        pooled = T.signal.pool.pool_2d(convolution,poolsize,ignore_border=True)

        self.output = Tool.functions[activation](pooled + self.B.dimshuffle('x', 0, 'x', 'x'))
        self.shape = np.array(self.output.shape)

class RecurrentLayer:
    """
    Recurrent layer class.
    # Arguments :
        : inputs : ndarray or T.tensor : Previous layer

        : channels : int : Word size or number of Channels

        : hdim : int : Dimension of hidden layer

        : activation : string : Activation function (None,sigmoid,tanh,relu,softmax)

    # Attributes :
        : params : list : List of all the parameters of the layer

        : output : ndarray or T.tensor : Layer's output : size = hdim

    # Functions :
    """
    c = 0
    def __init__(self,inputs,channels,hdim,truncate=-1,activation='tanh'):

        assert activation in Tool.functions.keys()

        RecurrentLayer.c += 1
        c = RecurrentLayer.c

        x = np.sqrt(1. / (channels+hdim))
        y = np.sqrt(1./hdim)

        w = rng.uniform(-x, x, (channels,hdim)).astype(theano.config.floatX)
        v = rng.uniform(-y,y,size=(hdim,hdim)).astype(theano.config.floatX)

        h0 = theano.shared(np.zeros((hdim),dtype=theano.config.floatX))
        b = np.zeros(hdim,dtype=theano.config.floatX)

        self.W = theano.shared(w,'W'+str(c))
        self.V = theano.shared(v,'V'+str(c))
        self.B = theano.shared(b,'B'+str(c))
        self.params = [self.W,self.V,self.B]

        H, _ = theano.scan(self._step,
                           sequences=inputs,
                           non_sequences=self.params,
                           outputs_info=[T.repeat(h0[None, :],inputs.shape[1], axis=0)],
                           truncate_gradient=truncate,
                           strict=True)

        self.output = Tool.functions[activation](H[-1])

    def _step(self, x, h_prev, W, V, B):
        """
        Updates cell state.
        """
        return T.tanh(T.dot(x,W) + T.dot(h_prev,V) + B)

class LSTMLayer:
    """
    LSTM layer class.
    # Arguments :
        : inputs : ndarray or T.tensor : Previous layer

        : channels : int : Word size or number of Channels

        : hdim : int : Dimension of hidden layer

        : activation : string : Activation function (None,sigmoid,tanh,relu,softmax)

    # Attributes :
        : params : list : List of all the parameters of the layer

        : output : ndarray or T.tensor : Layer's output : size = hdim

    # Functions :
    """
    c = 0
    def __init__(self,inputs,channels,hdim,truncate=-1,activation='tanh'):

        assert activation in Tool.functions.keys()

        LSTMLayer.c += 1
        c = LSTMLayer.c

        x = np.sqrt(1. / (channels+hdim))
        y = np.sqrt(1./hdim)

        keys = ['i','f','c','o'] # Input, Forget, Cell, Output
        self.W, self.U, self.B = {},{},{}

        for i in keys:
            self.W[i] = rng.uniform(-x, x, (channels,hdim)).astype(theano.config.floatX)
            self.W[i] = theano.shared(self.W[i],'W'+str(i)+str(c))

            self.U[i] = rng.uniform(-y,y,size=(hdim,hdim)).astype(theano.config.floatX)
            self.U[i] = theano.shared(self.W[i],'U'+str(i)+str(c))

            self.B[i] = np.zeros(hdim,dtype=theano.config.floatX)
            self.B[i] = theano.shared(self.W[i],'B'+str(i)+str(c))

        h0 = theano.shared(np.zeros((hdim),dtype=theano.config.floatX))
        self.V = 'Not implemented'

#        self.params = list(self.W.values()) + list(self.U.values()) + list(self.B.values()) + list(self.V)
        self.params = [self.W,self.U,self.V,self.B]

        H, _ = theano.scan(self._step,
                           sequences=inputs,
                           non_sequences=self.params,
                           outputs_info=[T.repeat(h0[None, :],inputs.shape[1], axis=0)],
                           truncate_gradient=truncate,
                           strict=True)

        self.output = Tool.functions[activation](H[-1])

    def _step(self, xt, ht_prev, c_prev, W, U, V, B):

        i = T.nnet.sigmoid(T.dot(W['i'],xt) + T.dot(U['i'],ht_prev) + B['i']) # Input gate
        f = T.nnet.sigmoid(T.dot(W['f'],xt) + T.dot(U['f'],ht_prev) + B['f']) # Forget gate
        c = T.tanh(T.dot(W['c'],xt) + T.dot(U['c'],ht_prev) + B['c']) # Intermediary state
        c = i * c + f * c_prev # Cell state !
        o = T.nnet.sigmoid(T.dot(W['o'],xt) + T.dot(U['o'],ht_prev) + T.dot(V,c) + B['o']) # Output gate

        return o * T.tanh(c) # Final output

class Dropout:
    """
    Dropout layer class.
    # Arguments :
        : weight : ndarray or T.tensor : The weights we want to drop out

        : phase : string ('train','test' or 'valid') : Train or Valid phase ?

        : drop : float32 : Proportion to dropout from the weight

        : seed : int : Random seed for generator (optional)
    """
    def __init__(self,weight,phase,drop=.5):

        assert phase in ['train','test','valid']
        self.drop = drop
        self.srng = RandomStreams(rng.randint(2**31))
        self.output = self._drop(weight) if phase=='train' else self._scale(weight)

    def _drop(self, weight):
        """
        # Returns: Dropped out matrix with binomial probability
        """
        mask = self.srng.binomial(n=1, p=1-self.drop, size=weight.shape, dtype=theano.config.floatX)
        return T.cast(weight * mask, theano.config.floatX)

    def _scale(self, weight):
        """
        # Returns: Scaled matrix
        """
        return (1 - self.drop) * T.cast(weight, theano.config.floatX)


