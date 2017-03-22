#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:01:33 2017

@author: mcomin
"""
import time
import numpy as np
import numpy.random as rng
import theano
import theano.tensor as T
import lasagne
from theano.tensor.nnet.abstract_conv import conv2d_grad_wrt_inputs, AbstractConv2d_gradInputs
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
        return T.sum(T.sqrt(T.pow(param,2)))

    @staticmethod
    def L2(param):
        return T.sum(T.pow(param,2))
    
    @staticmethod
    def MSE(param):
        return T.mean(T.pow(param,2))



class Tool:
    """
    Class containing miscellaneous utilities :
    ------------------------------------------
    
        Functions : Activation functions and Optimizers from lasage/theano
        Methods : Upsampling methods, seeds, ...
        Classes : Timers, ...
    """

########################## Functions #############################

    tanh = T.tanh
    sigmoid = T.nnet.sigmoid
    softmax = T.nnet.softmax
    relu = T.nnet.relu
    elu = lambda x :T.switch(x > 0, x, T.exp(x) - 1)
    softplus = T.nnet.softplus
    conv = T.nnet.conv2d
    pool = T.signal.pool.pool_2d
    upsamp = T.nnet.abstract_conv.bilinear_upsampling
    deconv = T.nnet.abstract_conv.conv2d_grad_wrt_inputs
    
    sgd = lasagne.updates.sgd
    momentum = lasagne.updates.momentum
    nesterov = lasagne.updates.nesterov_momentum
    adagrad = lasagne.updates.adagrad
    rmsprop = lasagne.updates.rmsprop
    adadelta = lasagne.updates.adadelta
    adam = lasagne.updates.adam
    
    fct = {None:lambda x,**args: x,
                    'sigmoid':T.nnet.sigmoid,
                    'tanh':T.tanh,
                    'softmax':T.nnet.softmax,
                    'relu':T.nnet.relu,
                    'elu' : elu,
                    'softplus':T.nnet.softplus}

    opt = {'sgd':lasagne.updates.sgd,
                  'momentum':lasagne.updates.momentum,
                  'nesterov':lasagne.updates.nesterov_momentum,
                  'adagrad':lasagne.updates.adagrad,
                  'rmsprop':lasagne.updates.rmsprop,
                  'adadelta':lasagne.updates.adadelta,
                  'adam':lasagne.updates.adam,}


########################### Methods ##############################

    @staticmethod
    def perforated(self, inp, ratio, **args):
        output_shape = [inp.shape[1], inp.shape[2] * ratio, inp.shape[3] * ratio]
        
        stride = inp.shape[2]
        offset = inp.shape[3]

        upsamp_matrix = T.zeros((stride * offset, stride * offset * ratio**2))

        rows = T.arange(stride * offset)
        cols = T.cast(rows * ratio + (rows / stride * ratio * offset), 'int32')

        upsamp_matrix = T.set_subtensor(upsamp_matrix[rows, cols], 1.)

        flat = T.reshape(inp, (inp.shape[0], output_shape[0], inp.shape[2] * inp.shape[3]))
        up_flat = T.dot(flat, upsamp_matrix)

        upsamp = T.reshape(up_flat, (inp.shape[0], output_shape[0], output_shape[1], output_shape[2]))
        return upsamp

    @staticmethod    
    def bilinear(self,inp,ratio,batch_size,num_input_chan):
        return T.nnet.abstract_conv.bilinear_upsampling(inp,ratio,batch_size,num_input_chan)

    @staticmethod 
    def repeat(self, inp, ratio):
        x_dims = inp.ndim
        output = inp
        for i, factor in enumerate(us[::-1]):
            if factor > 1:
                output = T.repeat(output, ratio, x_dims - i - 1)
        return output

    @staticmethod
    def setinit(fan_in, fan_out, act, size=None):
        if not size:
            size = (fan_in,fan_out)
        x = np.sqrt(6. / (fan_in + fan_out)) * (4. if act == 'sigmoid' else 1.)
        return rng.uniform(-x,x,size=size).astype(theano.config.floatX)

    @staticmethod
    def setseed():
        raise NotImplementedError

########################### Classes ##############################

    class Timer:
        """
        Embedding timer class. Example:
        with Timer() as t:
            run_code
        print('Code took %.03f sec.' % t.interval)
        """
        def __enter__(self):
            self.start = time.clock()
            return self
    
        def __exit__(self, *args):
            self.end = time.clock()
            self.interval = self.end - self.start

#*****************************************************************************#
#*****************************************************************************#

class DenseLayer:
    """
    Fully-connected dense layer class.
    ----------------------------------

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

        assert activation in Tool.fct.keys()

        DenseLayer.c += 1
        c = DenseLayer.c

        inputs = inputs.flatten(2)

        w = Tool.setinit(n_in,n_out,activation)
        b = np.zeros(n_out,dtype=theano.config.floatX)

        W = theano.shared(w,'W'+str(c))
        B = theano.shared(b,'B'+str(c))
        self.params = [W,B]
        self.weights = [W]
        self.output = Tool.fct[activation](T.dot(inputs,W) + B)



class ConvLayer:
    """
    Convolutional + Maxpooling layer class.
    ---------------------------------------
    
    # Arguments :
        : inputs : 4D tensor : Shape must be (batch size, channels, height, width)

        : nkernels : int : Number of kernels
        
        : kernerlsize : 2-tuple : Height and width of kernel

        : poolsize : 2-tuple : Height and width for maxpooling

        : act : string : Activation function (None,sigmoid,tanh,relu,softmax)
        
        : pad : string, int, tuple : Padding mode ('full','half','valid') or (pad height, pad width)
        
        : stride : int, tuple : Strides
        
        : pmode : string : 'max' or 'average_inc_pad'

    # Attributes :
        : params : list : List of all the parameters of the layer

        : output : ndarray or T.tensor : size = outshape

    # Functions : 
        : outshape : Returns the shape of the output image
    """
    c = 0
    def __init__(self,inputs,nchan,nkernels,kernelsize,poolsize=(2,2),
                 activation='relu',pad='valid',stride=(1,1), pmode='max',
                 pstride=None,ppad=(0,0)):

        assert activation in Tool.fct.keys()

        ConvLayer.c += 1
        c = ConvLayer.c
        
        filter_shape = (nkernels,nchan) + kernelsize

        fan_in = nchan * np.prod(kernelsize)
        fan_out = (nkernels * np.prod(kernelsize)) // np.prod(poolsize)

        w = Tool.setinit(fan_in, fan_out, activation, size = filter_shape)
        b = np.zeros(nkernels,dtype=theano.config.floatX)

        W = theano.shared(w,'Wconv'+str(c))
        B = theano.shared(b,'Bconv'+str(c))

        convolution = Tool.conv(inputs, W, border_mode = pad, subsample = stride)
        out = Tool.pool(convolution, poolsize, True, pstride, ppad, pmode)

        self.params = [W,B]
        self.weights = [W]
        self.shape = lambda x: self.outshape(x, kernelsize[0], pad, stride[0], poolsize[0])
        self.arguments = {'nchan':nchan, 'nkernels':nkernels,'kernelsize':kernelsize,'poolsize':poolsize,
                          'activation':activation,'pad':pad,'stride':stride,'W':W,'B':B}
        
        self.output = Tool.fct[activation](out + B.dimshuffle('x', 0, 'x', 'x'))

    def outshape(self,inp, k, p, s, pool):
        if p == 'valid':
            return int((np.floor((inp - k)/s) + 1)/pool)
        elif p == 'full':
            return int((np.floor((inp - k + 2*(k-1))/s) + 1)/pool)
        elif p == 'half':
            return int((np.floor((inp - k + 2*(k//2))/s) + 1)/pool)
        else:
            return int(np.floor(((inp - k + 2*p)/s) + 1 )/pool)



class TConvLayer:
    """
    Transposed Convolutional + Upsampling layer class.
    --------------------------------------------------
    IMPORTANT NOTE
    --------------
    This class is implemented to perform the transposed convolution of a given ConvLayer.
    All the arguments except the first two should be provided by the corresponding ConvLayer :
    
        TransposeConvLayer(some_input, shape, **direct_conv_layer.arguments)

    # Arguments :
        : inputs : 4D tensor : Previous layer
        
        : shape : tuple : Shape of the corresponding convolution
            
        : **args : dict : Corresponding direct convolution layer arguments

        : tied : bool : If true, will use the same kernels and biases as the direct convolution

    # Attributes :
        : params : list : List of all the parameters of the layer

        : output : ndarray or T.tensor : size = floor((i + 2*pmode - kernelsize) / stride) + 1
    
    !!! PROLEM WITH BIAS DIMENSIONALITY !!!
    """
    c=0
    def __init__(self,inputs,shape,nchan,nkernels,kernelsize,W,B,poolsize,
                 activation,pad,stride,batch,tied=True):

        assert activation in Tool.fct.keys()

        TConvLayer.c += 1
        c = TConvLayer.c

        filter_shape = (nkernels,nchan) + kernelsize
        inp_shape = (None,nchan) + shape

        if tied:
            W = theano.shared(W.eval(),'Wdconv'+str(c))
            B = theano.shared(B.eval(),'Bdconv'+str(c))
        else:
            del W,B 

            fan_in = nchan * np.prod(kernelsize)
            fan_out = (nkernels * np.prod(kernelsize)) // np.prod(poolsize)
    
            w = Tool.setinit(fan_in, fan_out, activation, size = filter_shape)
            b = np.zeros(nkernels,dtype=theano.config.floatX)
    
            W = theano.shared(w,'Wdconv'+str(c))
            B = theano.shared(b,'Bdconv'+str(c))

        upsampled = Tool.upsamp(inputs, poolsize[0], batch, nkernels) # batch =  inputs.shape[0]
        deconved = Tool.deconv(upsampled, W, inp_shape, border_mode=pad, subsample=stride)

        self.params = [W] #[W,B]
        self.weights = [W]
        self.output = Tool.fct[activation](deconved ) # + B)



class RecurrentLayer:
    """
    Recurrent layer class.
    ----------------------
    
    # Arguments :
        : inputs : ndarray or T.tensor : Previous layer

        : channels : int : Word size or number of Channels

        : hdim : int : Dimension of hidden layer

        : activation : string : Activation function (None,sigmoid,tanh,relu,softmax)

    # Attributes :
        : params : list : List of all the parameters of the layer

        : output : ndarray or T.tensor : size = hdim

    # Functions :
        : __step__ : Updates cell state
    """
    c = 0
    def __init__(self,inputs,channels,hdim,truncate=-1,activation='tanh'):

        assert activation in Tool.fct.keys()

        RecurrentLayer.c += 1
        c = RecurrentLayer.c

        w = Tool.setinit(channels, hdim, activation)
        v = Tool.setinit(hdim, hdim, activation)

        h0 = theano.shared(np.zeros((hdim), dtype=theano.config.floatX),'h0')
        b = np.zeros(hdim, dtype=theano.config.floatX)

        W = theano.shared(w,'Wrec'+str(c))
        V = theano.shared(v,'Vrec'+str(c))
        B = theano.shared(b,'Brec'+str(c))
        self.params = [W,V,B]
        self.weights = [W,V]

        H, _ = theano.scan(self.__step__,
                           sequences=inputs,
                           non_sequences=self.params,
                           outputs_info=[T.repeat(h0[None, :],inputs.shape[1], axis=0)],
                           truncate_gradient=truncate,
                           strict=True)

        self.output = Tool.fct[activation](H[-1])

    def __step__(self, x, h_prev, W, V, B):
        return T.tanh(T.dot(x,W) + T.dot(h_prev,V) + B)



class LSTMLayer:
    """
    LSTM layer class.
    -----------------
    
    # Arguments :
        : inputs : ndarray or T.tensor : Previous layer
    
        : channels : int : Word size or number of Channels
    
        : hdim : int : Dimension of hidden layer
    
        : activation : string : Activation function (None,sigmoid,tanh,relu,softmax)
    
    # Attributes :
        : params : list : List of all the parameters of the layer
    
        : output : ndarray or T.tensor : size = hdim
    
    # Functions :
        : __step__ : Updates cell state
    """
    c = 0
    def __init__(self,inputs,channels,hdim,truncate=-1,activation='tanh'):

        assert activation in Tool.fct.keys()

        LSTMLayer.c += 1
        c = LSTMLayer.c

        x = np.sqrt(1. / (channels+hdim))
        y = np.sqrt(1. / hdim)

        self.V = Tool.setinit(hdim, hdim, activation)
        self.V = theano.shared(self.V,'V'+str(c))

        self.W, self.U, self.B = {},{},{}

        for k in ['i','f','c','o']: # Input, Forget, Cell, Output
            self.W[k] = Tool.setinit(channels, hdim, activation)
            self.W[k] = theano.shared(self.W[k],'Wlstm'+str(k)+str(c))

            self.U[k] = Tool.setinit(hdim, hdim, activation)
            self.U[k] = theano.shared(self.U[k],'Ulstm'+str(k)+str(c))

            self.B[k] = np.zeros(hdim, dtype=theano.config.floatX)
            self.B[k] = theano.shared(self.B[k],'Blstm'+str(k)+str(c))

        h0 = theano.shared(np.zeros((hdim),dtype=theano.config.floatX),'h0')
        c0 = theano.shared(np.zeros((hdim),dtype=theano.config.floatX),'c0')

        val = lambda x : [x[k] for k in ['i','f','c','o']] #list(x.values())
        self.params = val(self.W) + val(self.U) + val(self.B)
        self.weights = val(self.W) + val(self.U)

        [H,C], _ = theano.scan(self.__step__,
                           sequences=inputs,
                           non_sequences=self.params,
                           outputs_info=[T.repeat(h0[None, :],inputs.shape[1], axis=0),
                                         T.repeat(c0[None, :], inputs.shape[1], axis=0)],
                           truncate_gradient=truncate,
                           strict=True)

        self.output = Tool.fct[activation](H[-1])
        self.state = C[-1]

    def __step__(self, xt, h_prev, c_prev, *yolo):
        # Not taking the parameters from scan but from self (Should not make a difference)

        i,f,c,o = [T.nnet.sigmoid(T.dot(xt,self.W[k]) + T.dot(h_prev,self.U[k]) + self.B[k])
                    for k in ['i','f','c','o']]

        c = i * c + f * c_prev
        h = o * T.tanh(c)
        
        return h,c



class Dropout:
    """
    Dropout layer class.
    --------------------
    
    # Arguments :
        : weight : ndarray or T.tensor : The weights we want to drop out

        : train : True if training phase, False else

        : drop : float32 : Proportion to dropout from the weight

        : seed : int : Random seed for generator (optional)
    """
    def __init__(self,weight,drop=.5,train=True):

        self.drop = drop
        self.srng = RandomStreams(rng.randint(2**31))
        self.output = self.__drop__(weight) if train else self.__scale__(weight)

    def __drop__(self, weight):
        """
        # Returns: Dropped out matrix with binomial probability
        """
        mask = self.srng.binomial(n=1, p=1-self.drop, size=weight.shape, dtype=theano.config.floatX)
        return T.cast(weight * mask, theano.config.floatX)

    def __scale__(self, weight):
        """
        # Returns: Scaled matrix
        """
        return (1 - self.drop) * T.cast(weight, theano.config.floatX)

