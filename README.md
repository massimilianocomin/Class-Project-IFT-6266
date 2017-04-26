# Class-Project-IFT-6266
Class project for IFT6266 : Conditional Image Generation


This repository contains the necessary codes for the class project of IFT6266 (Deep Learning course at Université de Montréal, Winter 2017).

The jobs are computed using the Hades GPU cluster of CalculQuébec. The submit.sh file is the script used to run jobs on the cluster. It calls Launch.py, a genereric model launcher.

The 'code' directory contains the different algorithms I used for this project. All the codes are commentend and documented.

The 'lib' directory contains generic layers (convolutional, LSTM, ..) and dataset (loading images, captions..) utilities shared by all codes e.g.

### Layers :
Here there are many classes used to build the networks. Each class corresponds to a given layer and they are coded such that one can stack them easily. Everything is commented and described in the class docstrings.
The Tool class is a utility class containing many different functions from either lasagne or theano. It is somewhat useless as it sometimes only changes a function's name, but I've done it so that everything can be found at the same place.

### Img : 
Here there is a single class whose methods are used to load mini-batches of data, plot results and save images.

### Captions :
This module is used only to train Word2Vec algorithm over the caption dictionary, for we need a text embedding to use the captions.

### Codes : 
The description of the codes and their results is in the blog : https://ift6266mcomin.wordpress.com/
