Activity Recognition using Temporal Optical Flow Convolutional Features and Multi-Layer LSTM (Caffe for FlowNet2 )
==================

This code is used for just research work:
 - If you want to use this code for commecial purpose please see the license information of Flownet2 in the following link
 - https://github.com/lmb-freiburg/flownet2 

It comes as a fork of the caffe master branch and with trained networks,
as well as examples to use and train them.

Citation
====================
1. Our accepted work will be online soon
2. https://github.com/lmb-freiburg/flownet2


Compiling
=========

First compile caffe, by configuring a

    "Makefile.config" (example given in Makefile.config.example)

then make with 

    $ make -j 5 all tools pycaffe 


Running 
=======

(this assumes you compiled the code sucessfully) 

IMPORTANT: make sure there is no other caffe version in your python and 
system paths and set up your environment with: 

    $ source set-env.sh 

This will configure all paths for you. Then go to the model folder 
and download models: 

    $ cd models 
    $ ./download-models.sh 
 
Extract sequancel temporal optical flow features for activity recogntion dataset: 

    $ scripts/FullDatasetFeatures.py
    change paths: line 16,18,19

Training
========

(this assumes you compiled the code sucessfully) 

First you need to download and prepare the training data using FullDatasetFeatures.py

Learning from sequanctal temporal optical flow features using multi-layer LSTM: 

    $ scripts/LSTM.py 
    Change path to save the trained model line 147

Testing video using trained multi-layer LSM: 

    $ scripts/OneFileOnlineTest.py 
    Change paths: Line 40, 62, 63, 175







