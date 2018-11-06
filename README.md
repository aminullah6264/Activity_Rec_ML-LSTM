Activity Recognition using Temporal Optical Flow Convolutional Features and Multi-Layer LSTM 
==================

This code can only be used for research purposes:
 - If you want to use this code for commercial purpose, please see the license information of Flownet2 in the following link
 - https://github.com/lmb-freiburg/flownet2 


Citation
====================
1. Our accepted work will be online soon
2. https://github.com/lmb-freiburg/flownet2


Compiling
=========

First compile caffe, by configuring 

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
 
Features Extraction 
========

Extract temporal optical flow features from activity recogntion datasets: 
*Activity recogntion datasets can be downloaded from the following Links
- http://crcv.ucf.edu/data/UCF101.php (UCF101)
- http://crcv.ucf.edu/data/UCF50.php  (UCF50)

    $ python scripts/Features_Extraction.py
    Change paths in code: Line No 16,18,19

Training
========

First you need to prepare the training data using Features_Extraction.py

    $ python scripts/Training_ML_LSTM.py 
    Change path in code: Line No. 147

Testing
========

Testing video using trained multi-layer LSTM 

    $ scripts/Video_Testing.py 
    Change paths: Line 40, 62, 63, 175







