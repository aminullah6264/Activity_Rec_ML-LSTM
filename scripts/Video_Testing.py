#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 16:08:22 2018

@author: imlab
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 10:22:19 2018

@author: imlab
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 29 15:55:09 2017

@author: AMIN
"""

import tensorflow as tf
import os, numpy as np


import cv2
from Tkinter import Tk
from tkFileDialog import askopenfilename


import caffe
import tempfile
from math import ceil

import time


text_file = open("/media/imlab/IMLab Server Data/Ubuntu/AminUllah/Optical Flow Net/flownet2-master/scripts/UCF101 model/ClassNames.txt", "r")
ClassNames =text_file.readlines()


Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename(initialdir='/home/imlab/Downloads') # show an "Open" dialog box and return the path to the selected
patth = filename.split('/')
VideoName = patth[len(patth)-1]

vidcap = cv2.VideoCapture(filename)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('UCF101T.avi',fourcc, 30.0, (320,240))


font                   = cv2.FONT_HERSHEY_TRIPLEX
bottomLeftCornerOfText = (20,20)
fontScale              = 0.5
fontColor              = (255,255,255)
lineType               = 1

#vidcap = cv2.VideoCapture('/media/imlab/IMLab Server Data/Datasets/KTH Dataset/running/person01_running_d1_uncomp.avi')
caffemodel = '/media/imlab/IMLab Server Data/Ubuntu/AminUllah/Optical Flow Net/flownet2-master/models/FlowNet2/FlowNet2_weights.caffemodel.h5'
deployproto = '/media/imlab/IMLab Server Data/Ubuntu/AminUllah/Optical Flow Net/flownet2-master/models/FlowNet2/FlowNet2_deploy.prototxt.template'

gpu = 0
verbose ='store_true'

#
#
if(not os.path.exists(caffemodel)): raise BaseException('caffemodel does not exist: '+caffemodel)
if(not os.path.exists(deployproto)): raise BaseException('deploy-proto does not exist: '+deployproto)


videoFeatures=[]

width = 512
height = 386
vars = {}
vars['TARGET_WIDTH'] = width
vars['TARGET_HEIGHT'] = height

divisor = 64.
vars['ADAPTED_WIDTH'] = int(ceil(width/divisor) * divisor)
vars['ADAPTED_HEIGHT'] = int(ceil(height/divisor) * divisor)

vars['SCALE_WIDTH'] = width / float(vars['ADAPTED_WIDTH']);
vars['SCALE_HEIGHT'] = height / float(vars['ADAPTED_HEIGHT']);

tmp = tempfile.NamedTemporaryFile(mode='w', delete=True)

proto = open(deployproto).readlines()
for line in proto:
    for key, value in vars.items():
        tag = "$%s$" % key
        line = line.replace(tag, str(value))

    tmp.write(line)

tmp.flush()

if not verbose:
    caffe.set_logging_disabled()
caffe.set_device(gpu)
caffe.set_mode_gpu()
net = caffe.Net(tmp.name, caffemodel, caffe.TEST)
#net = caffe.Net(deployproto, caffemodel, caffe.TEST)
num_blobs = 2
input_data = []
   


n_classes = 101
chunk_size =1024
n_chunks =15
rnn_size = 256
 

n_nodes_hl1 = 256
n_nodes_hl2 = 128
n_nodes_hl3 = 64




x = tf.placeholder('float', [None, n_chunks,chunk_size])
y = tf.placeholder('float')


def recurrent_neural_network(x):
    
    
  
    W = {
            'hidden': tf.Variable(tf.random_normal([chunk_size, rnn_size])),
            'output': tf.Variable(tf.random_normal([rnn_size, n_classes]))
        }
    biases = {
            'hidden': tf.Variable(tf.random_normal([rnn_size], mean=1.0)),
            'output': tf.Variable(tf.random_normal([n_classes]))
        }


    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1,chunk_size])
    x = tf.nn.relu(tf.matmul(x, W['hidden']) + biases['hidden'])
    x = tf.split (x,n_chunks, 0)
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # Get LSTM cell output
    outputs, final_states = tf.contrib.rnn.static_rnn(lstm_cells, x, dtype=tf.float32)
    # Get last time step's output feature for a "many to one" style classifier,
    # as in the image describing RNNs at the top of this page
#    lstm_last_output=tf.transpose(outputs, [1,0,2])
    # Linear activation
    return tf.matmul(outputs[-1], W['output']) + biases['output']
    
#####################################################################  
    


def train_recurrnet_neural_network(x):

    prediction= recurrent_neural_network(x)
#    tf.device('/gpu:0')
    config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, "/media/imlab/IMLab Server Data/Ubuntu/AminUllah/Optical Flow Net/flownet2-master/scripts/UCF101 model/model.chk")
        #print(sess.run(tf.all_variables()))
        videolength = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        videoFeatures=[]
        frame_no=-1;
        startTime = time.time()
        while (frame_no < videolength-1):  #(videolength%30)
            input_data = []
            frame_no = frame_no + 1
            vidcap.set(1,frame_no)
            ret0,img0 = vidcap.read()
            
            frame_no = frame_no + 1
            vidcap.set(1,frame_no)
            ret1,img1 = vidcap.read()
            if(ret0 == 1 and ret1 == 1):
                img1 = cv2.resize(img1, (512, 386))
                img0 = cv2.resize(img0, (512, 386)) 
                if len(img0.shape) < 3: input_data.append(img0[np.newaxis, np.newaxis, :, :])
                else:                   input_data.append(img0[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
                
                if len(img1.shape) < 3: input_data.append(img1[np.newaxis, np.newaxis, :, :])
                else:                   input_data.append(img1[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
            
            
                input_dict = {}
                for blob_idx in range(num_blobs):
                    input_dict[net.inputs[blob_idx]] = input_data[blob_idx]
            
        
                net.forward(**input_dict)
                
        
                
                blob1 = np.squeeze(net.blobs['conv6_1'].data).transpose(1, 2, 0)
        
                arr = np.array(blob1)
                arrReshap = arr.reshape([56,1024])
                bb = np.matrix(arrReshap)
                features = bb.max(0)
                
                
                videoFeatures.append(features)
#                print(frame_no % 30)
                if frame_no % 30 == 29:
                    aa = np.asarray(videoFeatures)
                    X_test = aa.reshape([1,15360])
#                    labled =sess.run(tf.argmax(prediction,1), feed_dict={x: X_test.reshape((-1,n_chunks, chunk_size))})
                    labled =sess.run(prediction, feed_dict={x: X_test.reshape((-1,n_chunks, chunk_size))})
                    label = labled.argmax(axis=1)
                    Confidence = labled[0,label[0]]
#                    print(label[0],'    ', Confidence)
                    print('Writing resutls....')
                    i = frame_no-29
                    for kk  in range(30):
                        vidcap.set(1,i)
                        ret1,img1 = vidcap.read()
                        if(ret1 == 1):
                            cv2.putText(img1, 'Category : '+ClassNames[label[0]].replace('\n',' ') ,  bottomLeftCornerOfText,  font,  fontScale, fontColor, lineType) #+ '     Confidence: ' + str(Confidence)
                            out.write(img1)
                            i=i+1
                    videoFeatures=[]
        endTime = time.time()
        print ('total time taken', endTime - startTime)
#
train_recurrnet_neural_network(x)

out.release()
#cap = cv2.VideoCapture(filename,0)
#while True:
#    ret,img= cap.read()
#    if (ret == False):
#        break
#    else:
#        cv2.imshow(ClassNames[indexOfClass],img)
#        cv2.waitKey(50)


