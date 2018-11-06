#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 01:29:17 2018

@author: imlab
"""


import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import sklearn.model_selection as sk
import scipy.io as sio
import time 

X_train, X_S, Y_train, Y_S = sk.train_test_split(YouTubeActions_TotalFeatures,DatabaseLabel,test_size=0.20,random_state = 42 ) #, shuffle=False



X_train, X_Validation, Y_train, Y_Validation = sk.train_test_split(X_train,Y_train,test_size=0.20,random_state = 42 ) #, shuffle=False
X_train=X_train
Y_train=Y_train
X_Validation=X_Validation
Y_Validation=Y_Validation



hm_epochs = 500
n_classes = 12
batch_size = 128
batch_size_val=64
chunk_size =1024
n_chunks =15
rnn_size = 256
 


trainSamples,FeaturesLength=Y_train.shape
ValidationSamples,FeaturesLength=Y_Validation.shape
loss=[];
Val_Accuracy=[];   

with tf.name_scope('Inputs'):
    x = tf.placeholder('float', [None, n_chunks,chunk_size],name="Features")
    y = tf.placeholder('float',name="Lables")

def recurrent_neural_network(x):
    
    
 #####################################################################

  
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
    

    t = time.time()
    
    prediction= recurrent_neural_network(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    best_accuracy = 0.0
   
    
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits
                      (logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    with tf.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
        tf.device('/gpu:0')
        sess.run(tf.global_variables_initializer())
        
       
#        print(sess.run(weights))
                          
        kk=0
        for epoch in range(hm_epochs):
            epoch_loss = 0
            valdd=[]  
            k=0;
            for _ in range(int(trainSamples/batch_size)):
                epoch_x = X_train[k:k+batch_size,:]
                epoch_y = Y_train[k:k+batch_size,:]
                epoch_x= epoch_x.reshape((batch_size, n_chunks, chunk_size ))
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
                k=k+batch_size
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            loss.append(epoch_loss)
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            
            kk=0
            for _ in range(int(ValidationSamples/batch_size_val)):
                valdd.append(accuracy.eval({x:X_Validation[kk:kk+batch_size_val,:].reshape((-1,n_chunks, chunk_size)), y:Y_Validation[kk:kk+batch_size_val,:]}))
                kk = kk+batch_size_val
                if kk > ValidationSamples:
                    kk=0
                    
                    

            accuracy_out=np.mean(valdd)
            Val_Accuracy.append(accuracy_out)
            print('Validation Accuracy : ',accuracy_out,'  ||| Best Accuracy :',best_accuracy)
            if  accuracy_out > best_accuracy:
                    best_accuracy=accuracy_out
                    saver = tf.train.Saver() 
                    save_path = saver.save(sess, "/media/imlab/IMLab Server Data/Ubuntu/AminUllah/Optical Flow Net/flownet2-master/scripts/YouTube model/model.chk")
                    print("Model saved in file: %s" % save_path)
                    
                    
            
            PreLabels=sess.run(tf.argmax(prediction,1), feed_dict={x: X_S.reshape((-1,n_chunks, chunk_size))})
            Labels = Y_S.argmax(axis=1)
            confusion = tf.confusion_matrix(Labels, PreLabels).eval()
        elapsed = time.time() - t
        print('elapsed Time : ', elapsed)    
        return PreLabels, Labels, confusion
                    
         



        
        #Save the variables to disk.
#        save_path = saver.save(sess, "D:\\Speech Project\\Dataset\\BerlinImages\\BerlinImages\\1_Singleimages\\RNN Model For 257x45 double data spects\\model.ckpt")
#        print("Best Accuracy ==  " ,best_accuracy)
       # merged = tf.summary.merge_all()
       # writer=tf.summary.FileWriter("C:\\Users\\AMIN\\Anaconda2\\envs\\py35\\Lib\\site-packages\\tensorflow\\tensorboard\\otherLogs",sess.graph)
        

PreLabels, Labels, confusion = train_recurrnet_neural_network(x)


#sio.savemat('./YouTube model/PreLabels.mat', mdict={'PreLabels': PreLabels})
#sio.savemat('./YouTube model/Labels.mat', mdict={'Labels': Labels})   
#sio.savemat('./YouTube model/confusion.mat', mdict={'confusion': confusion})
