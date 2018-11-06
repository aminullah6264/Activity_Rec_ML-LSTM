

from __future__ import print_function

import os, sys, numpy as np
import argparse
from scipy import misc
import caffe
import tempfile
from math import ceil
import cv2
import scipy.io as sio
from matplotlib import pyplot as plt


DatasetFolder = '/media/imlab/IMLab Server Data/Datasets/UCF50'

caffemodel = '/media/imlab/IMLab Server Data/Ubuntu/AminUllah/Optical Flow Net/flownet2-master/models/FlowNet2/FlowNet2_weights.caffemodel.h5'
deployproto = '/media/imlab/IMLab Server Data/Ubuntu/AminUllah/Optical Flow Net/flownet2-master/models/FlowNet2/FlowNet2_deploy.prototxt.template'
gpu = 0
verbose ='store_true'

#
#
if(not os.path.exists(caffemodel)): raise BaseException('caffemodel does not exist: '+caffemodel)
if(not os.path.exists(deployproto)): raise BaseException('deploy-proto does not exist: '+deployproto)



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

DatabaseFeautres = []
DatabaseLabel = []

for folderName in os.listdir(DatasetFolder):
    print(folderName)
    subFolder = DatasetFolder+'/'+ folderName
    for filename in os.listdir(subFolder):
        vidcap = cv2.VideoCapture(DatasetFolder+'/'+ folderName +'/'+filename)
        print('Feature Extraction of : ',filename)
        videolength = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        videoFeatures=[]
        frame_no=-1;
        
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
            
    #            i = 1
    #            while i<=5:
    #                i+=1
    #            
                net.forward(**input_dict)
                
    #                containsNaN = False
    #                for name in net.blobs:
    #                    blob = net.blobs[name]
    #        #            print(name)
    #                    has_nan = np.isnan(blob.data[...]).any()
    #            
    #                    if has_nan:
    #        #                print('blob %s contains nan' % name)
    #                        containsNaN = True
    #            
    #                if not containsNaN:
    #        #            print('Succeeded.')
    #                    break
    #                else:
    #                    print('**************** FOUND NANs, RETRYING ****************')
                
                blob1 = np.squeeze(net.blobs['conv6_1'].data).transpose(1, 2, 0)
    #            kk=0
    #            features = []
    #            while kk < 1024:
    #                features.append(np.amax(blob1[:,:,kk]))
    #                kk = kk + 1
                arr = np.array(blob1)
                arrReshap = arr.reshape([56,1024])
                bb = np.matrix(arrReshap)
                features = bb.max(0)
                
                
                videoFeatures.append(features)
    #            print(frame_no % 30)
                if frame_no % 30 == 29:
                    aa = np.asarray(videoFeatures)
                    DatabaseFeautres.append(aa)
                    DatabaseLabel.append(folderName)
                    videoFeatures=[]
            
       




#np.save('DatabaseFeaturesList',DatabaseFeautres)
#np.save('DatabaseLabelList',DatabaseLabel)

###################### One Hot and Train Test spilt
TotalFeatures= []
for sample in DatabaseFeautres:
    TotalFeatures.append(sample.reshape([1,15360]))
    
    
TotalFeatures = np.asarray(TotalFeatures)
TotalFeatures = TotalFeatures.reshape([len(DatabaseFeautres),15360])


OneHotArray = []
kk=1;
for i in range(len(DatabaseFeautres)-1):
    OneHotArray.append(kk)
    if (DatabaseLabel[i] != DatabaseLabel[i+1]):
        kk=kk+1;


OneHot=  np.zeros([len(DatabaseFeautres),50], dtype='int');


for i in range(len(DatabaseFeautres)-1):
    print(i)
    OneHot[i,OneHotArray[i]-1] = 1



np.save('UCF50_TotalFeatures',TotalFeatures)
sio.savemat('UCF50_Labels.mat', mdict={'DatabaseLabel': OneHot})
sio.savemat('UCF101_TotalFeatures.mat', mdict={'TotalFeatures': TotalFeatures},appendmat=True, format='5',
    long_field_names=False, do_compression=True, oned_as='row')





