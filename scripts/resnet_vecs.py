import numpy as np
import keras
import os
import cv2
import matplotlib.pyplot as plt
from keras.applications import ResNet50
from keras.models import Model
from datetime import datetime

datadir = 'original'
traindir = './train/' + datadir + '/'
testdir = './test/' + datadir + '/'
labels_code = {'nystagmus': 1, 'normal': 0}
bs = 20

def get_paths(datapath):
    labels = []
    paths = []
    for l in os.listdir(datapath):
        ldir = datapath + l + '/'
        for p in os.listdir(ldir):
            vidname = p.split('-')[0] + p.split('-')[1]
            label = labels_code[l]
            labels.append([vidname, label])
            paths.append(ldir+p)
    return labels, paths

if __name__=="__main__":
    scriptStart = datetime.now()
    
    labels, paths = get_paths(traindir)
    test_labels, test_paths = get_paths(testdir)
    
    model = ResNet50(weights='imagenet', include_top=True)
    
    vecs_all = np.empty([0])
    for i in range(0,len(labels),bs):
        batch = []
        for path in paths[i:i+bs]:
            im = cv2.imread(path)
            im = cv2.resize(im, (224,224))
            batch.append(im)
        batch = np.array(batch)
        vecs = model.predict(batch)
        if i < bs:
            vecs_all = vecs
        else:
            vecs_all = np.concatenate((vecs_all, vecs),axis=0)
        
    pred_name = './vectors/resnet50_train.npy'
    np.save(pred_name, vecs_all)
    labels_name = './vectors/resnet50_trainlabels.npy'
    np.save(labels_name, np.array(labels))
    
    vecs_all = np.empty([0])
    for i in range(0,len(test_labels),bs):
        batch = []
        for path in test_paths[i:i+bs]:
            im = cv2.imread(path)
            im = cv2.resize(im, (224,224))
            batch.append(im)
        batch = np.array(batch)
        vecs = model.predict(batch)
        if i < bs:
            vecs_all = vecs
        else:
            vecs_all = np.concatenate((vecs_all, vecs),axis=0)
        
    pred_name = './vectors/resnet50_test.npy'
    np.save(pred_name, vecs_all)
    labels_name = './vectors/resnet50_testlabels.npy'
    np.save(labels_name, np.array(test_labels))
    
    print('Total script time: ', datetime.now()-scriptStart)