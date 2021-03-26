import numpy as np
import os
import sys
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

param1 = [5, 10, 50, 100, 500, 598, 600, 1000]
param2 = ['uniform', 'distance']
param3 = ['auto', 'ball_tree', 'kd_tree', 'brute']
param4 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
param5 = [1, 2, 3, 4, 5]
param6 = ['euclidean', 'manhattan', 'minkowski']

label_dict = {0: 'normal', 1: 'nystagmus'}

if __name__=="__main__":
    scriptStart = datetime.now()
    
    vecprefix = str(sys.argv[1])
    name = str(sys.argv[2])
    
    trainvecpath = './vectors/' + vecprefix + '_train.npy'
    trainlabpath = './vectors/' + vecprefix + '_trainlabels.npy'
    testvecpath = './vectors/' + vecprefix + '_test.npy'
    testlabpath = './vectors/' + vecprefix + '_testlabels.npy'
    
    trainx = np.load(trainvecpath)
    trainy = np.load(trainlabpath)

    testx = np.load(testvecpath)
    testy = np.load(testlabpath)

    print('Train x shape: ', trainx.shape)
    print('Train y shape: ', trainy.shape)
    print('Test x shape: ', testx.shape)
    print('Test y shape: ', testy.shape)
    
    n_neighbors = param1[2]
    weights = param2[0]
    algorithm = param3[0]
    leaf_size = param4[2]
    p = param5[1]
    metric = param6[2]
    
    print('Begin fitting...')
    trainStart = datetime.now()
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(trainx, trainy)
    
    print('Fitting complete!')
    print('Fitting time: ', datetime.now()-trainStart)
    
    evalStart = datetime.now()
    print('Evaluating test data...')
    
    fiprobs = knn.predict_proba(testx)
    
    print('Eval time: ', datetime.now()-evalStart)
    
    filenames = [label_dict[int(testy[i][1])] + '/' + testy[i][0] for i in range(testy.shape[0])]
    filenames = np.array(filenames)
    xy = np.concatenate((filenames.reshape(-1,1), fiprobs),axis=1)
    pred_name = './preds/predictions_'+name+'.npy'
    np.save(pred_name, xy)
    
    print('Total script time: ', datetime.now()-scriptStart)
    
    
