import os
import cv2
import random
import numpy as np
import pandas as pd
from datetime import datetime
import keras
from keras import models
from keras.layers import TimeDistributed, Conv2D, Dense, MaxPooling2D, Flatten, LSTM, Dropout, BatchNormalization
from keras import optimizers


import pdb


datadir = '../data/images/frames/'
labelspath = '../data/splits.csv'
model_path = './models/frame-classification/cnn_lstm_model.h5'

# helper for reordering frames by timestamp
def ind(x):
    return int(x.split('_')[-1].split('.')[0])

def load_data(vids, labels):
    frames=[]
    corrected_labels = []
    for vid, label in zip(vids, labels):   
        vid_data=[]
        viddir = datadir + vid
        frames_to_select = []
        if os.path.exists(viddir):
            frames_to_select = [viddir + f for f in os.listdir(viddir)]
        if len(frames_to_select) == 600:
            sorted(frames_to_select, key=ind)
            for frame in frames_to_select:
                image=cv2.imread(frame,0)
                image=cv2.resize(image, (60, 60))
                dat=np.array(image)
                normu_dat=dat/255
                vid_data.append(normu_dat)
            vid_data=np.array(vid_data)
            frames.append(vid_data)
            corrected_labels.append(label)
    return np.array(frames), np.array(corrected_labels)

def create_model():
    model_cnlst = models.Sequential()
    model_cnlst.add(TimeDistributed(Conv2D(128, (3, 3), strides=(1,1),activation='relu'),input_shape=(600, 60, 60, 1)))
    model_cnlst.add(TimeDistributed(Conv2D(64, (3, 3), strides=(1,1),activation='relu')))
    model_cnlst.add(TimeDistributed(MaxPooling2D(2,2)))
    model_cnlst.add(TimeDistributed(Conv2D(64, (3, 3), strides=(1,1),activation='relu')))
    model_cnlst.add(TimeDistributed(Conv2D(32, (3, 3), strides=(1,1),activation='relu')))
    model_cnlst.add(TimeDistributed(MaxPooling2D(2,2)))
    model_cnlst.add(TimeDistributed(BatchNormalization()))


    model_cnlst.add(TimeDistributed(Flatten()))
    model_cnlst.add(Dropout(0.2))

    model_cnlst.add(LSTM(32,return_sequences=False,dropout=0.2)) # used 32 units

    model_cnlst.add(Dense(64,activation='relu'))
    model_cnlst.add(Dense(32,activation='relu'))
    model_cnlst.add(Dropout(0.2))
    model_cnlst.add(Dense(1, activation='sigmoid'))
    
    return model_cnlst

def get_samples(split_type):
    df = pd.read_csv(labelspath, index_col=0)
    
    split_df = df.loc[df['Split'] == split_type]
    labels = split_df['Label'].values
    paths = split_df['Video'].values
    
    temp = list(zip(paths, labels))
    random.shuffle(temp)
    paths, labels = zip(*temp)
    
    return paths, labels

if __name__=="__main__":
    scriptStart = datetime.now()
    
    # LOAD DATA
    loadStart = datetime.now()
    
    train_paths, train_labels = get_samples('Train')
    test_paths, test_labels = get_samples('Test')
    
    train_dataset, train_labels  = load_data(train_paths[:65], train_labels[:65])
    nb_train_samples = train_labels.shape[0]
    train_dataset = train_dataset.reshape((nb_train_samples, 600, 60, 60, 1))
    print('Train data shape: ', train_dataset.shape)
    print('Train labels shape: ', train_labels.shape)
    
    test_dataset, test_labels  = load_data(test_paths[:65], test_labels[:65])
    nb_test_samples = test_labels.shape[0]
    test_dataset = test_dataset.reshape((nb_test_samples, 600, 60, 60, 1))
    print('Test data shape: ', test_dataset.shape)
    print('Test labels shape: ', test_labels.shape)
    
    print('Total data loading time: ', datetime.now()-loadStart)
    
    # CREATE MODEL
    
    model_cnlst = create_model()
    callbacks_list_cnlst=[keras.callbacks.EarlyStopping(monitor='acc', 
                                                        patience=3), 
                          keras.callbacks.ModelCheckpoint(filepath=model_path, 
                                                          monitor='val_loss', 
                                                          save_best_only=True),
                          keras.callbacks.ReduceLROnPlateau(monitor = "val_loss", 
                                                            factor = 0.1, 
                                                            patience = 3)]
    optimizer=optimizers.RMSprop(lr=0.01)
    model_cnlst.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['acc'])
    print(model_cnlst.summary())
    
    # TRAIN MODEL
    trainStart = datetime.now()
    
    history_new_cnlst=model_cnlst.fit(train_dataset,
                                      train_labels,
                                      batch_size=2,
                                      epochs=5,
                                      callbacks=callbacks_list_cnlst)
    
    print('Total script time: ', datetime.now()-trainStart)
    
    
    # TEST MODEL
    testStart = datetime.now()
    
    preds = model_cnlst.predict(test_dataset, batch_size=2)
    print(preds.shape)
    print(preds)
    xy = np.concatenate((test_labels.reshape(-1,1), preds), axis=1)
    pred_name = './preds/predictions_FRAME_all.npy'
    np.save(pred_name, xy)
    
    print('Total script time: ', datetime.now()-testStart)
    
    print('Total script time: ', datetime.now()-scriptStart)
    