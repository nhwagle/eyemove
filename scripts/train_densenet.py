import pandas as pd
import numpy as np
import os
import sys
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import DenseNet121
from keras.preprocessing import image
from keras.applications.densenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from datetime import datetime

if __name__=="__main__":
    scriptStart = datetime.now()
    uniqueID = scriptStart.strftime('%H%M%S_%m-%d')
    #f = open('./stats/timing'+uniqueID+'.txt', 'w+')
    
    dataset_name = str(sys.argv[1])
    model_name = str(sys.argv[2])
    
    print('Loading base model')
    base_model=DenseNet121(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.
    print('Adding layers')
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dense(1024,activation='relu')(x) #dense layer 2
    x=Dense(512,activation='relu')(x) #dense layer 3
    preds=Dense(2,activation='softmax')(x) #final layer with softmax activation
    
    model=Model(inputs=base_model.input,outputs=preds)
    print('Model initialized')
    
    for layer in base_model.layers[:30]:
        layer.trainable=False
    for layer in model.layers[30:]:
        layer.trainable=True
    
    #datadir = '../../../data/Narayani/pcm/'    
    DATASET_PATH = 'train' + '/' + dataset_name
    test_dir = 'test' + '/' + dataset_name
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    NUM_EPOCHS = 5  # 15 good
    
    print('Load training data')
    dataStart = datetime.now()
    train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

    train_generator=train_datagen.flow_from_directory(DATASET_PATH, # this is where you specify the path to the main data folder
                                                     target_size=IMAGE_SIZE,
                                                     interpolation="nearest",
                                                     color_mode='rgb',
                                                     batch_size=BATCH_SIZE,
                                                     class_mode='categorical',
                                                     shuffle=True)
    
    print('Training Data Loading Time: ', datetime.now()-dataStart)
    
    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
    # Adam optimizer
    # loss function will be categorical cross entropy
    # evaluation metric will be accuracy
    
    print('Begin training...')
    trainStart = datetime.now()
    
    step_size_train=train_generator.n//train_generator.batch_size
    results = model.fit_generator(generator=train_generator,
                       steps_per_epoch=step_size_train,
                       epochs=NUM_EPOCHS)
    
    print('Training complete!')
    print('Training time: ', datetime.now()-trainStart)
    #f.write('train time: ' + (datetime.now()-trainStart)+'\n')
    
    model_path = './models/filtered-image-classification/'+model_name+uniqueID
    model.save(model_path)
    
    
    print('Loading test data')
    dataStart = datetime.now()
    
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

    test_generator = test_datagen.flow_from_directory(test_dir, # this is where you specify the path to the main data folder
                                                     target_size=IMAGE_SIZE,
                                                     interpolation="nearest",
                                                     color_mode='rgb',
                                                     class_mode='categorical',
                                                     shuffle=False,
                                                     batch_size=1)
    print('Test data load time: ', datetime.now()-dataStart)
    
    filenames = test_generator.filenames
    nb_samples = len(filenames)
    
    evalStart = datetime.now()
    print('Evaluating test data...')
    
    scores = model.evaluate_generator(test_generator, nb_samples)
    print('Score: ', scores)
    print('Eval time: ', datetime.now()-evalStart)
    #f.write('eval time: ' + (datetime.now()-evalStart)+'\n')
    
    predStart = datetime.now()
    print('Generating preds...')
    
    predict = model.predict_generator(test_generator,steps = nb_samples)
    print('Pred time: ', datetime.now()-predStart)
    #f.write('pred time: ' + (datetime.now()-predStart)+'\n')
    
    filenames = np.array(filenames)
    xy = np.concatenate((filenames.reshape(-1,1), predict),axis=1)
    pred_name = './preds/predictions_'+model_name+'_'+uniqueID+'.npy'
    np.save(pred_name, xy)
    
    #print('Generate loss plot')
    #plot_acc_loss(result, NUM_EPOCHS, uniqueID)
    
    print('Total script time: ', datetime.now()-scriptStart)
    print('Unique ID: ', uniqueID)
    #f.write('total time: ' + (datetime.now()-scriptStart)+'\n')
    
    
