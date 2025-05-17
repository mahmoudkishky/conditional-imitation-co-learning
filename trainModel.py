import numpy as np
import tensorflow as tf
from cilModel import *
from dataGenerator import DataGenerator
from pandas import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import argparse
import os
import configs
import trainingConfigs
import json
import matplotlib.pyplot as plt
import pandas as pd
import random
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import TensorBoard
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def set_seeds(seed):

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()



def loadDataset(balance, predefinedIds):

    data = {key: None for key in trainingConfigs.BRANCHES}
    dataIds = {key: None for key in trainingConfigs.BRANCHES}
    trainIds = {key: None for key in trainingConfigs.BRANCHES}
    valIds = {key: None for key in trainingConfigs.BRANCHES}

    q_low = -0.8
    q_hi = 0.8
    
    for branch in trainingConfigs.BRANCHES:

        data[branch] = read_csv(configs.BRANCHES_DATASET_PATH + branch + '/control.csv')

        dropped = data[branch][(data[branch]['steering_angle'] < q_low) | (data[branch]['steering_angle'] > q_hi)].sample(frac = 1.0, replace=False)
        data[branch] = data[branch].drop(list(dropped.index), axis = 0)

        dropped = data[branch][data[branch]['speed'] > trainingConfigs.MAX_FORWARD_SPEED].sample(frac = 1.0, replace=False)
        data[branch] = data[branch].drop(list(dropped.index), axis = 0) 
        
        if balance:
            # remove majority samples to ensure we have balanced learning
            dropped = data[branch][np.abs(data[branch]['steering_angle']) <= 0.05].sample(frac = 0.80, replace=False)
            data[branch] = data[branch].drop(list(dropped.index), axis = 0)

        """  
        if predefinedIds == False:

            print("Split %s branch dataset to train-validation sets" %branch)
            dataIds[branch] = np.array(data[branch]['step'])
            np.random.shuffle(dataIds[branch])
        """
        print("Split %s branch dataset to train-validation sets" %branch)
        dataIds[branch] = np.array(data[branch]['step'])
        np.random.shuffle(dataIds[branch])
        
    data[branch].set_index('step', inplace=True)

    # dataset is split once and saved for later, for accurate model comparison
    
    """
    if predefinedIds:

        print("Use predefined train-validation sets")
        trainIds = np.load('trainIds.npy', allow_pickle = 'TRUE').item()
        valIds = np.load('valIds.npy', allow_pickle = 'TRUE').item()

    else:
        
        for branch in trainingConfigs.BRANCHES:
            trainIds[branch], valIds[branch] = train_test_split(dataIds[branch], test_size = 0.25)

        np.save('trainIds.npy', trainIds)
        np.save('valIds.npy', valIds)

    """
    for branch in trainingConfigs.BRANCHES:
        trainIds[branch], valIds[branch] = train_test_split(dataIds[branch], test_size = 0.25)

    np.save('trainIds.npy', trainIds)
    np.save('valIds.npy', valIds)
    

    return data, trainIds, valIds
   

def lr_schedule(epoch, lr):
    if epoch > 0 and epoch % 10 == 0:
        return lr * 0.5
    return lr



def main():

    data, trainIds, valIds = loadDataset(args.balance, (os.path.exists('trainIds.npy') and os.path.exists('valIds.npy')))
    
    # Generators
    training_generator = DataGenerator( data= data,
                                        has_3cam = args.has_3cam,
                                        dataIds= trainIds,
                                        batch_size= trainingConfigs.BATCH_SIZE,
                                        shuffle= True,
                                        isAugmentation= True)
    
    print(training_generator)
    validation_generator = DataGenerator( data= data,
                                          has_3cam = args.has_3cam,
                                          dataIds= valIds,
                                          batch_size= trainingConfigs.BATCH_SIZE,
                                          shuffle= True,
                                          isAugmentation= False)
    
    if args.has_3cam:
        model = three_cam_model(args.cic)
    else:
        model = single_cam_model(args.cic)

    model.summary()


    with open("./outputs/cil_model.json", 'w') as outfile:
        json.dump(model.to_json(), outfile)

    if args.loadweights:
        print('loading weights ....')
        model.load_weights("./outputs/cil_model.keras",True)

    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    #tensorboard_callback = TensorBoard(log_dir='./logs')

    callbacks = [ 
                    tf.keras.callbacks.ModelCheckpoint("./outputs/cp-{epoch:04d}.keras", verbose = 1, save_weights_only = True, save_freq = 'epoch'),
                    tf.keras.callbacks.EarlyStopping(monitor = "val_loss", min_delta = 0.0001, patience = 5),
                    lr_scheduler,
                    #tensorboard_callback
                ]
    
    # Train model on dataset
    model.fit(training_generator, epochs = trainingConfigs.EPOCHS, validation_data = validation_generator, callbacks = [callbacks])
    model.save_weights("./outputs/cil_model.keras", True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Steering angle model trainer')
    parser.add_argument('--loadweights', dest='loadweights', action='store_true',default=False, help='Start from checkpoint')
    parser.add_argument('--balance', dest='balance', action='store_true', default=False, help='is balanced dataset')
    parser.add_argument('--has_3cam', dest='has_3cam', action='store_true', default=False, help='use three cam model: front, left and right')
    parser.add_argument('--cic', dest='cic', action='store_true', default=False, help='use conditional imitation co-learnig model')

    args = parser.parse_args()
    set_seeds(seed= 42)
    main()
