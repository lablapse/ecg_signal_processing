import os
os.environ["CUDA_VISIBLE_DEVICES"]="1" 

import graphviz
import keras
import matplotlib.pyplot as plt # for plotting
import numpy as np # some fundamental operations
import pathlib # for the paths 
import plot_utils as putils # importing custom code
import pandas as pd # for .csv manipulation
import pickle
import seaborn as sns # used in some plotting
import torch
import torchview #https://github.com/mert-kurttutan/torchview
import tqdm
import utils_general
import utils_torch
import utils_lightning


def saving_weights_keras_model(keras_model, desired_layers):
    '''
    This function receives a keras model and a string to the path of the file with the generated weights.
    This only modify convolutional
    '''
    
    for layer_name in desired_layers:
        weights = []
        for layer in keras_model.layers:
            if type(layer) == desired_layers[layer_name] and layer.weights:
                intermediate = []
                for weight in layer.weights:
                    intermediate.append(weight.numpy())
                weights.append(intermediate)
            
        with open(f'saved_weights_keras_{layer_name}', 'wb') as fb:
            pickle.dump(weights, fb)    


def retrieving_model_name(dir):
    
    for subdir, dirs, files in os.walk(dir):
        if subdir == dir:
            continue

    if 'ribeiro' in subdir:
        return 'ribeiro'
    elif 'rajpurkar' in subdir:
        return 'rajpurkar'
    else:
        raise ValueError('Folder with improper name.')


def creating_the_torch_model(model_name, optimizer, learning_rate):
    optimizer_dict = {'Adam': torch.optim.Adam, 'SGD': torch.optim.SGD, 'RMSprop': torch.optim.RMSprop}
    
    # Creating the model
    arguments = utils_torch.creating_the_kwargs(model_name, optimizer=optimizer_dict[optimizer], learning_rate=learning_rate)
    model = utils_lightning.creating_the_model(arguments)
    return model


def deleting_weights(desired_layers):
    for layer in desired_layers:
        try:
            os.remove(f'saved_weights_keras_{layer}')
        except Exception as error:
            print(f'{error}')


dir ='./results_grid_search_keras'

desired_layers = {'conv1d': keras.layers.Conv1D, 'batch_norm': keras.layers.BatchNormalization, 'dense': keras.layers.Dense}

if True:
    for subdir, dirs, files in os.walk(dir):
        if subdir == dir:
            continue

        model = keras.models.load_model(subdir + f'/model.h5')
        saving_weights_keras_model(model, desired_layers)

print(retrieving_model_name(dir))