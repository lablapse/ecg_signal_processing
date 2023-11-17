import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

import graphviz
import gc
import keras
import matplotlib.pyplot as plt # for plotting
import mlcm
import numpy as np # some fundamental operations
from numba import cuda
import pathlib # for the paths 
import plot_utils as putils # importing custom code
import pandas as pd # for .csv manipulation
import pickle
import seaborn as sns # used in some plotting
import tensorflow
import torch
import torchview #https://github.com/mert-kurttutan/torchview
import tqdm
import utils_general
import utils_lightning
import utils_torch

csv_file_path = 'torch_with_trained_keras_weights.csv'
gc.collect()
torch.cuda.empty_cache()

def saving_weights_keras_model(keras_model, desired_layers, name_model, batch_size):
    '''
    This function receives a keras model and a string to the path of the file with the generated weights.
    This only modify convolutional
    '''

    # Loading the data
    info = utils_general.load_data(False)
    datasets = list()
    datasets.append(utils_torch.CustomDataset(info[2], info[3]))

    learning_rate = float(keras_model.optimizer.learning_rate)
    optimizer = keras_model.optimizer._name

    torch_model = creating_the_torch_model(name_model, optimizer, learning_rate)

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

        types = [torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.BatchNorm1d, torch.nn.Dropout1d]
        desired_type = desired_layers[layer_name]
        generated_weights_path_file = f'saved_weights_keras_{layer_name}'

        torch_model = utils_general.setting_torch_weights(torch_model, types, desired_type, generated_weights_path_file)
        deleting_weights([layer_name])


    utils_general.new_reset_keras([keras_model])
    cuda.get_current_device().reset()

    # utils_general.another_reset_keras()

    torch_model.to('cuda')
    torch_model.eval()
    torch_model.freeze()
    
    # Computating the validation predictions
    prediction_bin_val = utils_torch.computate_predictions(torch_model, datasets[0], 5000)

    # Get the report using the MLCM confusion matrix
    print('\n MLCM - Validation')
    cm, _ = mlcm.cm(datasets[0].labels, prediction_bin_val, print_note=False)
    d_val = utils_general.get_mlcm_metrics(cm)
    del prediction_bin_val

    # This dictionary will be transformed into the .csv file
    current_result = {
                    'model_name': name_model,
                    'batch_size': batch_size,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    # 'learning_rate': str(learning_rate).split('\'')[1][-1],
                    'val_precision_macro_avg': d_val['macro_precision'],
                    'val_recall_macro_avg': d_val['macro_recall'],
                    'val_f1_score_macro_avg': d_val['macro_f1'],
                    }

    print('######################################################################################')
    print(f'Saving results to {csv_file_path}...')
    save_results_to_csv(current_result, csv_file_path)
    print('######################################################################################\n')

def save_results_to_csv(results, csv_file_path):
    """
    Save the results of a grid search iteration to a CSV file.

    Args:
        results (dict): Dictionary containing the results of the grid search iteration
    csv_file_path (str): Path to the CSV file where the results should be saved
    """
    if not pathlib.Path(csv_file_path).exists():
        header = ['model_name', 'batch_size', 'optimizer', 'learning_rate',\
                 'val_precision_macro_avg','val_recall_macro_avg', 'val_f1_score_macro_avg']
        with open(csv_file_path, 'w') as f:
            f.write(','.join(header) + '\n')

    # Adding results at each iteration
    with open(csv_file_path, 'a') as f:
        f.write(','.join(str(x) for x in results.values()) + '\n')

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

if True:

    dir ='./results_grid_search_keras'
    desired_layers = {'conv1d': keras.layers.Conv1D, 'batch_norm': keras.layers.BatchNormalization, 'dense': keras.layers.Dense}

    for subdir, dirs, files in os.walk(dir):
        if subdir == dir:
            continue
        
        print(f'########## {subdir = } ##########')

        model = keras.models.load_model(subdir + f'/model.h5')
        name_model = retrieving_model_name(subdir)
        print(name_model)

        if '_16_' in subdir[len(dir)+len(name_model)+4:]:
            batch_size = 16
        elif '_32_' in subdir[len(dir)+len(name_model)+4:]:
            batch_size = 32
        elif '_64_' in subdir[len(dir)+len(name_model)+4:]:
            batch_size = 64
        elif '_128_' in subdir[len(dir)+len(name_model)+4:]:
            batch_size = 128
        elif '_256_' in subdir[len(dir)+len(name_model)+4:]:
            batch_size = 256
    
        saving_weights_keras_model(model, desired_layers, name_model, batch_size)