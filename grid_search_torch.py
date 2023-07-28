import itertools
import pandas as pd
import pathlib, os
import mlcm

import gc
import matplotlib.pyplot as plt
import mlcm
import numpy as np
import torch
import tqdm
import utils_lightning
import utils_general
import utils_torch

from datetime import datetime
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callback
import torch
import utils_lightning
import utils_torch

gc.collect()
torch.cuda.empty_cache()

'''
This script runs several scenarios of training with multiples hiperparameters. 
'''

def save_results_to_csv(results, csv_file_path):
    """
    Save the results of a grid search iteration to a CSV file.

    Args:
        results (dict): Dictionary containing the results of the grid search iteration
    csv_file_path (str): Path to the CSV file where the results should be saved
    """
    if not pathlib.Path(csv_file_path).exists():
        header = ['model_name', 'index', 'batch_size', 'optimizer', 'learning_rate',\
                 'train_loss', 'train_accuracy', 'train_precision_macro_avg',\
                 'train_recall_macro_avg', 'train_f1_score_macro_avg', 'val_loss', 'val_accuracy',\
                 'val_precision_macro_avg','val_recall_macro_avg', 'val_f1_score_macro_avg']
        with open(csv_file_path, 'w') as f:
            f.write(','.join(header) + '\n')

    # Adding results at each iteration
    with open(csv_file_path, 'a') as f:
        f.write(','.join(str(x) for x in results.values()) + '\n')


def filter_combinations(file_path, combinations):
    """
    Filter out already executed combinations from the grid search.

    Args:
        file_path (str): Path to the CSV file containing the executed combinations
        combinations (list): List of all possible combinations for the grid search

    Returns:
        remaining_combinations (list): List of combinations that have not been executed yet
    """
    if os.path.exists(file_path):
        existing_results = pd.read_csv(file_path)
        # Convert CSV data into combinations format
        executed_combinations = set(zip(existing_results['batch_size'],
                                        existing_results['optimizer'],
                                        existing_results['learning_rate'],
                                        existing_results['model_name']))
    else:
        executed_combinations = set()
    remaining_combinations = [(i, comb) for i, comb in enumerate(combinations) if comb not in executed_combinations]
    return remaining_combinations


# Grid Search Parameters and CSV file path
parameters = {
            'batch_size': [16, 32, 64, 128, 256],\
            'optimizer': [torch.optim.SGD, torch.optim.RMSprop, torch.optim.Adam],\
            'learning_rate': [0.001, 0.01, 0.1],\
            'model_name': ['rajpurkar', 'ribeiro']
            }
csv_file_path = 'grid_search_results_torch.csv'

# Create an iterator with all parameters
combinations = itertools.product(parameters['batch_size'],\
                                 parameters['optimizer'],\
                                 parameters['learning_rate'],\
                                 parameters['model_name'])

# Filter the combinations to get the remaining ones
remaining_combinations = filter_combinations(csv_file_path, combinations)

# if not remaining_combinations:
#     remaining_combinations = combinations

for index, (batch_size, optimizer, learning_rate, model_name) in remaining_combinations:

    gc.collect()
    torch.cuda.empty_cache()

    # Showing current parameters
    print('######################################################################################')
    print(f'{index} | Modelo: {model_name} | Batch size: {batch_size} | Otimizador: {optimizer} | Learning rate: {learning_rate}')
    print('######################################################################################')

    datasets = utils_torch.creating_datasets()
    dataloaders = utils_torch.creating_dataloaders(datasets, batch_size)

    arguments = utils_torch.creating_the_kwargs(model_name, optimizer, learning_rate)
    model = utils_lightning.creating_the_model(arguments)

    # Paths
    model_name_path = f'{model_name}_{index}_{batch_size}_{optimizer}_{learning_rate}'
    model_path = f'results/{model_name_path}/'
    model_path_complete = f'results/{model_name_path}/model.ckpt'
    csv_path = f'results/{model_name_path}/history.csv'

    # Convert strings to Path type
    csv_path = pathlib.Path(csv_path)
    model_path = pathlib.Path(model_path)
    model_path_complete = pathlib.Path(model_path_complete)

    # Make sure the files are saved in a folder that exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path_complete.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = callback.ModelCheckpoint(dirpath=model_path, filename='model', 
                                                monitor='train_loss', save_top_k=1, mode='min')
    # Receiving information from the model
    rich_callback = callback.RichModelSummary(max_depth=3)

    # Early stopping callback
    early_stopping_callback = callback.EarlyStopping(monitor="val_loss", mode="min", patience=10)

    # Accumulating callbacks in a list()
    callbacks = [checkpoint_callback, rich_callback, early_stopping_callback]

    # Defining the trainer from pytorch lightning
    trainer = pl.Trainer(max_epochs=100, accelerator='gpu', callbacks=callbacks, fast_dev_run=False, devices='auto')

    # Fitting the model
    trainer.fit(model, train_dataloaders=dataloaders[0], val_dataloaders=dataloaders[1])
    
    loaded_model = utils_lightning.LitModel.load_from_checkpoint(model_path_complete)
    loaded_model.eval()
    loaded_model.freeze()
    
    prediction_bin_train = utils_torch.computate_predictions(loaded_model, datasets[0], 5000)

    # Get the report using the MLCM confusion matrix
    print('\n MLCM - Train')
    cm, _ = mlcm.cm(datasets[0].labels, prediction_bin_train, print_note=False)
    d_train = utils_general.get_mlcm_metrics(cm)
    del prediction_bin_train
    
    prediction_bin_val = utils_torch.computate_predictions(loaded_model, datasets[1], 5000)

    # Get the report using the MLCM confusion matrix
    print('\n MLCM - Validation')
    cm, _ = mlcm.cm(datasets[1].labels, prediction_bin_val, print_note=False)
    d_val = utils_general.get_mlcm_metrics(cm)
    del prediction_bin_val

    current_result = {
                    'model_name': model_name,
                    'index': index,
                    'batch_size': batch_size,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    # 'train_loss': train_metrics[0] ,
                    'train_loss': 'None' ,
                    # 'train_accuracy': train_metrics[1],
                    'train_accuracy': 'None',
                    'train_precision_macro_avg': d_train['macro_precision'],
                    'train_recall_macro_avg': d_train['macro_recall'],
                    'train_f1_score_macro_avg': d_train['macro_f1'],
                    # 'val_loss': val_metrics[0],
                    'val_loss': 'None',
                    # 'val_accuracy': val_metrics[1],
                    'val_accuracy': 'None',
                    'val_precision_macro_avg': d_val['macro_precision'],
                    'val_recall_macro_avg': d_val['macro_recall'],
                    'val_f1_score_macro_avg': d_val['macro_f1'],
                    }

    print('######################################################################################')
    print(f'Saving results to {csv_file_path}...')
    save_results_to_csv(current_result, csv_file_path)
    print('######################################################################################\n')

# Sorting the results
# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# Sort the DataFrame by the index column
df_sorted = df.sort_values('index')

# Save the sorted DataFrame to a new CSV file
df_sorted.to_csv(csv_file_path, index=False)
