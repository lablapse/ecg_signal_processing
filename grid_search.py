import itertools
from keras import backend as K # To clear gpu memory
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.layers import Input
from keras.models import load_model
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
import pandas as pd
import pathlib, os, gc # Garbage collector
import tensorflow as tf
import utils, mlcm


def reset_keras():
    """
    Reset Keras session and clear GPU memory.
    """
    session = K.get_session()
    if session:
        session.close()

    K.clear_session()
    tf.compat.v1.reset_default_graph()
    try:
        del model, loaded_model, history # this is from global space - change this as you need
    except:
        pass

    gc.collect() # if it's done something you should see a number being outputted

    # Create a new interactive session
    config = tf.compat.v1.ConfigProto()
    # Enable dynamic memory allocation
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = '0'
    session = tf.compat.v1.InteractiveSession(config=config)


def load_data():
    """
    Load data from the 'data.npz' file.

    Returns:
        X_train, y_train, X_val, y_val: numpy arrays of train and validation data
    """
    with np.load('data.npz') as data:
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
    return X_train, y_train, X_val, y_val


def create_model(model_name, input_shape, optimizer, learning_rate):
    """
    Create and compile a Keras model with the specified parameters.

    Args:
        model_name (str): Name of the model architecture to use
        input_shape (tuple): Shape of the input data
        optimizer (str): Optimizer to use
        learning_rate (float): Learning rate for the optimizer

    Returns:
        model: Compiled Keras model
    """
    model = utils.get_model(Input(shape=input_shape), model_name)

    # Choose the optimizer
    if optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    elif optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


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


# Load data
X_train, y_train, X_val, y_val = load_data()

# Grid Search Parameters and CSV file path
parameters = {
            'batch_size': [16, 32, 64, 128, 256],\
            'optimizer': ['sgd', 'rmsprop', 'adam'],\
            'learning_rate': [0.001, 0.01, 0.1],\
            'model_name': ['rajpurkar', 'ribeiro']
            }
csv_file_path = 'grid_search_results.csv'

# Create an iterator with all parameters
combinations = itertools.product(parameters['batch_size'],\
                                 parameters['optimizer'],\
                                 parameters['learning_rate'],\
                                 parameters['model_name'])

# Filter the combinations to get the remaining ones
remaining_combinations = filter_combinations(csv_file_path, combinations)

for index, (batch_size, optimizer, learning_rate, model_name) in remaining_combinations:

    # Clear GPU memory and reset Keras Session
    print('######################################################################################')
    print('Resetting Keras Session...')
    reset_keras()

    # Showing current parameters
    print('######################################################################################')
    print(f'{index} | Modelo: {model_name} | Batch size: {batch_size} | Otimizador: {optimizer} | Learning rate: {learning_rate}')
    print('######################################################################################')

    # Convert to tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).shuffle(len(X_val))

    # Prepare the data
    train_dataset = train_dataset.batch(batch_size).prefetch(1) # 1 batch is prepared while the other is being trained
    val_dataset = val_dataset.batch(batch_size).prefetch(1)     # 1 batch is prepared while the other is being trained

    # Creating and compiling the model
    model = create_model(model_name=model_name, input_shape=X_train.shape[1:],\
                         optimizer=optimizer, learning_rate=learning_rate)

    # Paths
    model_name_path = f'{model_name}_{index}_{batch_size}_{optimizer}_{learning_rate}'
    model_path = f'results/{model_name_path}/model.h5'
    csv_path = f'results/{model_name_path}/history.csv'

    # Convert strings to Path type
    csv_path = pathlib.Path(csv_path)
    model_path = pathlib.Path(model_path)

    # Make sure the files are saved in a folder that exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Callbacks parameters
    callbacks = [
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, mode='min', min_lr=1e-6),
                EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=15),
                ModelCheckpoint(model_path, monitor='val_loss', mode='auto', verbose=1, save_best_only=True),
                CSVLogger(csv_path, separator=",", append=True)
                ]

    # Train the model
    history = model.fit(train_dataset, epochs=100, validation_data=val_dataset, callbacks=callbacks)

    # Load the model
    loaded_model = load_model(model_path)

    # ().evaluate returns a vector with position 0: loss, the other positions are the metrics,
    # in this case, just accuracy.
    train_metrics = loaded_model.evaluate(train_dataset, verbose=0) # train_metrics = loaded_model.evaluate(X_train, y_train, verbose=0)
    val_metrics = loaded_model.evaluate(val_dataset, verbose=0)     # val_metrics = loaded_model.evaluate(X_val, y_val, verbose=0)    

    # Convert the prediction_trains to binary values
    y_pred_train = loaded_model.predict(X_train, verbose=0)         # y_pred_train = loaded_model.predict(train_dataset, verbose=0)
    y_pred_train = (y_pred_train > 0.5).astype(int)

    # Get the report using the MLCM confusion matrix
    print('\n MLCM - Train')
    cm, _ = mlcm.cm(y_train, y_pred_train, print_note=False)
    d_train = utils.get_mlcm_metrics(cm)

    # Convert the prediction_vals to binary values
    y_pred_val = loaded_model.predict(X_val, verbose=0)             # y_pred_val = loaded_model.predict(val_dataset, verbose=0)
    y_pred_val = (y_pred_val > 0.5).astype(int)

    # Get the report using the MLCM confusion matrix
    print('\n MLCM - Validation')
    cm, _ = mlcm.cm(y_val, y_pred_val, print_note=False)
    d_val = utils.get_mlcm_metrics(cm)

    current_result = {
                    'model_name': model_name,
                    'index': index,
                    'batch_size': batch_size,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'train_loss': train_metrics[0] ,
                    'train_accuracy': train_metrics[1],
                    'train_precision_macro_avg': d_train['macro_precision'],
                    'train_recall_macro_avg': d_train['macro_recall'],
                    'train_f1_score_macro_avg': d_train['macro_f1'],
                    'val_loss': val_metrics[0],
                    'val_accuracy': val_metrics[1],
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