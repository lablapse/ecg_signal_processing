import itertools
import pandas as pd
import pathlib
# import timeit
from datetime import datetime

import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.models import load_model

import utils, mlcm

# Set GPU to be used
utils.set_gpu(1)

# Load data
X_train, y_train, X_val, y_val, X_test, y_test = utils.load_data()

# Sequence of classes names
target_names = ['NORM', 'STTC', 'CD', 'MI', 'HYP']

# Grid search hyperparameters and csv file path
hyperparameters = {
            'batch_size': [16, 32, 64, 128, 256],\
            'optimizer': ['sgd', 'rmsprop', 'adam'],\
            'learning_rate': [0.001, 0.01, 0.1],\
            'model_name': ['rajpurkar', 'ribeiro']
            }
epochs = 100

csv_file_path = 'results/grid_search_results.csv'

# Create an iterator with all parameters
combinations = itertools.product(hyperparameters['batch_size'],\
                                 hyperparameters['optimizer'],\
                                 hyperparameters['learning_rate'],\
                                 hyperparameters['model_name'])

# Filter the combinations to get the remaining ones
remaining_combinations = utils.filter_combinations(csv_file_path, combinations)

for index, (batch_size, optimizer, learning_rate, model_name) in remaining_combinations:


    # Clear GPU memory and reset Keras Session
    print('-------------------------')
    print('Resetting Keras Session...')
    utils.reset_keras()

    # Showing current hyperparameters
    print('\n-------------------------')
    print(f'{index} | Modelo: {model_name} | Batch size: {batch_size} | Otimizador: {optimizer} | Learning rate: {learning_rate}')
    print('-------------------------\n')

    # Convrt to tf.data.Dataset and suffle the order of the examples
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).shuffle(len(X_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).shuffle(len(X_test))
    
    # Prepare the data
    train_dataset = train_dataset.batch(batch_size).prefetch(1)
    val_dataset = val_dataset.batch(batch_size).prefetch(1)  

    # Creating and compiling the model
    model = utils.create_model(model_name = model_name, input_shape=X_train.shape[1:],\
                               optimizer=optimizer, learning_rate=learning_rate)
    
    # Get and print the memory usage of the model
    gbytes, mbytes = utils.get_model_memory_usage(batch_size, model)
    print(f'Model: {model_name} - (GPU) Memory requirements: {gbytes} GB and {mbytes} MB')

    # Paths
    model_name_path = f'{model_name}_{index}_{batch_size}_{optimizer}_{learning_rate}'
    # model_path = f'results/{model_name_path}/model.h5'
    model_path = f'results/{model_name_path}/model.tf'
    csv_path = f'results/{model_name_path}/history.csv'

    # Convert strings to Path type
    # csv_path = pathlib.Path(csv_path)
    # model_path = pathlib.Path(model_path)

    # Make sure the files are saved in a folder that exists
    # csv_path.parent.mkdir(parents=True, exist_ok=True)
    # model_path.parent.mkdir(parents=True, exist_ok=True)

    # Callbacks parameters
    callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, mode='min', min_lr=1e-6),
                 EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=15),
                 ModelCheckpoint(model_path, monitor='val_loss', mode='auto', verbose=1, save_best_only=True, save_format='tf'),
                 CSVLogger(csv_path, separator=',', append=True)]
    
    # Train the model
    # tic = timeit.default_timer()
    tic = datetime.now().isoformat()
    # history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=callbacks) # epochs = 100
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, callbacks=callbacks)
    # toc = timeit.default_timer()
    toc = datetime.now().isoformat()

    # Load the model
    loaded_model = load_model(model_path)

    # ().evaluate returns a vector with position 0: loss, the other positions are the metrics,
    # in this case, just accuracy.
    # train_metrics = loaded_model.evaluate(train_dataset, verbose=0) # OR (X_train, y_train, verbose=0)
    # val_metrics = loaded_model.evaluate(val_dataset, verbose=0)
    # test_metrics = loaded_model.evaluate(test_dataset, verbose=0)
    train_metrics = loaded_model.evaluate(X_train, y_train, verbose=0)
    val_metrics = loaded_model.evaluate(X_val, y_val, verbose=0)
    test_metrics = loaded_model.evaluate(X_test, y_test, verbose=0)

    # Convert the predictions to binary values
    y_pred_train = loaded_model.predict(X_train, verbose=0)
    y_pred_train = (y_pred_train > 0.5).astype(int)

    y_pred_val = loaded_model.predict(X_val, verbose=0)
    y_pred_val = (y_pred_val > 0.5).astype(int)

    y_pred_test = loaded_model.predict(X_test, verbose=0)
    y_pred_test = (y_pred_test > 0.5).astype(int)

    # Get the report using the MLCM confusion matrix
    print('\nMLCM - Train')
    cm, _ = mlcm.cm(y_train, y_pred_train, print_note=False)
    utils.get_mlcm_report(cm, target_names, model_name_path, dataset='train')
    # _ = mlcm.stats(cm, print_binary_mat=False)
    d_train = utils.get_mlcm_metrics(cm)

    print('\n MLCM - Validation')
    cm, _ = mlcm.cm(y_val, y_pred_val, print_note=False)
    utils.get_mlcm_report(cm, target_names, model_name_path, dataset='val')
    # _ = mlcm.stats(cm, print_binary_mat=False)
    d_val = utils.get_mlcm_metrics(cm)
    # Plot the MLCM
    utils.plot_confusion_matrix(cm, model_name_path, target_names, plot_path = 'results', dataset = 'val')

    print('\n MLCM - Test')
    cm, _ = mlcm.cm(y_test, y_pred_test, print_note=False)
    # _ = mlcm.stats(cm, print_binary_mat=False)
    d_test = utils.get_mlcm_metrics(cm)
    # Save a csv file with the reports from the MLCM - test
    utils.get_mlcm_report(cm, target_names, model_name_path, dataset='test')
    # Plot the MLCM
    # !!!! OBS.: "dataset" estava como "val". Dessa forma, todas as CMs são de teste e não de validação !!!!
    utils.plot_confusion_matrix(cm, model_name_path, target_names, plot_path = 'results', dataset = 'test')

    # Plot the loss from the training e validation set
    utils.plot_results(history, model_name_path, metric='loss')

    current_result = {
                    'model_name': model_name,
                    'index': index,
                    'batch_size': batch_size,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'time_start': tic,
                    'time_end': toc,
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
                    'test_loss': test_metrics[0],
                    'test_accuracy': test_metrics[1],
                    'test_precision_macro_avg': d_test['macro_precision'],
                    'test_recall_macro_avg': d_test['macro_recall'],
                    'test_f1_score_macro_avg': d_test['macro_f1']
                    }

    print('\n-------------------------')
    print(f'Saving results to {csv_file_path}...')
    utils.save_results_to_csv(current_result, csv_file_path)
    print('\n-------------------------')

# Sorting the results 
# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# Sort the DataFrame by the index column
df_sorted = df.sort_values('index')

# Save the sorted DataFrame to a new CSV file
df_sorted.to_csv(csv_file_path, index=False)