# Python packages
import gc                           # Garbage collector
from keras import backend as K      # Used to clear GPU memory
from keras.layers import Add, BatchNormalization, Conv1D, Dense, Dropout,\
                         Flatten, Input, MaxPooling1D, ReLU # Keras layers
from keras.models import Model      # Easier way to write some keras functions
import tensorflow as tf             # Deep learning framework
import matplotlib.pyplot as plt     # for plotting
import numpy as np                  # some fundamental operations
import pathlib                      # for the paths
import plot_utils as putils         # importing custom code
import pandas as pd                 # for .csv manipulation
import seaborn as sns               # used in some plotting
import sklearn.metrics as skmetrics # Will be dismissed (in future versions)
import os                           # For the paths


def reset_keras():
    '''
    Reset Keras session and clear GPU memory.
    '''
    
    # Reset Keras session
    K.clear_session()
    
    # Explicitly delete model variables
    try:
        del model, loaded_model, history  # these are from global space - adapt as needed
    except NameError:
        pass
    
    # Garbage collection
    gc.collect()
    
    # Refresh TensorFlow memory management
    # tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

    return


def set_gpu(gpu_index):
    '''
    Set the GPU to be used by TensorFlow.
    
    :param gpu_index: Index of the GPU to be used (0-based index)
    '''
    # Ensure that the GPU order follows PCI_BUS_ID order
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID' 
    
    # List available GPUs
    physical_devices = tf.config.list_physical_devices('GPU')
    
    # Ensure that the GPU index is within the valid range
    if gpu_index < 0 or gpu_index >= len(physical_devices):
        print('Invalid GPU index.')
        return False
    
    try:
        # Set the visible GPU devices
        tf.config.set_visible_devices(physical_devices[gpu_index:gpu_index + 1], 'GPU')

        # Validate that only the selected GPU is set as a logical device
        assert len(tf.config.list_logical_devices('GPU')) == 1
        
        print(f'GPU {gpu_index} has been set as the visible device.')
        return True
    except Exception as e:
        print(f'An error occurred while setting the GPU: {e}')
        return False


def reset_keras_old():
    '''
    Reset Keras session and clear GPU memory.
    '''
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


def load_data(filename='data.npz'):
    '''
    Load data from the 'data.npz' file.

    inputs:
        filename: str
    returns:
        X_train, y_train, X_val, y_val, X_test, y_test: numpy.ndarray type

    '''

    with np.load(filename) as data:
        # Train data
        X_train = data['X_train']
        y_train = data['y_train']
        # Validation data
        X_val = data['X_val']
        y_val = data['y_val']
        # Test data
        X_test = data['X_test']
        y_test = data['y_test']

    return X_train, y_train, X_val, y_val, X_test, y_test
    

def filter_combinations(file_path, combinations):
    '''
    Filter out already executed combinations from the grid search.

    Args:
        file_path (str): Path to the CSV file containing the executed combinations
        combinations (list): List of all possible combinations for the grid search

    Returns:
        remaining_combinations (list): List of combinations that have not been executed yet
    '''
    if os.path.exists(file_path):
        existing_results = pd.read_csv(file_path)
        # Convert csv data into combinations format
        executed_combinations = set(zip(existing_results['batch_size'],
                                        existing_results['optimizer'],
                                        existing_results['learning_rate'],
                                        existing_results['model_name']))
    else:
        executed_combinations = set()
    remaining_combinations = [(i, comb) for i, comb in enumerate(combinations) if comb not in executed_combinations]
    return remaining_combinations


# Rajpurkar's model functions
def residual_blocks_rajpurkar(input, i = 0, strides = 1, num_filter = 64, dropout_rate = 0.5, initializer = 'none'):

    '''
    This function creates the residual block for the Rajpurkar architecture.

    inputs:
        input: keras.engine.keras_tensor.KerasTensor
        i: int
        stride: int
        num_filter: int
        dropout_rate: float
        initializer: str -> kernel_initializer for Conv1D

    return:
        skip: keras.engine.keras_tensor.KerasTensor
    '''
    
    # Convolutional layer configuration
    conv_config = {'kernel_size': 16,
                   'filters': num_filter,
                   'padding': 'same',
                   'kernel_initializer': initializer}

    # Define the main path layers
    layers = BatchNormalization()(input)
    layers = ReLU()(layers)
    layers = Dropout(dropout_rate)(layers)
    layers = Conv1D(strides=1, **conv_config)(layers)
    layers = BatchNormalization()(layers)
    layers = ReLU()(layers)
    layers = Dropout(dropout_rate)(layers)
    layers = Conv1D(strides=strides, **conv_config)(layers)

    # Short connection
    if i in [3, 7, 11]:
        skip = Conv1D(strides = 1, **conv_config)(input)
        skip = MaxPooling1D(pool_size = 1, strides=2)(skip)
    else:
        skip = MaxPooling1D(pool_size = 1, strides=strides)(input)

    # Adding layers
    return Add()([layers, skip])


# Ribeiro's model functions
def skip_connection_ribeiro(skip, num_filter = 128, downsample=1):

    '''
    Create the skip connection A with MaxPooling and Conv layers.

    inputs:
        skip: keras.engine.keras_tensor.KerasTensor;
        num_filter: int;
        downsample: int;

    return:
        skip: keras.engine.keras_tensor.KerasTensor

    '''

    skip = MaxPooling1D(pool_size = downsample, strides = downsample, padding = 'same')(skip)
    skip = Conv1D(filters = num_filter, kernel_size = 1, strides = 1, padding = 'same')(skip)

    return skip


def residual_blocks_ribeiro(input, num_filter = 128, dropout_rate = 0, initializer = 'none', downsample = 1):

    '''
    inputs:
        input: keras.engine.keras_tensor.KerasTensor;
        num_filter: int;
        dropout_rate: float;
        initializer: str -> kernel_initializer for Conv1D;
        downsample: int;

    return:
        layers, skip: a tuple of two keras.engine.keras_tensor.KerasTensor
    '''

    # Unpack into two separate tensors
    layers, skip_A = input

    # Short connection A
    skip_A = skip_connection_ribeiro(skip_A, num_filter = num_filter, downsample = downsample)

    # Convolutional layer configuration
    conv_config = {'kernel_size': 16,
                   'filters': num_filter,
                   'padding': 'same',
                   'kernel_initializer': initializer}
    
    # Define the first block of layers of the main path
    layers = Conv1D(strides = 1, **conv_config)(layers) 
    layers = BatchNormalization()(layers)
    layers = ReLU()(layers)
    layers = Dropout(dropout_rate)(layers)
    layers = Conv1D(strides = downsample, **conv_config)(layers) 

    # Adding layers
    layers = Add()([layers, skip_A])

    # Short connection B
    skip_B = layers

    # Define the second block of layers of the main path
    layers = BatchNormalization()(layers)
    layers = ReLU()(layers)
    layers = Dropout(dropout_rate)(layers)

    return layers, skip_B


# Get the models of the network
def get_model(input_shape, model_name):

    '''
    inputs:
        input: keras.engine.keras_tensor.KerasTensor;
        model_name: str -> selecting between 'rajpurkar' and 'ribeiro' architecture;

    return:
        model: keras.engine.functional.Functional;
    '''

    # Define the input layer
    input_layer = Input(shape = input_shape)

    if 'rajpurkar' in model_name:
        initializer = 'he_normal'
        dropout_rate = 1 - 0.8

        # Convolutional layer configuration
        conv_config = {'kernel_size': 16,
                       'filters': 64,
                       'padding': 'same',
                       'kernel_initializer': initializer}

        # First block
        layers = Conv1D(strides = 1, **conv_config)(input_layer)
        layers = BatchNormalization()(layers)
        layers = ReLU()(layers)

        # Short connection
        skip = MaxPooling1D(pool_size = 1, strides = 2)(layers)

        # Second block
        layers = Conv1D(strides = 1, **conv_config)(layers)
        layers = BatchNormalization()(layers)
        layers = ReLU()(layers)
        layers = Dropout(dropout_rate)(layers)
        layers = Conv1D(strides = 2, **conv_config)(layers)

        # Adding layers
        layers = Add()([layers, skip])

        # Residual blocks
        num_filter = [64, 64, 64,
                      128, 128, 128, 128,
                      192, 192, 192, 192,
                      256, 256, 256, 256]
        
        for i in range(15):
            layers = residual_blocks_rajpurkar(layers,
                                               i = i,
                                               strides = (i % 2)+1,
                                               num_filter = num_filter[i],
                                               dropout_rate = dropout_rate,
                                               initializer = initializer)

        # Output block
        layers = BatchNormalization()(layers)
        layers = ReLU()(layers)
        layers = Flatten()(layers)
        layers = Dense(32)(layers)
        # classification = Dense(5, activation='sigmoid')(layers)

    elif 'ribeiro' in model_name:
        initializer = 'he_normal'
        dropout_rate = 1 - 0.8
        downsample = 4

        # Convolutional layer configuration
        conv_config = {'kernel_size': 16,
                       'filters': 64,
                       'strides': 1,
                       'padding': 'same',
                       'kernel_initializer': initializer}

        # Input block
        layers = Conv1D(**conv_config)(input_layer)
        layers = BatchNormalization()(layers)
        layers = ReLU()(layers)
        
        # Residual Blocks
        layers, skip = layers, layers
        num_filter = [128, 192, 256, 320]
        for i in range(4):
            layers, skip = residual_blocks_ribeiro([layers, skip],
                                                   num_filter = num_filter[i],
                                                   dropout_rate = dropout_rate,
                                                   initializer = initializer,
                                                   downsample = downsample)

        # Output block
        layers = Flatten()(layers)
        # classification = Dense(5, activation='sigmoid', kernel_initializer=initializer)(layers)

    else:
        raise NameError('Wrong model name.')

    # Constructing the model
    classification = Dense(5, activation = 'sigmoid', kernel_initializer = initializer)(layers)
    model = Model(inputs = input_layer, outputs = classification, name = model_name)

    return model


def create_model(model_name, input_shape, optimizer, learning_rate):
    '''
    Create and compile a Keras model with the specified parameters.

    Args:
        model_name (str): Name of the model architecture to use
        input_shape (tuple): Shape of the input data
        optimizer (str): Optimizer to use
        learning_rate (float): Learning rate for the optimizer

    Returns:
        model: Compiled Keras model
    '''
    model = get_model(input_shape, model_name)

    # Choose the optimizer
    if optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate = learning_rate)
    elif optimizer == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate = learning_rate)
    elif optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# Plot results
def plot_results(history, model_name, metric, plot_path = 'results'):

    '''
    inputs:
        history: keras.callbacks.History
        name: str
        metric: str
        plot_path: str

    return:
        This function returns nothing
    '''

    # Make sure the plot folder exists
    plot_path = pathlib.Path(plot_path)
    plot_path.mkdir(parents = True, exist_ok = True)

    # Plot results
    fig, ax = plt.subplots()
    ax.plot(history.epoch, history.history[metric], '-o')
    ax.plot(history.epoch, history.history[f'val_{metric}'], '-*')
    ax.set_xlabel('Epochs')
    ax.set_ylabel(f'{metric}'.capitalize())
    ax.legend(['Training set', 'Validation set'])

    # Save figure in multiple formats
    name = f"{model_name.split('-')[0]}-{metric}"
    # filename = plot_path / name
    filename = f"{plot_path}/{model_name}/{name}"
    fig.savefig(f'{filename}.png', format='png', dpi=600)
    fig.savefig(f'{filename}.pdf', format='pdf')

    plt.close(fig)

    return


# This function plots the normalized confusion matrix from mlcm
def plot_confusion_matrix(cm, model_name, target_names, plot_path, dataset):

    '''
    inputs:
        cm: np.ndarray
        model_name: str
        target_names: list
        plot_path: str

    return:
        This function returns nothing
    '''

    # Make sure the plot folder exists
    plot_path = pathlib.Path(plot_path) / model_name
    plot_path.mkdir(parents = True, exist_ok = True)

    # Confusion matrix
    target_names = np.array([*target_names, 'NoC'])

    # Calculating the normalization of the confusion matrix
    divide = cm.sum(axis = 1, dtype = 'int64')
    divide[divide == 0] = 1
    cm_norm = 100 * cm / divide[:, None]

    # Plot the confusion matrix
    fig = plot_cm(cm_norm, target_names)
    name = f"{model_name.split('-')[0]}-{dataset}-cm"
    tight_kws = {'rect' : (0, 0, 1.1, 1)}
    putils.save_fig(fig, name, path = plot_path, figsize = 'square',
                    tight_scale = 'both', usetex = False, tight_kws = tight_kws)
    
    plt.close(fig)

    return


def plot_cm(confusion_matrix, class_names, fontsize = 10, cmap = 'Blues'):

    '''
    inputs:
        confusion_matrix: np.ndarray
        class_names: list
        fontsize: int
        cmap: str

    return:
        fig: Figure
    '''

    # Plot the confusion matrix
    fig, ax = plt.subplots()

    df_cm = pd.DataFrame(confusion_matrix, index = class_names, columns = class_names)

    sns.heatmap(df_cm, annot = True, square = True, fmt = '.1f', cbar = False,
                annot_kws = {"size": fontsize}, cmap = cmap, ax = ax,
                xticklabels = class_names, yticklabels = class_names)
    
    for t in ax.texts:
        t.set_text(t.get_text() + '%')

    xticks = ax.get_xticklabels()
    xticks[-1].set_text('NPL')
    ax.set_xticklabels(xticks)

    yticks = ax.get_yticklabels()
    yticks[-1].set_text('NTL')
    ax.set_yticklabels(yticks)

    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    # ax.set_xlabel('Rótulo predito')
    ax.set_xlabel('Predicted label')
    # ax.set_ylabel('Rótulo verdadeiro')
    ax.set_ylabel('True Label')
    fig.tight_layout()

    return fig


def get_mlcm_metrics(conf_mat):

    '''
    input:
        conf_mat: np_ndarray -> the 'conf_mat' returned in the 'cm' function from the 'mlcm' paper;

    return:
        d: dict;
    '''

    num_classes = conf_mat.shape[1]
    tp = np.zeros(num_classes, dtype = np.int64)
    tn = np.zeros(num_classes, dtype = np.int64)
    fp = np.zeros(num_classes, dtype = np.int64)
    fn = np.zeros(num_classes, dtype = np.int64)

    precision = np.zeros(num_classes, dtype = float)
    recall = np.zeros(num_classes, dtype = float)
    f1_score = np.zeros(num_classes, dtype = float)

    # Calculating TP, TN, FP, FN from MLCM
    for k in range(num_classes):
        tp[k] = conf_mat[k][k]
        for i in range(num_classes):
            if i != k:
                tn[k] += conf_mat[i][i]
                fp[k] += conf_mat[i][k]
                fn[k] += conf_mat[k][i]

    # Calculating precision, recall, and F1-score for each of classes
    epsilon = 1e-6 # A small value to prevent division by zero
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1_score = 2*tp / (2*tp + fn + fp + epsilon)

    divide = conf_mat.sum(axis=1, dtype='int64') # sum of each row of MLCM

    if divide[-1] != 0: # some instances have not been assigned with any label
        micro_precision = tp.sum()/(tp.sum() + fp.sum())
        macro_precision = precision.sum()/num_classes
        weighted_precision = (precision*divide).sum()/divide.sum()

        micro_recall = tp.sum()/(tp.sum() + fn.sum())
        macro_recall = recall.sum()/num_classes
        weighted_recall = (recall*divide).sum()/divide.sum()

        micro_f1 = (2*tp.sum())/(2*tp.sum() + fn.sum() + fp.sum())
        macro_f1 = f1_score.sum()/num_classes
        weighted_f1 = (f1_score*divide).sum()/divide.sum()

    else:
        precision = precision[:-1]
        recall = recall[:-1]
        f1_score = f1_score[:-1]
        divide = divide[:-1]
        num_classes -= 1

        micro_precision = tp.sum()/(tp.sum()+fp.sum())
        macro_precision = precision.sum()/num_classes
        weighted_precision = (precision*divide).sum()/divide.sum()

        micro_recall = tp.sum()/(tp.sum()+fn.sum())
        macro_recall = recall.sum()/num_classes
        weighted_recall = (recall*divide).sum()/divide.sum()

        micro_f1 = (2*tp.sum())/(2*tp.sum()+fn.sum()+fp.sum())
        macro_f1 = f1_score.sum()/num_classes
        weighted_f1 = (f1_score*divide).sum()/divide.sum()

    # construct a dict to store values
    d = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'precision': precision,
        'recall': recall, 'f1_score': f1_score, 'divide': divide,
        'micro_precision': micro_precision, 'macro_precision': macro_precision,
        'weighted_precision': weighted_precision, 'micro_recall': micro_recall,
        'macro_recall': macro_recall, 'weighted_recall': weighted_recall,
        'micro_f1': micro_f1, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1}

    return d


def get_mlcm_report(conf_mat, target_names, model_name, dataset):
    '''
    This function is a modified version of the 'stats' function presented in the mlcl paper.

    inputs:
        conf_mat: numpy.ndarray -> the 'conf_mat' returned from the 'cm' function of the 'mlcm' paper;
        target_names: list;
        model_name: str;
    '''

    num_classes = conf_mat.shape[1]
    tp = np.zeros(num_classes, dtype = np.int64)
    tn = np.zeros(num_classes, dtype = np.int64)
    fp = np.zeros(num_classes, dtype = np.int64)
    fn = np.zeros(num_classes, dtype = np.int64)

    precision = np.zeros(num_classes, dtype=float)
    recall = np.zeros(num_classes, dtype=float)
    f1_score = np.zeros(num_classes, dtype=float)

    # Calculating TP, TN, FP, FN from MLCM
    for k in range(num_classes):
        tp[k] = conf_mat[k][k]
        for i in range(num_classes):
            if i != k:
                tn[k] += conf_mat[i][i]
                fp[k] += conf_mat[i][k]
                fn[k] += conf_mat[k][i]

    # Calculating precision, recall, and F1-score for each of classes
    epsilon = 1e-6 # A small value to prevent division by zero
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1_score = 2 * tp / (2 * tp + fn + fp + epsilon)

    divide = conf_mat.sum(axis=1, dtype='int64') # sum of each row of MLCM

    d = {}

    if divide[-1] != 0: # some instances have not been assigned with any label
        micro_precision = tp.sum()/(tp.sum() + fp.sum())
        macro_precision = precision.sum()/num_classes
        weighted_precision = (precision*divide).sum()/divide.sum()

        micro_recall = tp.sum()/(tp.sum() + fn.sum())
        macro_recall = recall.sum()/num_classes
        weighted_recall = (recall*divide).sum()/divide.sum()

        micro_f1 = (2*tp.sum())/(2*tp.sum() + fn.sum() + fp.sum())
        macro_f1 = f1_score.sum()/num_classes
        weighted_f1 = (f1_score*divide).sum()/divide.sum()

        total_weight = divide.sum()

        for k in range(num_classes-1):
            d[f'{target_names[k]}'] = {'precision':precision[k], 'recall':recall[k], \
                                     'f1_score':f1_score[k], 'weight':divide[k]}
        k = num_classes-1
        d['NTL'] = {'precision':precision[k], 'recall':recall[k], \
                    'f1_score':f1_score[k], 'weight':divide[k]}

        d['micro avg'] = {'precision':micro_precision, 'recall':micro_recall, \
                          'f1_score':micro_f1, 'weight':total_weight}

        d['macro avg'] = {'precision':macro_precision, 'recall':macro_recall, \
                          'f1_score':macro_f1, 'weight':total_weight}

        d['weighted avg'] = {'precision':weighted_precision, 'recall':weighted_recall, \
                          'f1_score':weighted_f1, 'weight':total_weight}
    else:
        precision = precision[:-1]
        recall = recall[:-1]
        f1_score = f1_score[:-1]
        divide = divide[:-1]
        num_classes -= 1

        micro_precision = tp.sum()/(tp.sum() + fp.sum())
        macro_precision = precision.sum()/num_classes
        weighted_precision = (precision*divide).sum()/divide.sum()

        micro_recall = tp.sum()/(tp.sum()+fn.sum())
        macro_recall = recall.sum()/num_classes
        weighted_recall = (recall*divide).sum()/divide.sum()

        micro_f1 = (2*tp.sum())/(2*tp.sum() + fn.sum() + fp.sum())
        macro_f1 = f1_score.sum()/num_classes
        weighted_f1 = (f1_score*divide).sum()/divide.sum()

        total_weight = divide.sum()

        for k in range(num_classes):
            d[f'{target_names[k]}'] = {'precision':precision[k], 'recall':recall[k], \
                                     'f1_score':f1_score[k], 'weight':divide[k]}

        # print(sp,' NoC',sp,'There is not any data with no true-label assigned!')
        d[f'NoC'] = {'precision':'no data', 'recall':'no data', \
                     'f1_score':'no data', 'weight':'no data'}

        d['micro avg'] = {'precision':micro_precision, 'recall':micro_recall, \
                          'f1_score':micro_f1, 'weight':total_weight}

        d['macro avg'] = {'precision':macro_precision, 'recall':macro_recall, \
                          'f1_score':macro_f1, 'weight':total_weight}

        d['weighted avg'] = {'precision':weighted_precision, 'recall':weighted_recall, \
                          'f1_score':weighted_f1, 'weight':total_weight}

    # Path
    csv_report = f'results/{model_name}/report-{dataset}.csv'
    # csv_path_auc = f'results/{model_name}/roc_auc.csv'

    # Convert strings to Path type
    csv_report = pathlib.Path(csv_report)
    # csv_path_auc = pathlib.Path(csv_path_auc)

    # Make sure the files are saved in a folder that exists
    csv_report.parent.mkdir(parents=True, exist_ok=True)
    # csv_path_auc.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame.from_dict(d, orient='index').to_csv(csv_report)

    return


def save_results_to_csv(results, csv_file_path):
    '''
    Save the results of a grid search iteration to a CSV file.

    Args:
        results (dict): Dictionary containing the results of the grid search iteration
    csv_file_path (str): Path to the CSV file where the results should be saved
    '''
    if not pathlib.Path(csv_file_path).exists():
        header = ['model_name', 'index', 'batch_size', 'optimizer', 'learning_rate',\
                 'time_start', 'time_end', 'train_loss', 'train_accuracy', \
                 'train_precision_macro_avg', 'train_recall_macro_avg', \
                 'train_f1_score_macro_avg', 'val_loss', 'val_accuracy',\
                 'val_precision_macro_avg','val_recall_macro_avg', 'val_f1_score_macro_avg',\
                 'test_loss', 'test_accuracy', 'test_precision_macro_avg',\
                 'test_recall_macro_avg', 'test_f1_score_macro_avg']
        with open(csv_file_path, 'w') as f:
            f.write(','.join(header) + '\n')
    
    # Adding results at each iteration
    with open(csv_file_path, 'a') as f:
        f.write(','.join(str(x) for x in results.values()) + '\n')
    
    return

#This function lost its purpose
def get_metrics_skmetrics(y_test, prediction, prediction_bin, target_names, model_name):
    '''
    This function originally had a purpose, but it's now deprecated. 

    input:
        y_test (np.ndarray): The true labels for the test set.
        prediction (np.ndarray): The model's predicted probabilities for each label in the test set.
        prediction_bin (np.ndarray): The model's predicted labels for the test set.
        target_names (list): The list of class names.
        model_name (str): The name of the model used for prediction.

    returns:
        None
    '''
    
    # Path
    csv_report = f'results/{model_name}/report.csv'
    csv_path_auc = f'results/{model_name}/roc_auc.csv'

    # Convert strings to Path type
    csv_report = pathlib.Path(csv_report)
    csv_path_auc = pathlib.Path(csv_path_auc)

    # Make sure the files are saved in a folder that exists
    csv_report.parent.mkdir(parents=True, exist_ok=True)
    csv_path_auc.parent.mkdir(parents=True, exist_ok=True)

    # Get the reports
    report = skmetrics.classification_report(y_test, prediction_bin, output_dict=True, target_names=target_names, zero_division=1)

    # Save the reports
    pd.DataFrame.from_dict(report, orient='index').to_csv(csv_report)

    # ROC AUC metrics
    roc_auc_macro = skmetrics.roc_auc_score(y_test, prediction, average = 'macro')
    roc_auc_micro = skmetrics.roc_auc_score(y_test, prediction, average = 'micro')
    roc_auc_weighted = skmetrics.roc_auc_score(y_test, prediction, average = 'weighted')
    roc_auc_none = skmetrics.roc_auc_score(y_test, prediction, average = None)

    # Save the AUC metrics
    auc_dict = {
        'roc auc macro' : roc_auc_macro, 'roc auc micro' : roc_auc_micro,
        'roc auc weighted' : roc_auc_weighted, 'roc auc none' : roc_auc_none
    }
    pd.DataFrame.from_dict(data=auc_dict, orient='index').to_csv(csv_path_auc, header=False)

    return

def get_model_memory_usage(batch_size, model):
    """
    Calculate the memory usage of a Keras model.

    Args:
    batch_size (int): The batch size you are using or plan to use.
    model (keras.Model or tensorflow.keras.Model): The model to calculate memory usage for.

    Returns:
    tuple: A tuple containing memory usage in whole gigabytes and the remaining usage in megabytes.
    """

    import numpy as np
    # Try importing from tensorflow.keras backend first as it's more standard
    try:
        from keras import backend as K
    except ImportError:
        from keras import backend as K

    # Initialize counters for shapes memory and nested model memory
    shapes_mem_count = 0
    internal_model_mem_count = 0
    # Iterate over layers in the model
    for l in model.layers:
        layer_type = l.__class__.__name__
        # If layer is a model itself, recursively calculate its memory usage
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        else:
            # Calculate memory usage of the layer's output shape
            single_layer_mem = 1
            out_shape = l.output_shape
            # If output shape is a list, consider only the first shape
            if type(out_shape) is list:
                out_shape = out_shape[0]
            # Multiply all dimensions to get total number of units
            for s in out_shape:
                if s is not None:
                    single_layer_mem *= s
            shapes_mem_count += single_layer_mem

    # Calculate total number of trainable and non-trainable parameters
    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    # Define number size based on the float type used by Keras backend
    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    # Calculate total memory usage
    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    total_memory += internal_model_mem_count

    # Convert from bytes to GB
    gbytes = total_memory / (1024 ** 3)
    # Get the fraction part of GB
    remaining_gbytes = gbytes - int(gbytes)
    # Convert the remaining GB to MB
    mbytes = remaining_gbytes * 1024

    # Return both GB and MB
    return int(gbytes), round(mbytes, 2)