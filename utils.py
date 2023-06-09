# Python packages
import gc # Garbage collector
import keras # for the machine learning models
from keras import backend as K # To clear gpu memory
from keras.layers import Conv1D, MaxPooling1D, Dropout, BatchNormalization, Flatten, Dense, ReLU, Add, Input # Easier way to write some keras functions 
from keras.models import Model # Easier way to write some keras functions
import matplotlib.pyplot as plt # for plotting
import numpy as np # some fundamental operations
import pathlib # for the paths 
import plot_utils as putils # importing custom code
import pandas as pd # for .csv manipulation
import seaborn as sns # used in some plotting
import sklearn.metrics as skmetrics # Will be destroyed, killed, forgotten
import tensorflow as tf

# Rajpurkar model functions
# Create residual blocks
def residual_blocks_rajpurkar(input, i=0, stride=1, 
                              num_filter=64, rate_drop=0.5, 
                              initializer='none'):
    '''
    This function creates the residual block for the Rajpurkar architecture
    
    inputs:
        input: keras.engine.keras_tensor.KerasTensor;
        i: int;
        stride: int;
        num_filter: int;
        rate_drop: float;
        initializer: str -> kernel_initializer for the kernel_initializer in Conv1D;
        
    return:
        skip: keras.engine.keras_tensor.KerasTensor;
    '''
    
    layer = keras.Sequential([BatchNormalization(),
                              ReLU(),
                              Dropout(rate_drop),
                              Conv1D(kernel_size=16, filters=num_filter, strides=1, padding="same", kernel_initializer=initializer),
                              BatchNormalization(),
                              ReLU(),
                              Dropout(rate_drop),
                              Conv1D(kernel_size=16, filters=num_filter, strides=stride, padding="same", kernel_initializer=initializer)]
                              )(input)

    #Short connection
    if i == 3 or i == 7 or i == 11:
        skip = keras.Sequential([Conv1D(kernel_size=16, filters=num_filter, strides=1, padding="same"),
                                 MaxPooling1D(pool_size = 1, strides=2)])(input)
    else:
        skip = MaxPooling1D(pool_size = 1, strides=stride)(input)

    # Adding layers
    return Add()([layer, skip])


def skip_connection(skip, num_filter=128, downsample=1):
    
    '''
    inputs:
        skip: keras.engine.keras_tensor.KerasTensor;
        num_filter: int;
        downsample: int;
        
    return: keras.engine.keras_tensor.KerasTensor

    '''
    
    skip = keras.Sequential([MaxPooling1D(pool_size=downsample,strides=downsample,padding='same'), 
                            Conv1D(filters=num_filter,kernel_size=1,strides=1,padding='same')]
                            )(skip)
    return skip


# Create the residual blocks
def residual_blocks_ribeiro(input, num_filter=128, 
                            rate_drop=0, initializer='none', downsample=1):

    '''
    inputs:
        input: keras.engine.keras_tensor.KerasTensor;
        num_filter: int;
        rate_drop: float;
        initializer: str -> kernel_initializer for the kernel_initializer in Conv1D;
        downsample: int;
        
    return:
        layer, skip: a tuple of two keras.engine.keras_tensor.KerasTensor
    '''

    layer, skip = input

    skip = skip_connection(skip, num_filter=num_filter, downsample=downsample)

    layer = keras.Sequential([Conv1D(kernel_size=16, filters=num_filter, strides=1, padding="same", kernel_initializer=initializer),
                              BatchNormalization(),
                              ReLU(),
                              Dropout(rate_drop),
                              Conv1D(kernel_size=16, filters=num_filter, strides=downsample, padding="same", kernel_initializer=initializer)]
                              )(layer)

    layer = Add()([layer, skip])
    skip = layer

    layer = keras.Sequential([BatchNormalization(),
                              ReLU(),
                              Dropout(rate_drop)]
                              )(layer)

    return layer, skip


# Get the models of the network
def get_model(input_shape, model_name):

    '''
    inputs:
        input: keras.engine.keras_tensor.KerasTensor;
        model_name: str -> selecting between 'rajpurkar' and 'ribeiro' architecture;
        
    return:
        model: keras.engine.functional.Functional;
    '''
    input_layer = Input(shape=input_shape)

    if 'rajpurkar' in model_name:
        rate_drop = 1 - 0.8
        initializer = 'he_normal'

        conv_config = dict(
            kernel_size=16, filters=64,
            padding="same", kernel_initializer=initializer
        )

        # First block
        layers = keras.Sequential([Conv1D(strides=1, **conv_config),
                                   BatchNormalization(),
                                   ReLU()]
                                   )(input_layer)

        # Short connection
        skip = MaxPooling1D(pool_size=1, strides=2)(layers)

        # Second block
        layers = keras.Sequential([Conv1D(strides=1, **conv_config),
                                   BatchNormalization(),
                                   ReLU(),
                                   Dropout(rate_drop),
                                   Conv1D(strides=2, **conv_config)]
                                   )(layers)
        
        # Adding layers
        layers = Add()([layers, skip])

        num_filter = [
            64, 64, 64,
            128, 128, 128, 128,
            192, 192, 192, 192,
            256, 256, 256, 256
        ]
        # Residual blocks
        for i in range(15):
            layers = residual_blocks_rajpurkar(
                layers, i=i, stride=(i % 2)+1, num_filter=num_filter[i],
                rate_drop=rate_drop, initializer=initializer
            )

        # Output block
        layers = keras.Sequential([BatchNormalization(),
                                   ReLU(),
                                   Flatten(),                  
                                   Dense(32)]
                                   )(layers)
        
        classification = Dense(5, activation='sigmoid')(layers)
    
    elif 'ribeiro' in model_name:
        initializer = 'he_normal'
        rate_drop = 1 - 0.8
        downsample = 4

        # Input block
        layers = keras.Sequential([Conv1D(kernel_size=16, filters=64, strides=1,
                                          padding="same", kernel_initializer=initializer),
                                   BatchNormalization(),
                                   ReLU()]
                                   )(input_layer)
        
        num_filter = np.array([128, 192, 256, 320])

        layer = layers
        skip = layers

        # Residual Blocks
        for i in range(4):
            layer, skip = residual_blocks_ribeiro(
                [layer, skip], num_filter=num_filter[i], rate_drop=rate_drop, initializer=initializer, downsample=downsample
            )

        # Output block
        layer = Flatten()(layer)
        classification = Dense(5, activation='sigmoid',
                               kernel_initializer=initializer)(layer)
    else:
        raise NameError(" Wrong Name. Allowed names are 'rajpurkar' and 'ribeiro'. ")

    # Constructing the model
    model = Model(inputs=input_layer, outputs=classification)

    return model


#This function lost its purpose
# Get the metrics Linha 168 ta abaixo
def get_metrics_skmetrics(y_test: np.ndarray, prediction: np.ndarray, prediction_bin: np.ndarray, 
                target_names: list, model_name: str) -> None:

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
    roc_auc_macro = skmetrics.roc_auc_score(y_test, prediction, average='macro')
    roc_auc_micro = skmetrics.roc_auc_score(y_test, prediction, average='micro')
    roc_auc_weighted = skmetrics.roc_auc_score(y_test, prediction, average='weighted')
    roc_auc_none = skmetrics.roc_auc_score(y_test, prediction, average=None)

    # Save the AUC metrics
    auc_dict = {
        'roc auc macro' : roc_auc_macro, 'roc auc micro' : roc_auc_micro,
        'roc auc weighted' : roc_auc_weighted, 'roc auc none' : roc_auc_none
    }
    pd.DataFrame.from_dict(data=auc_dict, orient='index').to_csv(csv_path_auc, header=False)

    return # Essa é a linha 202


# Plot results
def plot_results(history, name, metric, plot_path='plots'):

    '''
    inputs:
        history; name: str;
        metric: str;
        plot_path:str;
        
    return:
        This function returns nothing
    '''

    # Make sure the plot folder exists
    plot_path = pathlib.Path(plot_path)
    plot_path.mkdir(parents=True, exist_ok=True)

    # Plot results
    fig, ax = plt.subplots()
    ax.plot(history.epoch, history.history[metric], '-o')
    ax.plot(history.epoch, history.history[f'val_{metric}'], '-*')

    ax.set_xlabel('Epochs')
    ax.set_ylabel(f'{metric}'.capitalize())
    ax.legend(['Training set', 'Validation set'])

    # Save figure in multiple formats
    filename = plot_path / f'[{name}][{metric}]'
    fig.savefig(f'{filename}.png', format='png', dpi=600)
    fig.savefig(f'{filename}.pdf', format='pdf')

    return


# This function plots the normalized confusion matrix from mlcm
def plot_confusion_matrix(cm, model_name, target_names, plot_path='results'):

    '''
    inputs:
        cm: np.ndarray; 
        model_name: str; 
        target_names: list;
        plot_path: str;
        
    return:
        This function returns nothing
    '''

    # Make sure the plot folder exists
    plot_path = pathlib.Path(plot_path) / model_name
    plot_path.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    target_names = np.array([*target_names, 'NoC'])

    # Calculating the normalization of the confusion matrix
    divide = cm.sum(axis=1, dtype='int64')
    divide[divide == 0] = 1
    cm_norm = 100 * cm / divide[:, None]

    # Plot the confusion matrix
    fig = plot_cm(cm_norm, target_names)
    name = f"{model_name.split('-')[0]}-cm"
    tight_kws = {'rect' : (0, 0, 1.1, 1)}
    putils.save_fig(fig, name, path=plot_path, figsize='square',
                    tight_scale='both', usetex=False, tight_kws=tight_kws)

    return


def plot_cm(confusion_matrix, class_names, fontsize=10, cmap='Blues'):

    '''
    inputs:
        confusion_matrix: np.ndarray; 
        class_names: list;
        fontsize:int; 
        cmap: str;
        
    return:
        fig: Figure;
    '''

    # Plot the confusion matrix
    fig, ax = plt.subplots()

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    sns.heatmap(df_cm, annot=True, square=True, fmt='.1f', cbar=False, annot_kws={"size": fontsize},
        cmap=cmap, xticklabels=class_names, yticklabels=class_names, ax=ax)
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

    ax.set_xlabel('Rótulo predito')
    ax.set_ylabel('Rótulo verdadeiro')
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
    tp = np.zeros(num_classes, dtype=np.int64)  
    tn = np.zeros(num_classes, dtype=np.int64)  
    fp = np.zeros(num_classes, dtype=np.int64)  
    fn = np.zeros(num_classes, dtype=np.int64)  

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
    epsilon = 1e-7 # A small value to prevent division by zero
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1_score = 2 * tp / (2 * tp + fn + fp + 2 * epsilon)

    divide = conf_mat.sum(axis=1, dtype='int64') # sum of each row of MLCM

    if divide[-1] != 0: # some instances have not been assigned with any label 
        micro_precision = tp.sum()/(tp.sum()+fp.sum())
        macro_precision = precision.sum()/num_classes
        weighted_precision = (precision*divide).sum()/divide.sum()

        micro_recall = tp.sum()/(tp.sum()+fn.sum())
        macro_recall = recall.sum()/num_classes
        weighted_recall = (recall*divide).sum()/divide.sum()

        micro_f1 = (2*tp.sum())/(2*tp.sum()+fn.sum()+fp.sum())
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
    d = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'precision': precision,\
        'recall': recall, 'f1_score': f1_score, 'divide': divide,\
        'micro_precision': micro_precision, 'macro_precision': macro_precision,\
        'weighted_precision': weighted_precision, 'micro_recall': micro_recall,\
        'macro_recall': macro_recall, 'weighted_recall': weighted_recall,\
        'micro_f1': micro_f1, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1}
    
    return d


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

def load_data(test=False):
    """
    Load data from the 'data.npz' file.

    inputs:
        test: boolean;

    Returns:
        X_train, y_train, X_val, y_val: list of train and validation data if test=False;
        X_train, y_train, X_val, y_val, X_test, y_test: list of train, validation and test data if test=True;
        
        The values in the returned list are numpy.ndarray type
    """
    info = []
    with np.load('data.npz') as data:
        info.append(data['X_train'])
        info.append(data['y_train'])
        info.append(data['X_val'])
        info.append(data['y_val'])
        if test:
            info.append(data['X_test'])
            info.append(data['y_test'])
            
    return info

def get_mlcm_report(conf_mat, target_names, model_name):
    '''
    This function is a modified version of the 'stats' function presented in the mlcl paper.
    
    inputs:
        conf_mat: numpy.ndarray -> the 'conf_mat' returned from the 'cm' function of the 'mlcm' paper;
        target_names: list;
        model_name: str;
    '''
    
    num_classes = conf_mat.shape[1]
    tp = np.zeros(num_classes, dtype=np.int64)  
    tn = np.zeros(num_classes, dtype=np.int64)  
    fp = np.zeros(num_classes, dtype=np.int64)  
    fn = np.zeros(num_classes, dtype=np.int64)  

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
    epsilon = 1e-7 # A small value to prevent division by zero
    precision = tp/(tp+fp+epsilon)
    recall = tp/(tp+fn+epsilon)
    f1_score = 2*tp/(2*tp+fn+fp+epsilon)

    divide = conf_mat.sum(axis=1, dtype='int64') # sum of each row of MLCM

    d = {}

    if divide[-1] != 0: # some instances have not been assigned with any label 
        micro_precision = tp.sum()/(tp.sum()+fp.sum())
        macro_precision = precision.sum()/num_classes
        weighted_precision = (precision*divide).sum()/divide.sum()

        micro_recall = tp.sum()/(tp.sum()+fn.sum())
        macro_recall = recall.sum()/num_classes
        weighted_recall = (recall*divide).sum()/divide.sum()

        micro_f1 = (2*tp.sum())/(2*tp.sum()+fn.sum()+fp.sum())
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

        micro_precision = tp.sum()/(tp.sum()+fp.sum())
        macro_precision = precision.sum()/num_classes
        weighted_precision = (precision*divide).sum()/divide.sum()

        micro_recall = tp.sum()/(tp.sum()+fn.sum())
        macro_recall = recall.sum()/num_classes
        weighted_recall = (recall*divide).sum()/divide.sum()

        micro_f1 = (2*tp.sum())/(2*tp.sum()+fn.sum()+fp.sum())
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
    csv_report = f'results/{model_name}/report.csv'
    csv_path_auc = f'results/{model_name}/roc_auc.csv'
    
    # Convert strings to Path type
    csv_report = pathlib.Path(csv_report)
    csv_path_auc = pathlib.Path(csv_path_auc)

    # Make sure the files are saved in a folder that exists
    csv_report.parent.mkdir(parents=True, exist_ok=True)
    csv_path_auc.parent.mkdir(parents=True, exist_ok=True)
        
    pd.DataFrame.from_dict(d, orient='index').to_csv(csv_report)

    return