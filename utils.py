# Python packages
import keras
from keras.layers import Conv1D, MaxPooling1D, Dropout, BatchNormalization, Flatten, Dense, ReLU, Add
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import plot_utils as putils
import pandas as pd
import seaborn as sns
import sklearn.metrics as skmetrics

# Rajpurkar model functions
# Create residual blocks
def residual_blocks_rajpurkar(input: keras.engine.keras_tensor.KerasTensor, i: int=0, stride: int=1, 
                              num_filter: int=64, rate_drop: float=0.5, 
                              initializer: str='none') -> keras.engine.keras_tensor.KerasTensor:
    
    layer = keras.Sequential([BatchNormalization(),
                              ReLU(),
                              Dropout(rate_drop),
                              Conv1D(kernel_size=16, filters=num_filter, strides=1, padding="same", kernel_initializer=initializer),
                              BatchNormalization(),
                              ReLU(),
                              Dropout(rate_drop),
                              Conv1D(kernel_size=16, filters=num_filter, strides=stride, padding="same", kernel_initializer=initializer)]
                              )(input)

    # layer = BatchNormalization()(input)
    # layer = ReLU()(layer)
    # layer = Dropout(rate_drop)(layer)
    # layer = Conv1D(kernel_size=16, filters=num_filter, strides=1, padding="same", kernel_initializer=initializer)(layer)
    # layer = BatchNormalization()(layer)
    # layer = ReLU()(layer)
    # layer = Dropout(rate_drop)(layer)
    # layer = Conv1D(kernel_size=16, filters=num_filter, strides=stride, padding="same", kernel_initializer=initializer)(layer)

    #Short connection
    if i == 3 or i == 7 or i == 11:
        layer_aj = Conv1D(kernel_size=16, filters=num_filter, strides=1, padding="same")(input)
        skip = MaxPooling1D(pool_size = 1, strides=2)(layer_aj)
    else:
        skip = MaxPooling1D(pool_size = 1, strides=stride)(input)

    # Adding layers
    return Add()([layer, skip])

# # Ribeiro's model functions
# # Create the skip connection A with MaxPooling and Conv layers
# '''argumento initializer não utilizado'''
# def skip_connection(skip: keras.engine.keras_tensor.KerasTensor, num_filter=128, rate_drop=0, 
#                     initializer='none', downsample=1) -> keras.engine.keras_tensor.KerasTensor:
    
#     skip = MaxPooling1D(pool_size=downsample,strides=downsample,padding='same')(skip)
#     skip = Conv1D(filters=num_filter,kernel_size=1,strides=1,padding='same')(skip)
#     return skip

def skip_connection(skip: keras.engine.keras_tensor.KerasTensor, num_filter: int=128, 
                    downsample: int=1) -> keras.engine.keras_tensor.KerasTensor:
    
    skip = keras.Sequential([MaxPooling1D(pool_size=downsample,strides=downsample,padding='same'), 
                            Conv1D(filters=num_filter,kernel_size=1,strides=1,padding='same')]
                            )(skip)
    return skip

# Create the residual blocks
def residual_blocks_ribeiro(input: keras.engine.keras_tensor.KerasTensor, num_filter: int=128, 
                            rate_drop: float=0, initializer: str='none', downsample: int=1) -> tuple:

    layer, skip = input

    skip = skip_connection(skip, num_filter=num_filter, downsample=downsample)

    layer = keras.Sequential([Conv1D(kernel_size=16, filters=num_filter, strides=1, padding="same", kernel_initializer=initializer),
                              BatchNormalization(),
                              ReLU(),
                              Dropout(rate_drop),
                              Conv1D(kernel_size=16, filters=num_filter, strides=downsample, padding="same", kernel_initializer=initializer)]
                              )(layer)

    # layer = Conv1D(kernel_size=16, filters=num_filter, strides=1, padding="same", kernel_initializer=initializer)(layer) 
    # layer = BatchNormalization()(layer)
    # layer = ReLU()(layer)
    # layer = Dropout(rate_drop)(layer)
    # layer = Conv1D(kernel_size=16, filters=num_filter, strides=downsample, padding="same", kernel_initializer=initializer)(layer) 

    layer = Add()([layer, skip])
    skip = layer

    layer = keras.Sequential([BatchNormalization(),
                              ReLU(),
                              Dropout(rate_drop)]
                              )(layer)

    # layer = BatchNormalization()(layer)
    # layer = ReLU()(layer)
    # layer = Dropout(rate_drop)(layer)

    return layer, skip

# Get the models of the network
def get_model(input_layer: keras.engine.keras_tensor.KerasTensor, 
              model_name: str) -> keras.engine.functional.Functional:

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
        
        # layers = Conv1D(strides=1, **conv_config)(input_layer)
        # layers = BatchNormalization()(layers)
        # layers = ReLU()(layers)

        # Short connection
        skip = MaxPooling1D(pool_size=1, strides=2)(layers)

        # Second block
        layers = keras.Sequential([Conv1D(strides=1, **conv_config),
                                   BatchNormalization(),
                                   ReLU(),
                                   Dropout(rate_drop),
                                   Conv1D(strides=2, **conv_config)]
                                   )(layers)
        
        # layers = Conv1D(strides=1, **conv_config)(layers)
        # layers = BatchNormalization()(layers)
        # layers = ReLU()(layers)
        # layers = Dropout(rate_drop)(layers)
        # layers = Conv1D(strides=2, **conv_config)(layers)

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
        
        # layers = BatchNormalization()(layers)
        # layers = ReLU()(layers)
        # layers = Flatten()(layers)
        # layers = Dense(32)(layers)
        
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
        
        # layers = Conv1D(
        #     kernel_size=16, filters=64, strides=1, padding="same", kernel_initializer=initializer
        # )(input_layer)  # Output_size = (1000, 64)
        # layers = BatchNormalization()(layers)
        # layers = ReLU()(layers)

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
        ''' FAZER ISSO USANDO RAISE ERROR -> retornar dois tipos diferentes na função me desagrada'''
        print('Wrong name')
        return

    # Constructing the model
    model = Model(inputs=input_layer, outputs=classification)

    return model


# Get the metrics
def get_metrics(y_test: np.ndarray, prediction: np.ndarray, prediction_bin: np.ndarray, 
                target_names: list, model_name: str, cm: np.ndarray) -> None:

    # Path
    csv_report = f'results/{model_name}/report.csv'
    csv_path_auc = f'results/{model_name}/roc_auc.csv'
    # csv_report_mlcm = f'results/{model_name}/report_mlcm.csv'
    
    # Convert strings to Path type
    csv_report = pathlib.Path(csv_report)
    csv_path_auc = pathlib.Path(csv_path_auc)
    # csv_report_mlcm = pathlib.Path(csv_report_mlcm)

    # Make sure the files are saved in a folder that exists
    csv_report.parent.mkdir(parents=True, exist_ok=True)
    csv_path_auc.parent.mkdir(parents=True, exist_ok=True)
    # csv_report_mlcm.parent.mkdir(parents=True, exist_ok=True)

    # Get the reports
    report = skmetrics.classification_report(y_test,prediction_bin,output_dict=True,target_names=target_names, zero_division=1)
    # report_mlcm = mlcm.stats(cm, False)

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

    return

# Plot results
def plot_results(history, name: str, metric: str, plot_path='plots') -> None:

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

''' Função não utilizada em lugar algum
# Transform the values in the confusion matrix in percentage
def get_cm_percent(cm: np.ndarray, total):
    cm_perc = np.zeros_like(cm, dtype='float')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            for k in range(cm.shape[2]):
                cm_perc[i][j][k] = round((cm[i][j][k] / total) * 100, 2)
    return cm_perc
'''

# # Plot confusion matrix
# ''' TALVEZ TRANSFORMAR EM DUAS FUNÇÕES '''
# def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray, model_name: str, 
#                           target_names: list, plot_path='results') -> np.ndarray:

#     # Make sure the plot folder exists
#     plot_path = pathlib.Path(plot_path) / model_name
#     plot_path.mkdir(parents=True, exist_ok=True)

#     # Confusion matrix
#     cm, _ = mlcm.cm(y_test, y_pred, print_note=False)
#     target_names = np.array([*target_names, 'NoC'])

#     # Calculating the normalization of the confusion matrix
#     divide = cm.sum(axis=1, dtype='int64')
#     divide[divide == 0] = 1
#     cm_norm = 100 * cm / divide[:, None]

#     # Plot the confusion matrix
#     fig, _ = plot_cm(cm_norm, target_names)
#     name = f"{model_name.split('-')[0]}-cm"
#     tight_kws = {'rect' : (0, 0, 1.1, 1)}
#     putils.save_fig(fig, name, path=plot_path, figsize='square',
#                     tight_scale='both', usetex=False, tight_kws=tight_kws)

#     return cm

def plot_confusion_matrix(cm: np.ndarray, model_name: str, 
                          target_names: list, plot_path='results') -> np.ndarray:

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
    fig, _ = plot_cm(cm_norm, target_names)
    name = f"{model_name.split('-')[0]}-cm"
    tight_kws = {'rect' : (0, 0, 1.1, 1)}
    putils.save_fig(fig, name, path=plot_path, figsize='square',
                    tight_scale='both', usetex=False, tight_kws=tight_kws)

    return


def plot_cm(confusion_matrix: np.ndarray, class_names: list, fontsize=10, cmap='Blues') -> tuple:

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

    return fig, ax

# ########################################################################### #
# Developing a function to produce some statistics based on the MLCM  
# ########################################################################### #
def get_mlcm_metrics(conf_mat: np.ndarray) -> dict:
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

        print('\n       class#     precision        recall      f1-score\
        weight\n')
        sp = '        '
        sp2 = '  '
        total_weight = divide.sum()
        float_formatter = "{:.2f}".format
        for k in range(num_classes-1):
            print(sp2,sp,k,sp,float_formatter(precision[k]),sp, \
                  float_formatter(recall[k]), sp,float_formatter(f1_score[k]),\
                  sp,divide[k])
        k = num_classes-1
        print(sp,' NTL',sp,float_formatter(precision[k]),sp, \
              float_formatter(recall[k]), sp,float_formatter(f1_score[k]), \
              sp,divide[k])

        print('\n    micro avg',sp,float_formatter(micro_precision),sp, \
              float_formatter(micro_recall),sp,float_formatter(micro_f1),\
              sp,total_weight)
        print('    macro avg',sp,float_formatter(macro_precision),sp,
              float_formatter(macro_recall),sp,float_formatter(macro_f1),sp,\
              total_weight)
        print(' weighted avg',sp,float_formatter(weighted_precision),sp,\
              float_formatter(weighted_recall),sp, \
              float_formatter(weighted_f1),sp,total_weight)
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

        print('\n       class#     precision        recall      f1-score\
        weight\n')
        sp = '        '
        sp2 = '  '
        total_weight = divide.sum()
        float_formatter = "{:.2f}".format
        for k in range(num_classes):
            print(sp2,sp,k,sp,float_formatter(precision[k]),sp, \
                  float_formatter(recall[k]), sp,\
                  float_formatter(f1_score[k]),sp,divide[k])
        print(sp,' NoC',sp,'There is not any data with no true-label assigned!')

        print('\n    micro avg',sp,float_formatter(micro_precision),sp,\
              float_formatter(micro_recall),sp,float_formatter(micro_f1),\
              sp,total_weight)
        print('    macro avg',sp,float_formatter(macro_precision),sp,\
              float_formatter(macro_recall),sp,float_formatter(macro_f1),sp,\
              total_weight)
        print(' weighted avg',sp,float_formatter(weighted_precision),sp,\
              float_formatter(weighted_recall),sp,\
              float_formatter(weighted_f1),sp,total_weight)

    # construct a dict to store values
    d = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'precision': precision,\
        'recall': recall, 'f1_score': f1_score, 'divide': divide,\
        'micro_precision': micro_precision, 'macro_precision': macro_precision,\
        'weighted_precision': weighted_precision, 'micro_recall': micro_recall,\
        'macro_recall': macro_recall, 'weighted_recall': weighted_recall,\
        'micro_f1': micro_f1, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1}
    
    return d
