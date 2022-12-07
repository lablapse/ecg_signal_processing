# Python packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

import tensorflow as tf
from keras.layers import Conv1D, MaxPooling1D, Dropout, BatchNormalization, Flatten, Dense, ReLU, Add
from keras.models import Model

import sklearn.metrics as skmetrics
from mlcm import mlcm

import plot_utils as putils


physical_devices = tf.config.list_physical_devices('GPU')
try:
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    pass
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

# Rajpurkar model functions
# Create residual blocks
def residual_blocks_rajpurkar(input,i=0, stride=1, num_filter=64, rate_drop=0.5, initializer='none'):

    layer = BatchNormalization()(input)
    layer = ReLU()(layer)
    layer = Dropout(rate_drop)(layer)
    layer = Conv1D(kernel_size=16, filters=num_filter, strides=1, padding="same", kernel_initializer=initializer)(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = Dropout(rate_drop)(layer)
    layer = Conv1D(kernel_size=16, filters=num_filter, strides=stride, padding="same", kernel_initializer=initializer)(layer)

    #Short connection
    if i == 3 or i == 7 or i == 11:
        layer_aj = Conv1D(kernel_size=16, filters=num_filter, strides=1, padding="same")(input)
        skip = MaxPooling1D(pool_size = 1, strides=2)(layer_aj)
    else:
        skip = MaxPooling1D(pool_size = 1, strides=stride)(input)

    # Adding layers
    return Add()([layer, skip])

# Ribeiro's model functions
# Create the skip connection A with MaxPooling and Conv layers
def skip_connection(skip, num_filter=128, rate_drop=0, initializer='none', downsample=1):
    skip = MaxPooling1D(pool_size=downsample,strides=downsample,padding='same')(skip)
    skip = Conv1D(filters=num_filter,kernel_size=1,strides=1,padding='same')(skip)
    return skip

# Create the residual blocks
def residual_blocks_ribeiro(input, num_filter=128, rate_drop=0, initializer='none', downsample=1):

    layer, skip = input

    skip = skip_connection(skip, num_filter=num_filter, rate_drop=rate_drop, initializer=initializer, downsample=downsample)

    layer = Conv1D(kernel_size=16, filters=num_filter, strides=1, padding="same", kernel_initializer=initializer)(layer) 
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = Dropout(rate_drop)(layer)
    layer = Conv1D(kernel_size=16, filters=num_filter, strides=downsample, padding="same", kernel_initializer=initializer)(layer) 

    layer = Add()([layer, skip])
    skip = layer

    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = Dropout(rate_drop)(layer)

    return layer, skip

# Get the models os the network
def get_model(input_layer, model_name):

    if 'rajpurkar' in model_name:
        rate_drop = 1 - 0.8
        initializer = 'he_normal'

        conv_config = dict(
            kernel_size=16, filters=64,
            padding="same", kernel_initializer=initializer
        )

        # First block
        layers = Conv1D(strides=1, **conv_config)(input_layer)
        layers = BatchNormalization()(layers)
        layers = ReLU()(layers)

        # Short connection
        skip = MaxPooling1D(pool_size=1, strides=2)(layers)

        # Second block
        layers = Conv1D(strides=1, **conv_config)(layers)
        layers = BatchNormalization()(layers)
        layers = ReLU()(layers)
        layers = Dropout(rate_drop)(layers)
        layers = Conv1D(strides=2, **conv_config)(layers)

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
        layers = BatchNormalization()(layers)
        layers = ReLU()(layers)
        layers = Flatten()(layers)
        layers = Dense(32)(layers)
        classification = Dense(5, activation='sigmoid')(layers)
    
    elif 'ribeiro' in model_name:
        initializer = 'he_normal'
        rate_drop = 1 - 0.8
        downsample = 4

        # Input block
        layers = Conv1D(
            kernel_size=16, filters=64, strides=1, padding="same", kernel_initializer=initializer
        )(input_layer)  # Output_size = (1000, 64)
        layers = BatchNormalization()(layers)
        layers = ReLU()(layers)

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
        print('Wrong name')
        return

    # Constructing the model
    model = Model(inputs=input_layer, outputs=classification)

    return model


# Get the metrics
def get_metrics(y_test, prediction, prediction_bin, target_names, model_name, cm):

    # Path
    csv_report = f'results/{model_name}/report.csv'
    csv_report_mlcm = f'results/{model_name}/report_mlcm.csv'
    csv_path_auc = f'results/{model_name}/roc_auc.csv'
    
    # Convert strings to Path type
    csv_report = pathlib.Path(csv_report)
    csv_report_mlcm = pathlib.Path(csv_report_mlcm)
    csv_path_auc = pathlib.Path(csv_path_auc)

    # Make sure the files are saved in a folder that exists
    csv_report.parent.mkdir(parents=True, exist_ok=True)
    csv_report_mlcm.parent.mkdir(parents=True, exist_ok=True)
    csv_path_auc.parent.mkdir(parents=True, exist_ok=True)

    # Get the reports
    report = skmetrics.classification_report(y_test,prediction_bin,output_dict=True,target_names=target_names)
    report_mlcm = mlcm.stats(cm, False)

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
def plot_results(history, name, metric, plot_path='plots'):

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

# Transform the values in the confusion matrix in percentage
def get_cm_percent(cm, total):
    cm_perc = np.zeros_like(cm, dtype='float')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            for k in range(cm.shape[2]):
                cm_perc[i][j][k] = round((cm[i][j][k] / total) * 100, 2)
    return cm_perc

# Plot confusion matrix
def plot_confusion_matrix(y_test, y_pred, model_name, target_names, plot_path='results', print_note='false'):

    # Make sure the plot folder exists
    plot_path = pathlib.Path(plot_path) / model_name
    plot_path.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    cm, _ = mlcm.cm(y_test, y_pred)
    target_names = np.array([*target_names, 'NoC'])

    # Calculating the normalization of the confusion matrix
    divide = cm.sum(axis=1, dtype='int64')
    divide[divide == 0] = 1
    cm_norm = 100 * cm / divide[:, None]

    # Plot the confusion matrix
    fig, ax = plot_cm(cm_norm, target_names)
    name = f"{model_name.split('-')[0]}-cm"
    tight_kws = {'rect' : (0, 0, 1.1, 1)}
    putils.save_fig(fig, name, path=plot_path, figsize='square',
                    tight_scale='both', usetex=False, tight_kws=tight_kws)

    # print('Raw confusion Matrix:')
    # print(cm)
    # print('Normalized confusion Matrix (%):')
    # print(cm_norm)

    return cm


def plot_cm(confusion_matrix, class_names, fontsize=10, cmap='Blues'):

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
