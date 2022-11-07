# Python packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Conv1D, MaxPooling1D, Dropout, BatchNormalization, Flatten, Dense, ReLU, Add
from keras.models import Model
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import multilabel_confusion_matrix


# Rajpurkar model functions
# Create residual blocks
def residual_blocks_rajpurkar(x,i=0, stride=1, num_filter=64, rate_drop=0.5, initializer='none'):

    bn_1 = BatchNormalization()(x)
    relu_1 = ReLU()(bn_1)
    drop_1 = Dropout(rate_drop)(relu_1)
    conv_1 = Conv1D(kernel_size=16, filters=num_filter, strides=1, padding="same", kernel_initializer=initializer)(drop_1)
    bn_2 = BatchNormalization()(conv_1)
    relu_2 = ReLU()(bn_2)
    drop_2 = Dropout(rate_drop)(relu_2)
    conv_2 = Conv1D(kernel_size=16, filters=num_filter, strides=stride, padding="same", kernel_initializer=initializer)(drop_2)

    if i == 3 or i == 7 or i == 11:  #Verifica se houve mudança na quantidade de número de filtros
        #Short connection
        conv_aj = Conv1D(kernel_size=16, filters=num_filter, strides=1, padding="same")(x) #Ajustar o número de filtros
        short = MaxPooling1D(pool_size = 1, strides=2)(conv_aj)
    else:
        #Short connection
        short = MaxPooling1D(pool_size = 1, strides=stride)(x)

    # Adding layers
    return Add()([conv_2, short])


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

    if model_name == 'rajpurkar':
        rate_drop = 1 - 0.8
        initializer='he_normal'

        # First layer
        conv_1 = Conv1D(kernel_size=16, filters=64, strides=1, padding="same", kernel_initializer=initializer)(input_layer)
        bn_1 = BatchNormalization()(conv_1)
        relu_1 = ReLU()(bn_1)

        # Second layer
        conv_2 = Conv1D(kernel_size=16, filters=64, strides=1, padding="same", kernel_initializer=initializer)(relu_1)
        bn_2 = BatchNormalization()(conv_2)
        relu_2 = ReLU()(bn_2)
        drop_1 = Dropout(rate_drop)(relu_2)
        conv_3 = Conv1D(kernel_size = 16, filters=64, strides=2, padding="same", kernel_initializer=initializer)(drop_1)

        # Short connection
        short_1 = MaxPooling1D(pool_size=1, strides=2)(relu_1)

        # Adding layers
        layers = Add()([conv_3, short_1])

        num_filter = np.array([64, 64, 64, 128, 128, 128, 128, 192, 192, 192, 192, 256, 256, 256, 256])
        for i in range(15):
            #print(f"i = {i} STRIDE = {(i % 2)+1}, FILTER LENGHT = {num_filter[i]}")
            layers = residual_blocks_rajpurkar(
                layers, i=i, stride=(i % 2)+1, num_filter = num_filter[i], 
                rate_drop=rate_drop, initializer=initializer
            )

        # Last layers
        # The ﬁnal fully connected layer and sigmoid activation produce a distribution 
        # over the 5 output superclasses for each time-step.
        bn_x = BatchNormalization()(layers)
        relu_x = ReLU()(bn_x)
        flat_x = Flatten()(relu_x)
        dense_x = Dense(32)(flat_x)
        classification = Dense(5, activation='sigmoid')(dense_x)
    
    elif model_name == 'ribeiro':
        initializer = 'he_normal'
        rate_drop = 1- 0.8
        downsample = 4

        # Input block
        layers = Conv1D(kernel_size=16, filters=64, strides=1, padding="same", kernel_initializer=initializer)(input_layer) # Output_size = (1000, 64)
        layers = BatchNormalization()(layers)
        layers = ReLU()(layers)

        num_filter = np.array([128, 192, 256, 320])

        layer = layers
        skip = layers

        # Residual Blocks
        for i in range(4):
            layer, skip = residual_blocks_ribeiro([layer,skip], num_filter = num_filter[i], rate_drop=rate_drop, initializer=initializer, downsample=downsample)

        # Output block
        layer = Flatten()(layer)
        classification = Dense(5,activation='sigmoid',kernel_initializer=initializer)(layer)

    # Constructing the model
    model = Model(inputs=input_layer, outputs=classification)

    return model

# Get the metrics
def get_metrics(y_test, prediction, prediction_bin, target_names):

    print("\nReports from the classification:")

    report = classification_report(y_test,prediction_bin,output_dict=True,target_names=target_names)
    report_df = pd.DataFrame.from_dict(report, orient='index')
    print(report_df)

    roc_auc_macro = roc_auc_score(y_test, prediction, average='macro')
    roc_auc_micro = roc_auc_score(y_test, prediction, average='micro')
    roc_auc_weighted = roc_auc_score(y_test, prediction, average='weighted')
    roc_auc_none = roc_auc_score(y_test, prediction, average=None)

    print(f'ROC AUC macro = {roc_auc_macro}')
    print(f'ROC AUC micro = {roc_auc_micro}')
    print(f'ROC AUC weighted = {roc_auc_weighted}')
    print(f'ROC AUC None = {roc_auc_none}')

    return

# Plot results
def plot_results(history,model_name,epochs):
    # Plot results
    plt.plot(history.epoch, history.history['loss'], '-o')
    plt.plot(history.epoch, history.history['val_loss'], '-*')

    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.legend(['Training set', 'Validation set'])
    plt.savefig(f'./plots/[{model_name}][Loss][{100}].png')
    # plt.show()

    # #Plot results
    plt.plot(history.epoch, history.history['accuracy'], '-o')
    plt.plot(history.epoch, history.history['val_accuracy'], '-*')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training set', 'Validation set'])
    plt.savefig(f'./plots/[{model_name}][Accuracy][100].png')
    # plt.show()

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
def plot_confusion_matrix(y_test, prediction_bin, model_name):
    # Confusion matrix
    cm = multilabel_confusion_matrix(y_test, prediction_bin)
    cm_perc = get_cm_percent(cm=cm, total=len(prediction_bin))

    print(f'\n{cm}')
    # print(f'\n{cm_perc}')

    # Plot confusion matrix
    # fig = plt.figure(figsize = (14, 8))

    # for i, (label, matrix) in enumerate(zip(label_string, cm_perc)):
    #     plt.subplot(f'23{i+1}')
    #     labels = [f'Not {label}', label]
    #     ax = sns.heatmap(matrix, annot = True, square = True, fmt = '.2f', cbar = False, cmap = 'Blues',
    #                 xticklabels = labels, yticklabels = labels, linecolor = 'black', linewidth = 1)
    #     for t in ax.texts: t.set_text(t.get_text() + "%")
    #     plt.title(label)
    # plt.tight_layout()
    # plt.savefig(f'./plots/[{model_name}][CM][100].png')
    # plt.show()


