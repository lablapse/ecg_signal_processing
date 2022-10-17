# File that contain all the function used in the ECG classification code

# Python packages
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Flatten, Dense, ReLU, Add
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight

import pandas as pd
import seaborn as sns

# Get the weights for deal with the imbalanced dataset
def get_weights(y_train):
    # Get the possible labels in the training set
    y_unique = np.unique(y_train, axis=0)

    lst = [0 for x in range(len(y_unique))]
    soma = 0

    # Count the quantity of the labels y_unique appears in y_train
    for i in range(len(y_train)):
        for j in range(len(y_unique)):
            if np.array_equal(y_train[i], y_unique[j]):
                lst[j] += 1

    y_unique_str = ['0' for x in range(len(y_unique))]
    perc = [0 for x in range(len(y_unique))]
    perc_inv = [0 for x in range(len(y_unique))]
    lst_perc = [0 for x in range(len(y_train))]

    # Get the labels and transform in string / Sum the total of labels
    for i in range(len(y_unique)):
        y_unique_str[i] = str(y_unique[i])
        soma += lst[i]

    # Calculate the the ratio of quantity of labels in the training set
    for i in range(len(lst)):
        perc[i] = round(((lst[i] / soma) * 100), 2)
        perc_inv[i] = round(((1 - (lst[i] / soma)) * 100), 2)

    # Get a list with the corresponding weight for all the examples of the y_train
    for i in range(len(y_train)):
        for j in range(len(y_unique)):
            if np.array_equal(y_train[i], y_unique[j]):
                lst_perc[i] = perc_inv[j]

    d = {'Labels':y_unique_str, 'Quantity':lst, 'Percent (%)':perc, 'Inv Percent (%)':perc_inv}

    df = pd.DataFrame(data=d)
    print(df)

    return np.array(lst_perc)


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
        rate_drop = 0.5
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



# Transform the binary vector in a list with strings (could be y_test or the predictions)
def get_strings(label_string,label_bin):
    label_bin_string = []
    for x in range(len(label_bin)):
        lst = []
        for y in range(len(label_string)):
            value = label_bin[x][y]
            if value == 1:
                lst.append(label_string[y])
        label_bin_string.append(lst)
    
    return label_bin_string

# Transform the values in the confusion matrix in percentage
def get_cm_percent(cm, total):
    cm_perc = np.zeros_like(cm, dtype='float')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            for k in range(cm.shape[2]):
                cm_perc[i][j][k] = round((cm[i][j][k] / total) * 100, 2)
    return cm_perc
