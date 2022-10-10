# Python packages
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Flatten, Dense, ReLU, Add
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
import pandas as pd
import seaborn as sns

import utils

# Load data
data = np.load('dados.npz')

# Training set
X_train = data['X_train']
y_train = data['y_train']

# Test set
X_test = data['X_test']
y_test = data['y_test']

# Create the skip connection A with MaxPooling and Conv layers
def skip_connection(skip, num_filter=128, rate_drop=0, initializer='none', downsample=1):
    skip = MaxPooling1D(pool_size=downsample,strides=downsample,padding='same')(skip)
    skip = Conv1D(filters=num_filter,kernel_size=1,strides=1,padding='same')(skip)
    return skip

# Create the residual blocks
def residual_blocks(input, i=0, num_filter=128, rate_drop=0, initializer='none', downsample=1):

    layer, skip = input

    skip = skip_connection(skip, num_filter=num_filter, rate_drop=rate_drop, initializer=initializer, downsample=downsample)

    layer = Conv1D(kernel_size=16, filters=num_filter, strides=1, padding="same", kernel_initializer=initializer)(layer) 
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = Dropout(rate_drop)(layer)
    layer = Conv1D(kernel_size=16, filters=num_filter, strides=downsample, padding="same", kernel_initializer=initializer)(layer) 

    layer = Add()([layer, skip])

    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = Dropout(rate_drop)(layer)

    if i==3: skip = None

    return layer, skip

# Input layer
input_layer = Input(shape=X_train.shape[1:]) # Size = (1000, 12)

initializer = 'he_normal'
rate = 0.8
rate_drop = 1- rate
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
    layer, skip = residual_blocks([layer,skip], i=i, num_filter = num_filter[i], rate_drop=rate_drop, initializer=initializer, downsample=downsample)

# Output block
layer = Flatten()(layer)
classification = Dense(5,activation='sigmoid',kernel_initializer=initializer)(layer)



# Constructing the model
model = Model(inputs=input_layer, outputs=classification)

# model = utils.get_model(input_layer, 'rajpurkar')
m = model.summary()
print(m)


print(end)