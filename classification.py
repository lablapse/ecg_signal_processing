# Python packages
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Flatten, Dense, ReLU, Add
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_roc_curve
from sklearn.utils.class_weight import compute_sample_weight
import pandas as pd
import seaborn as sns

from datetime import datetime
import pathlib

import utils
import train_model

# Load data
data = np.load('data_val_2.npz') # Dataset with train, val and test, and in the correct sequence of classes

# Training set
X_train = data['X_train']
y_train = data['y_train']
# Validation set
X_val = data['X_val']
y_val = data['y_val']
# Test set
X_test = data['X_test']
y_test = data['y_test']

# Sequence of classes names
target_names = ['NORM', 'STTC', 'CD', 'MI', 'HYP']

# Get the model
# model_name = 'rajpurkar'
model_name = 'ribeiro'

# Save the model name with the data and time
timestamp = datetime.now().isoformat()
model_name = f'{model_name}-{timestamp}'

# Get the model
input_layer = Input(shape=X_train.shape[1:])
model = utils.get_model(input_layer, model_name)
# model.summary()

# Train the model
history = train_model.training(model, X_train, y_train, X_val, y_val, model_name, save_parameters=True)
# print(history.params)
# print(model_name)

# Evaluate the model
score = model.evaluate(X_test, y_test)
print(f"Custo de teste = {score[0]:.4f}")
print(f"Acurácia de teste = {100*score[1]:.2f}%")

# Prediction of the model
prediction = model.predict(X_test)
# Convert the predictions to binary values
prediction_bin = np.array(prediction)
prediction_bin = (prediction > 0.5).astype('int')

# # Save results
cm = utils.plot_confusion_matrix(y_test, prediction_bin, model_name, target_names)
utils.get_metrics(y_test, prediction, prediction_bin, target_names, model_name, cm)
utils.plot_results(history, name=model_name, metric='loss')
# utils.plot_results(history, name=model_name, metric='accuracy')
