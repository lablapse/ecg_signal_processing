# Python packages
import numpy as np
import keras
import pandas as pd

import matplotlib.pyplot as plt
import utils

# Load data
data = np.load('data_val.npz')

# Training set
X_train = data['X_train']
y_train = data['y_train']
# Validation set
X_val = data['X_val']
y_val = data['y_val']
# Test set
X_test = data['X_test']
y_test = data['y_test']

target_names = ['CD', 'HYP', 'MI', 'NORM', 'STTC']

# Get the model
# author_name = 'rajpurkar'
# author_name = 'ribeiro'
model_name = 'ribeiro-2022-11-09T20:01:18.469718'

model = keras.models.load_model(f'results/{model_name}/model.h5')
history = pd.read_csv(f'results/{model_name}/history.csv')

# Prediction of the model
prediction = model.predict(X_test)
# Convert the predictions to binary values
prediction_bin = np.array(prediction)
prediction_bin = (prediction > 0.5).astype('int')

# Save results
utils.get_metrics(y_test, prediction, prediction_bin, target_names)
utils.plot_confusion_matrix(y_test, prediction_bin, model_name, target_names)
utils.plot_results(history, name=model_name, metric='loss')
utils.plot_results(history, name=model_name, metric='accuracy')

plt.show()

print()
