# Python packages
import numpy as np
from keras.layers import Input
from datetime import datetime

import utils
import train_model

# Load data
# The dataset contains train, validation and test sets
data = np.load('data.npz')

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
model.summary()

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

# Save results
cm = utils.plot_confusion_matrix(y_test, prediction_bin, model_name, target_names)
utils.get_metrics(y_test, prediction, prediction_bin, target_names, model_name, cm)
# utils.plot_results(history, name=model_name, metric='loss')
# utils.plot_results(history, name=model_name, metric='accuracy')
