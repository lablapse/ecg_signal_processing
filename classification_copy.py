# Python packages
from datetime import datetime
from keras.layers import Input
import mlcm
import train_model
import utils
import utils_general
import numpy as np


# Load data
X_train, y_train, X_val, y_val, X_test, y_test = utils_general.load_data(test=True)
X_train = np.transpose(X_train, axes=(0,2,1))
X_val = np.transpose(X_val, axes=(0,2,1))
X_test = np.transpose(X_test, axes=(0,2,1))

# Sequence of classes names
target_names = ['NORM', 'STTC', 'CD', 'MI', 'HYP']

# Get the model
# model_name = 'rajpurkar'
model_name = 'ribeiro'

# Save the model name with the data and time
timestamp = datetime.now().isoformat()
model_name = f'{model_name}-{timestamp}'

# Get the model
input_shape = Input(shape=X_train.shape[1:])
model = utils.get_model(input_shape, model_name)
model = utils_general.setting_keras_weights(model, 'matrizes_weight_keras')
model.summary()

# Train the model
history = train_model.training(model, X_train, y_train, X_val, y_val, model_name, save_parameters=True)

# Evaluate the model
score = model.evaluate(X_test, y_test)
print(f"Custo de teste = {score[0]:.4f}")
print(f"Acurácia de teste = {100*score[1]:.2f}%")

# Prediction of the model
prediction = model.predict(X_test)
# Convert the predictions to binary values
prediction_bin = (prediction > 0.5).astype('int')

# Save results
cm, _ = mlcm.cm(y_test, prediction_bin, print_note=False)
utils_general.plot_confusion_matrix(cm, model_name, target_names)
# utils.get_metrics_skmetrics(y_test, prediction, prediction_bin, target_names, model_name)
print(f"#############################################################################")
mlcm.stats(cm)
utils_general.get_mlcm_report(cm, target_names, model_name)
print()
