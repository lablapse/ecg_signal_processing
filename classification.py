# Python packages
from datetime import datetime
from keras.layers import Input
import mlcm
import numpy as np
import train_model
import utils

# Load data
X_train, y_train, X_val, y_val, X_test, y_test = utils.load_data(test=True)

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
utils.plot_confusion_matrix(cm, model_name, target_names)
# utils.get_metrics_skmetrics(y_test, prediction, prediction_bin, target_names, model_name)
print(f"#############################################################################")
mlcm.stats(cm)
utils.get_mlcm_report(cm, target_names, model_name)
print()
# utils.plot_results(history, name=model_name, metric='loss')
# utils.plot_results(history, name=model_name, metric='accuracy')