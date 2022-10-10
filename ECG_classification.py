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

# Data summary
# print(f'X_train = {X_train.shape}')
# print(f'y_train = {y_train.shape}')
# print(f'X_test = {X_test.shape}')
# print(f'y_test = {y_test.shape}')

# Get the sample_weights for deal with the imbalanced dataset
# If method equal to 0, get the weights by the method of 1 - num_amostras_label / num_amostras_totais
# If method unequal to 0, get the weights by the function of sklearn
method = 1
if method == 0:
    sample_weights_train = utils.get_weights(y_train=y_train)
else:
    sample_weights_train = compute_sample_weight(class_weight='balanced',y=y_train)

# print(sample_weights_train)


# Input layer
input_layer = Input(shape=X_train.shape[1:])

# model_name = 'rajpurkar'
model_name = 'ribeiro'
print(f'\n\nModel name = ', model_name)
model = utils.get_model(input_layer, model_name)
m = model.summary()


# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Callbacks
callbacks = []
# filepath='C:/Users/sarah/TCC/ECG Classification Models/Weights improvement/weights-improvement-{epoch:02d}-{val_loss:.2f}.h5'
monitor = 'val_loss'
callbacks.append(ReduceLROnPlateau(monitor=monitor, factor=0.1, patience=10, min_lr=0.001/1000))
# callbacks.append(EarlyStopping(monitor=monitor, mode='auto', verbose=1, patience=10))
# callbacks.append(ModelCheckpoint(filepath, monitor=monitor, mode='auto', verbose=1, save_best_only=True))
print(callbacks)

# Training the model
# history = model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_test, y_test), callbacks=callbacks)
history = model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_test, y_test), callbacks=callbacks,sample_weight=sample_weights_train)

# Plot results
plt.plot(history.epoch, history.history['loss'], '-o')
plt.plot(history.epoch, history.history['val_loss'], '-*')

plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.legend(['Training set', 'Validation set'])
plt.show()

#Plot results
plt.plot(history.epoch, history.history['accuracy'], '-o')
plt.plot(history.epoch, history.history['val_accuracy'], '-*')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training set', 'Validation set'])
plt.show()

# Evaluate the model
score = model.evaluate(X_test, y_test)
print(f"Custo de teste = {score[0]:.4f}")
print(f"Acurácia de teste = {100*score[1]:.2f}%")

# Prediction of the model
prediction = model.predict(X_test)
prediction_bin = np.array(prediction)
prediction_bin = (prediction > 0.5).astype('int')
print(f'Prediction: {prediction_bin[67]} \t\tReal Label: {y_test[67]}')

# np.savez('prediction.npz', prediction_bin)

# List with the labels strings
label_string = ['NORM','MI','CD','STTC','HYP']

# Y_test string labels
y_test_string = utils.get_strings(label_string, y_test)

# Predictions strings labels
prediction_string = utils.get_strings(label_string, prediction_bin)

# Visualizing an example
index = 67
print(f'Index: {index}\n')

print(f'Diagnostic = {y_test[index]}')
print(f'Prediction = {prediction_bin[index]}\n')

print(f'Diagnostic = {y_test_string[index]}')
print(f'Prediction = {prediction_string[index]}')

# Accuracy from the example below
acc_index = accuracy_score(y_test[index],prediction_bin[index])
print(f'Example accuracy = {100 * acc_index:.2f}%')

# Plot the example with the diagnostic and prediction
valor_med = X_test[index, ].mean(axis=-1)
fig_s, ax_s = plt.subplots(figsize=(10,7))
ax_s.set_title(f'Diagnostic: {y_test_string[index]}       Prediction:{prediction_string[index]}')
ax_s.plot(valor_med)
plt.show()

# Another metrics
report = classification_report(y_test,prediction_bin,output_dict=True,target_names=['NORM', 'MI', 'CD', 'STTC', 'HYP'])

report_df = pd.DataFrame.from_dict(report, orient='index')
print(report_df)

roc_auc = roc_auc_score(y_test, prediction, average=None)
print(f'ROC AUC score = {roc_auc}')

# Confusion matrix
cm = multilabel_confusion_matrix(y_test, prediction_bin)
cm_perc = utils.get_cm_percent(cm = cm, total = len(prediction_bin))

# Plot confusion matrix
fig = plt.figure(figsize = (14, 8))

for i, (label, matrix) in enumerate(zip(label_string, cm_perc)):
    plt.subplot(f'23{i+1}')
    labels = [f'Not {label}', label]
    ax = sns.heatmap(matrix, annot = True, square = True, fmt = '.2f', cbar = False, cmap = 'Blues',
                xticklabels = labels, yticklabels = labels, linecolor = 'black', linewidth = 1)
    for t in ax.texts: t.set_text(t.get_text() + "%")
    plt.title(label)
plt.tight_layout()
plt.show()


# Multilabel model evaluation:
# https://colab.research.google.com/github/kmkarakaya/ML_tutorials/blob/master/Multi_Label_Model_Evaulation.ipynb#scrollTo=w26dgZXKzhJN
