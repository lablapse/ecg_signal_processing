import keras
from keras.layers import Conv1D, MaxPooling1D, Dropout, BatchNormalization, Flatten, Dense, ReLU, Add
from keras.activations import sigmoid
# from keras.losses import binary_crossentropy
from keras.losses import BinaryCrossentropy
import numpy as np
import os
import pathlib
import tensorflow as tf

# Thats important because in the equivalent Pytorch script, the GPU is not used.
os.environ["CUDA_VISIBLE_DEVICES"]="" 

# This function receives a keras operation and a numpy array, then, saves the calculated result
def calculate_and_save(keras_operation, numpy_array, path='comparing_gradient_keras_torch/keras_results/'):

    assert numpy_array.dtype == np.float32

    if keras_operation == Conv1D:
        # numpy_array = np.transpose(numpy_array, axes=(0, 2, 1))

        # assert result.shape == (numpy_array.shape[0], numpy_array.shape[1], numpy_array.shape[2] * 2)
        layer = keras_operation(numpy_array.shape[2] * 2, 1, kernel_initializer='ones', bias_initializer='zeros')
        
        with tf.GradientTape() as tape:
            y = layer(numpy_array)
            labels = np.ones(y.shape)
            loss = BinaryCrossentropy()(y, labels)
            grad = tape.gradient(loss, layer.trainable_variables)
            

            result = {}
            result['loss'] = loss
            for var, g in zip(layer.trainable_variables, grad):
                print(f'{var.name = }, shape: {g.shape = }')
                result[f'{var.name}'] = g

        name = 'conv'

    elif keras_operation == BatchNormalization:
        layer = keras_operation()

        with tf.GradientTape() as tape:
            y = layer(numpy_array, training=True)
            labels = np.ones(y.shape)
            loss = BinaryCrossentropy()(y, labels)
            grad = tape.gradient(loss, layer.trainable_variables)
            
            result = {}
            result['loss'] = loss
            for var, g in zip(layer.trainable_variables, grad):
                print(f'{var.name = }, shape: {g.shape = }')
                result[f'{var.name}'] = g

        name = 'batchnorm'

    elif keras_operation == ReLU:
        result = keras_operation()(numpy_array)

        assert result.shape == numpy_array.shape
        name = 'relu'

    elif keras_operation == sigmoid:
        result = keras_operation(numpy_array)

        assert result.shape == numpy_array.shape
        name = 'sigmoid'

    elif keras_operation == Dense:

        channels_out = numpy_array.shape[2] // 2
        
        layer = keras_operation(channels_out, kernel_initializer='ones', bias_initializer='zeros')

        with tf.GradientTape() as tape:
            y = layer(numpy_array)
            labels = np.ones(y.shape)
            loss = BinaryCrossentropy()(y, labels)
            grad = tape.gradient(loss, layer.trainable_variables)

            result = {}
            result['loss'] = loss
            for var, g in zip(layer.trainable_variables, grad):
                print(f'{var.name = }, shape: {g.shape = }')
                result[f'{var.name}'] = g

        name = 'linear'

    else:
        raise ValueError(f'Missing if statements. Check the operation in keras_operation')

    # path = pathlib.Path(f'{path}{name}.npz')
    # path.parent.mkdir(parents=True, exist_ok=True)
    # result = np.transpose(result, axes=(0, 2, 1))
    # np.savez(path, result)

    return

data = np.load('comparing_gradient_keras_torch/informacao_para_as_comparacoes.npz')
numpy_array = data['arr_0']
numpy_array = np.transpose(numpy_array, axes=(0, 2, 1))

# keras_operations = [Conv1D, BatchNormalization, ReLU, Dense]
# keras_operations = [Conv1D, BatchNormalization, ReLU, sigmoid, Dense]
keras_operations = [Conv1D, BatchNormalization, Dense]

for keras_operation in keras_operations:
    calculate_and_save(keras_operation, numpy_array)