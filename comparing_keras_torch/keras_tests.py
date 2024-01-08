import keras
from keras.layers import Conv1D, MaxPooling1D, Dropout, BatchNormalization, Flatten, Dense, ReLU, Add
from keras.activations import sigmoid
import numpy as np
import os
import pathlib

# Thats important because in the equivalent Pytorch script, the GPU is not used.
os.environ["CUDA_VISIBLE_DEVICES"]="" 

# This function receives a keras operation and a numpy array, then, saves the calculated result
def calculate_and_save(keras_operation, numpy_array, path='comparing_keras_torch/keras_results/'):

    assert numpy_array.dtype == np.float32

    if keras_operation == Conv1D:
        result = keras_operation(numpy_array.shape[2] * 2, 1, kernel_initializer='ones', bias_initializer='zeros')(numpy_array)
        name = 'conv'

    elif keras_operation == BatchNormalization:
        result = keras_operation()(numpy_array)
        name = 'batchnorm'

    elif keras_operation == ReLU:
        result = keras_operation()(numpy_array)
        name = 'relu'

    elif keras_operation == sigmoid:
        shapes = numpy_array.shape
        total = shapes[0] * shapes[1] * shapes[2]
        result = Dense(total, activation='sigmoid', kernel_initializer='ones', bias_initializer='zeros')(numpy_array)
        name = 'sigmoid'

    elif keras_operation == Dense:
        numpy_array = numpy_array[:,0,:]

        shapes = numpy_array.shape
        total = shapes[0] * shapes[1]
        
        result = keras_operation(total // 2, kernel_initializer='ones', bias_initializer='zeros')(numpy_array)
        name = 'linear'

    else:
        raise ValueError(f'Missing if statements or exciding torch operations.')

    path = pathlib.Path(f'{path}{name}.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(path, result)

    return

data = np.load('comparing_keras_torch/informacao_para_as_comparacoes.npz')
numpy_array = data['arr_0']
numpy_array = np.transpose(numpy_array, axes=(0, 2, 1))

keras_operations = [Conv1D, BatchNormalization, ReLU, sigmoid, Dense]

for keras_operation in keras_operations:
    calculate_and_save(keras_operation, numpy_array)