import utils
import utils_general
import utils_torch
import utils_lightning

import numpy as np
import torch
import keras

def test_funcs(data, func_torch, func_keras):
    with torch.no_grad():
        y_torch = func_torch(torch.from_numpy(data))
    y_keras = np.transpose(func_keras, axes=(0, 2, 1))
    return np.linalg.norm(y_torch - y_keras) / np.linalg.norm(y_keras)

def test_funcs_realmente(data, func_torch, func_keras):
    with torch.no_grad():
        y_torch = func_torch(torch.from_numpy(data))
    y_keras = func_keras(np.transpose(data, axes=(0, 2, 1)))
    y_keras = np.transpose(y_keras, (0, 2, 1))
    return np.linalg.norm(y_torch - y_keras) / np.linalg.norm(y_keras)

data = np.random.randn(1, 128, 1000)
data = data.astype(np.float32)
# func_torch = utils_torch.skip_connection_torch()
# func_keras = utils.skip_connection(np.transpose(data, axes=(0, 2, 1)))
# erro = test_funcs(data, func_torch, func_keras)

# defining the functions
func_torch = torch.nn.Conv1d(in_channels=128, out_channels=128, kernel_size=16)
func_keras = keras.layers.Conv1D(filters=128, kernel_size=16, kernel_initializer='ones', bias_initializer='ones')

# defining the weights and bias
weights = np.ones((func_torch.weight.shape), dtype=np.float32)
bias = np.ones((func_torch.bias.shape), dtype=np.float32)
torch_weights = torch.nn.Parameter(torch.from_numpy(weights))
torch_bias = torch.nn.Parameter(torch.from_numpy(bias))
keras_weights = np.transpose(weights, axes=(2, 1, 0))
keras_bias = bias

# passing the weights and bias
func_torch.weight = torch_weights
func_torch.bias = torch_bias

erro = test_funcs_realmente(data, func_torch, func_keras)

print()