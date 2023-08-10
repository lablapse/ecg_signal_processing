import utils
import utils_general
import utils_torch
import utils_lightning

import numpy as np
import torch

def test_funcs(data, func_torch, func_keras):
    y_torch = func_torch(torch.tensor(data))
    y_keras = func_keras
    return np.norm(y_torch - y_keras) / np.norm(y_keras)

data = np.random.randn(1, 128, 1000)
func_torch = utils_torch.skip_connection_torch()
func_keras = utils.skip_connection(data)

print()