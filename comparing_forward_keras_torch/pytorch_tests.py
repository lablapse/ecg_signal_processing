import numpy as np
import pathlib
import torch
import torch.nn as nn

# This function receives a pytorch operation and a numpy array, then, saves the calculated result
def calculate_and_save(torch_operation, numpy_array, path='comparing_forward_keras_torch/torch_results/'):

    assert numpy_array.dtype == np.float32

    with torch.no_grad():
    # The kernels have some weight initialization. Wanting to make the operations
    # behave the same both in Keras and Pytorch, we need to impose 

        if torch_operation == nn.Conv1d:
            operation = torch_operation(numpy_array.shape[1], numpy_array.shape[1] * 2, 1)
            operation.weight.data *= 0
            operation.bias.data *= 0
            operation.weight.data += 1
            result = operation(torch.as_tensor(numpy_array))

            assert result.shape == (numpy_array.shape[0], numpy_array.shape[1] * 2, numpy_array.shape[2])
            name = 'conv'

        elif torch_operation == nn.BatchNorm1d:
            operation = torch_operation(numpy_array.shape[1], eps=0.001, momentum=0.99)
            result = operation(torch.as_tensor(numpy_array))

            assert result.shape == numpy_array.shape
            name = 'batchnorm'

        elif torch_operation == nn.ReLU:
            result = torch_operation()(torch.as_tensor(numpy_array))

            assert result.shape == numpy_array.shape
            name = 'relu'

        elif torch_operation == nn.Sigmoid:
            torch_array = torch.as_tensor(numpy_array)
            result = torch_operation()(torch_array)

            assert result.shape == numpy_array.shape
            name = 'sigmoid'

        elif torch_operation == nn.Linear:
            numpy_array = np.transpose(numpy_array, axes=(0, 2, 1))
            channels = numpy_array.shape[2]
            
            operation = torch_operation(in_features=channels, out_features=channels // 2)

            operation.weight.data *= 0
            operation.bias.data *= 0
            operation.weight.data += 1

            result = operation(torch.as_tensor(numpy_array))
            assert result.shape == (numpy_array.shape[0], numpy_array.shape[1], numpy_array.shape[2] // 2)
            result = np.transpose(result, axes=(0, 2, 1))
            name = 'linear'

        else:
            raise ValueError(f'Missing if statements or exciding torch operations.')

    path = pathlib.Path(f'{path}{name}.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(path, result)

    return

data = np.load('comparing_forward_keras_torch/informacao_para_as_comparacoes.npz')
numpy_array = data['arr_0']

torch_operations = [nn.Conv1d, nn.BatchNorm1d, nn.ReLU, nn.Sigmoid, nn.Linear]

for torch_operation in torch_operations:
    calculate_and_save(torch_operation, numpy_array)