import numpy as np
import os
import pathlib

def loading_results_and_generating_error_file():

    for instance in ('weights_before', 'weights_after', 'gradients'):
        torch_gradients = np.load(f'torch_gradients/{instance}.npz')
        keras_gradients = np.load(f'keras_gradients/{instance}.npz')

        errors = {}
        for file in ('batchnorm0', 'batchnorm1', 'conv_weights', 'conv_bias', 'linear_weights', 'linear_bias', 'forward', 'loss'):
            error = calculation_the_errors(torch_gradients[f'{file}'], keras_gradients[f'{file}'])
            errors[f'{file}'] = error

        path_errors = pathlib.Path(f'errors/{instance}.npz')
        path_errors.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path_errors, **errors)


def calculation_the_errors(array1, array2):
    numerator = np.linalg.norm(array1 - array2)
    denominator = np.linalg.norm(array1)
    return numerator / denominator

def opening_errors_file(paths=[f'errors/weights_before.npz', f'errors/weights_after.npz', f'errors/gradients.npz']):

    for path in paths:
        spliting = path.split('/')
        print(f'######################## Printing norms ########################')
        print(f'{spliting[-1]}')
        path = pathlib.Path(path)
        errors = np.load(path)
        for key in errors:
            print(f'{key = }, {errors[key] = }')

loading_results_and_generating_error_file()
opening_errors_file()