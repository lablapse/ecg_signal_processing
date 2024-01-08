import numpy as np
import os
import pathlib

def loading_results_and_generating_error_file(path='comparing_keras_torch/'):

    files_list = []
    subdir_list = []
    for subdir, dirs, files in os.walk(path):
        if subdir not in subdir_list and dirs == []:
            subdir_list.append(subdir)
        if files not in files_list and dirs == []:
            files_list.append(files)
    
    
    errors = {}
    for file in files_list[0]:
        result_0 = np.load(f"{subdir_list[0]}/{file}")
        result_0 = result_0['arr_0']
        result_1 = np.load(f"{subdir_list[1]}/{file}")
        result_1 = result_1['arr_0']
        if len(result_1.shape) == 3:
            result_1 = np.transpose(result_1, axes=(0, 2, 1))

        error = calculation_the_errors(result_0, result_1)
        
        errors[f'{file[:-4]}'] = error

    path = pathlib.Path(f'errors/erros.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **errors)

def calculation_the_errors(array1, array2):
    numerator = np.linalg.norm(array1 - array2)
    denominator = np.linalg.norm(array1)
    return numerator / denominator

def opening_errors_file(path='errors/erros.npz'):

    path = pathlib.Path(path)
    errors = np.load(path)
    for key in errors:
        print(f'{key = }, {errors[key] = }')

loading_results_and_generating_error_file()
# opening_errors_file()