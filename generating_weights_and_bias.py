import utils
import utils_general
import utils_torch
import utils_lightning

import numpy as np
import torch
import keras
import pickle
import tqdm

# This script generates weights to be passed to keras and pytorch models.
# The objective of using the same weights in both framework is to
# to make more fair comparations of performance.

initializer = keras.initializers.he_normal()
matrix = []
with open('dimensoes_weights_keras.txt', 'r') as f:
    for dimension in tqdm.tqdm(f):
        dimension = dimension[1:-3]
        dimension = tuple(map(int, dimension.split(',')))
        matrix_atual = np.array(initializer(dimension))
        matrix_atual = map(lambda value : np.round(value, decimals=4), matrix_atual)
        matrix.append(np.array(list(matrix_atual)))
with open("matrizes_weight_keras", "wb") as fp: # https://stackoverflow.com/questions/27745500/how-to-save-a-list-to-a-file-and-read-it-as-a-list-type
    pickle.dump(matrix, fp)
print()