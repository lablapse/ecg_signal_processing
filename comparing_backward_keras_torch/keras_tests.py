import keras
from keras.layers import Conv1D, MaxPooling1D, Dropout, BatchNormalization, Flatten, Dense, ReLU, Add
from keras.activations import sigmoid
import numpy as np
import os
import pathlib
import tensorflow as tf

# Thats important because in the equivalent Pytorch script, the GPU is not used.
os.environ["CUDA_VISIBLE_DEVICES"]="" 

# Function to create the Keras model
def keras_tests():
    model = keras.Sequential()
    model.add(keras.Input(shape=(64,64)))
    model.add(BatchNormalization())
    model.add(Conv1D(4, 1))
    model.add(Flatten())
    model.add(Dense(5))

    return model
    
def putting_alltogether():  
    loss_fn = keras.losses.CategoricalCrossentropy(reduction='sum_over_batch_size', from_logits=True)

    # loading the numpy arrays that will be used as X and Y
    data = np.load('data.npz')
    X, Y = np.transpose(data['X'], axes=(0,2,1)), data['Y']
    
    # defining some of the structure of the model
    model = keras_tests()

    # loading the weights for convolution and linear, and applying them to the model
    weights = np.load('weights.npz')
    layers = model.layers
    layers[1].set_weights([np.transpose(weights['conv_weights'], axes=(2,1,0)), weights['conv_bias']])
    layers[3].set_weights([np.transpose(weights['linear_weights'], axes=(1,0)), weights['linear_bias']])


    # model.compile(loss=loss_fn, optimizer=keras.optimizers.Adam(learning_rate=0.01))
    model.compile(loss=loss_fn, optimizer=keras.optimizers.SGD(learning_rate=0.01))

    pred = model(X)
    loss = loss_fn(Y, pred)

    # Saving gradients before .fit method
    saving_weights(model, 'keras_gradients/weights_before', [pred, loss])

    model.fit(X, Y, epochs=1, batch_size=X.shape[0])

    collecting_and_saving_gradients(X, Y, model)

    new_pred = model(X)
    new_loss = loss_fn(Y, new_pred)

    # Saving gradients after .fit method
    saving_weights(model, 'keras_gradients/weights_after', [new_pred, new_loss])


# This function does not exist in the equivalent Pytorch script. It's integrated in the 'train_loop' function
def saving_weights(model, path, pred_and_loss):
    dict_weights = dict(batchnorm0 = model.layers[0].get_weights()[0],
                         batchnorm1 = model.layers[0].get_weights()[1],
                         conv_weights = np.transpose(model.layers[1].get_weights()[0], axes=(2,1,0)),
                         conv_bias = model.layers[1].get_weights()[1],
                         linear_weights = np.transpose(model.layers[3].get_weights()[0], axes=(1,0)),
                         linear_bias = model.layers[3].get_weights()[1],
                         forward = pred_and_loss[0],
                         loss = pred_and_loss[1],
    )

    path = pathlib.Path(f'{path}.npz')
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(path, **dict_weights)

def collecting_and_saving_gradients(X, Y, model, path='keras_gradients/gradients'):
    
    # That's a way to compute the gradients in keras. 
    with tf.GradientTape() as tape:
        pred = model(X)
        loss = model.loss(Y, pred)

    grads = tape.gradient(loss, model.trainable_variables)
    
    # Here, the forward and loss are not calculated, but, are passed to the dictionary
    # to maintain a pattern.
    dict_grads = dict(batchnorm0 = grads[0],
                      batchnorm1 = grads[1],
                      conv_weights = np.transpose(grads[2], axes=(2,1,0)),
                      conv_bias = grads[3],
                      linear_weights = np.transpose(grads[4], axes=(1,0)),
                      linear_bias = grads[5],
                      forward = 0,
                      loss = 0
    )

    path = pathlib.Path(f'{path}.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **dict_grads)


putting_alltogether()