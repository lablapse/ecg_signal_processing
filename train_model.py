# Python packages
import keras.optimizers as kopt
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger
import numpy as np
import pandas as pd
import pathlib


# Train the model
def training(
    model, X_train, y_train,
    X_val, y_val, model_name,
    save_parameters, learning_rate=0.1, 
    epochs=100, factor=0.5, patience_RLR=10,
    patience_ES=15, min_lr=1e-6, 
    loss='binary_crossentropy',
    batch_size=256, monitor = 'val_loss'
    ):

    '''
    inputs:
        model: keras.engine.functional.Functional;
        X_train: np.ndarray; 
        y_train: np.ndarray;
        X_val: np.ndarray;
        y_val: np.ndarray;
        model_name: str;
        save_parameters: bool -> if True, saves some information in a .csv file;
        learning_rate: float;
        epochs: int;
        factor: float;
        patience_RLR: float;
        patience_ES: float;
        min_lr: float;
        loss: str -> loss function;
        batch_size: int;
        monitor: str;
        
    return:
        history: fitted model;
    '''

    # Parameters
    optimizer = kopt.Adam(learning_rate) # nao da de mudar por keras.optimizers.kopt.Adam pois da erro. Acontece com outros modulos do Keras tbm.

    # Paths
    model_path = f'results/{model_name}/model.h5'
    csv_path = f'results/{model_name}/history.csv'
    csv_path_parameter = f'results/{model_name}/parameter.csv'

    # Convert strings to Path type
    csv_path = pathlib.Path(csv_path)
    model_path = pathlib.Path(model_path)
    csv_path_parameter = pathlib.Path(csv_path_parameter)

    # Make sure the files are saved in a folder that exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path_parameter.parent.mkdir(parents=True, exist_ok=True)

    # Callbacks
    callbacks = [
        ReduceLROnPlateau(monitor=monitor, factor=factor, patience=patience_RLR, mode='min', min_lr=min_lr),
        ModelCheckpoint(model_path, monitor=monitor, mode='auto', verbose=1, save_best_only=True),
        CSVLogger(csv_path, separator=",", append=True),
        EarlyStopping(monitor=monitor, mode='auto', verbose=1, patience=patience_ES),
    ]

    # Compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    # Training the model

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size, epochs=epochs,
        callbacks=callbacks
    )

    # Save the parameters
    if save_parameters == True:
        parameters = {
            'loss' : loss, 'optimizer' : optimizer, 'learning rate' : learning_rate,
            'epochs' : epochs, 'batch size' : batch_size, 'factor' : factor, 
            'patience RLR' : patience_RLR, 'patience ES': patience_ES, 'min LR' : min_lr
        }
        pd.DataFrame.from_dict(data=parameters, orient='index').to_csv(csv_path_parameter, header=False)

    return history
