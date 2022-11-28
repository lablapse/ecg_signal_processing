# Python packages
import keras.optimizers as kopt
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.utils.class_weight import compute_sample_weight
import pandas as pd

import pathlib

# Train the model


def training(
    model,
    X_train, y_train,
    X_val, y_val,
    model_name,
    save_parameters,
    learning_rate=0.1,
    epochs=100,
    ):

    # Parameters
    loss = 'binary_crossentropy'
    optimizer = kopt.Adam(learning_rate)
    # optimizer = kopt.SGD(learning_rate, momentum=0.9)
    # batch_size = 16
    batch_size = 256
    # batch_size = 64
    # batch_size = 256+128
    monitor = 'val_loss'
    # Callbacks parameters
    factor = 0.5
    patience_RLR = 10
    patience_ES = 15
    min_lr = 1e-6

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

    # # Get sample weights
    # sample_weights_train = compute_sample_weight(
        # class_weight='balanced', y=y_train)

    # Callbacks
    callbacks = [
        ReduceLROnPlateau(monitor=monitor, factor=factor, patience=patience_RLR, mode='min', min_lr=min_lr),
        # ReduceLROnPlateau(monitor=monitor, factor=factor,patience=patience_RLR, mode='auto', min_lr=learning_rate / 100),
        ModelCheckpoint(model_path, monitor=monitor, mode='auto', verbose=1, save_best_only=True),
        CSVLogger(csv_path, separator=",", append=True),
        EarlyStopping(monitor=monitor, mode='auto', verbose=1, patience=patience_ES),
        # EarlyStopping(monitor=monitor, mode='auto', verbose=1, patience=9, min_delta=0.00001),
    ]

    # Compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    # Training the model
    print('Batch Size:', batch_size)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size, epochs=epochs,
        callbacks=callbacks,
        # sample_weight=sample_weights_train
    )

    # Save the parameters
    parameters = { 
        'loss' : loss, 'optimizer' : optimizer, 'learning rate' : learning_rate,
        'epochs' : epochs, 'batch size' : batch_size, 'factor' : factor, 
        'patience RLR' : patience_RLR, 'patience ES': patience_ES, 'min LR' : min_lr
    }
    if save_parameters == True:
        pd.DataFrame.from_dict(data=parameters, orient='index').to_csv(csv_path_parameter, header=False)

    # Save the model
    # model.save('./model_{model_name}.h5')

    return history
