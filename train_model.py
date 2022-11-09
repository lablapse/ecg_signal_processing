# Python packages
import keras.optimizers as kopt
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.utils.class_weight import compute_sample_weight

import pathlib

# Train the model


def training(
    model,
    X_train, y_train,
    X_val, y_val,
    model_name,
    learning_rate=0.1,
    epochs=100,
    ):

    # Parameters
    loss = 'binary_crossentropy'
    # optimizer = kopt.Adam(learning_rate)
    optimizer = kopt.SGD(learning_rate, momentum=0.9)
    # batch_size = 16
    batch_size = 256
    # batch_size = 256+128
    monitor = 'val_loss'

    # Paths
    # model_path = 'saved_models/weights-improvement-{epoch:02d}-{val_loss:.2f}.h5'
    model_path = f'results/{model_name}/model.h5'
    csv_path = f'results/{model_name}/history.csv'

    # Convert strings to Path type
    csv_path = pathlib.Path(csv_path)
    model_path = pathlib.Path(model_path)

    # Make sure the files are saved in a folder that exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # # Get sample weights
    # sample_weights_train = compute_sample_weight(
    #     class_weight='balanced', y=y_train)

    # Callbacks
    callbacks = [
        ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=10, mode='min', min_lr=1e-6),
        # ReduceLROnPlateau(monitor=monitor, factor=0.1,patience=7, mode='auto', min_lr=learning_rate / 100),
        ModelCheckpoint(model_path, monitor=monitor, mode='auto', verbose=1, save_best_only=True),
        CSVLogger(csv_path, separator=",", append=True),
        EarlyStopping(monitor=monitor, mode='auto', verbose=1, patience=15),
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

    # Save the model
    # model.save('./model_{model_name}.h5')

    return history
