# Python packages
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.utils.class_weight import compute_sample_weight

# Train the model
def training(model, X_train, y_train, X_val, y_val, model_name):

    # Parameters
    loss = 'binary_crossentropy'
    learning_rate = 0.01
    optimizer = Adam(learning_rate)
    batch_size = 16
    epochs = 100
    monitor = 'val_loss'

    # Paths
    filepath='C:/Users/sarah/TCC/ECG Classification Models/Weights improvement/weights-improvement-{epoch:02d}-{val_loss:.2f}.h5'
    filename='results/ribeiro_100_test10.csv'

    # Get sample weights
    sample_weights_train = compute_sample_weight(class_weight='balanced',y=y_train)

    # Callbacks
    callbacks = [
        ReduceLROnPlateau(monitor=monitor, factor=0.1, patience=10, mode='min', min_lr=1e-6),
        # EarlyStopping(monitor=monitor, mode='auto', verbose=1, patience=11),
        # ModelCheckpoint(filepath, monitor=monitor, mode='auto', verbose=1, save_best_only=True),
        # CSVLogger(filename, separator=",", append=True)
    ]

    # Compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    # Training the model
    history = model.fit(
        X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
        callbacks=callbacks,sample_weight=sample_weights_train
    )

    # Save the model
    # model.save('./model_{model_name}.hdf5')

    return history

