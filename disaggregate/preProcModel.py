import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape, Conv1D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint


def train_from_precomputed(X_path, y_path, sequence_length=200, batch_size=512, n_epochs=15):
    # Load data
    X = np.load(X_path)  # shape: (N, 200, 200, 1)
    y = np.load(y_path)  # shape: (N, 200)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    # Model
    model = Sequential()
    model.add(Conv2D(8, 4, strides=2, activation='relu', input_shape=(sequence_length, sequence_length, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(sequence_length * 8, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(sequence_length * 8, activation='relu'))
    model.add(Reshape((sequence_length, 8)))
    model.add(Conv1D(1, 4, padding='same', activation='linear', dtype='float32'))  # for mixed precision stability

    # Compile
    model.compile(
        loss='mse',
        optimizer=SGD(learning_rate=1e-2, momentum=0.8),
        metrics=['mae']
    )

    # Train
    checkpoint = ModelCheckpoint('trained_model.keras', monitor='val_loss', save_best_only=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=n_epochs,
        callbacks=[checkpoint],
        shuffle=True
    )

    return model
