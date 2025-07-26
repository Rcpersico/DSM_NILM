from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, GlobalAveragePooling1D, Input, Reshape, Conv2D
import tensorflow as tf


class CNN_1D:

    def build_model(input_Length, filters1_count = 32, filters2_count=64, kernel1_size = 7, kernel2_size = 5, dense_neuron_count = 128,output_Length = None):
        #Here the input length is the amount of timesteps the model looks at to predict

        #if the output length is not defined set it to the input length
        if output_Length is None:
            output_Length = input_Length

        #Model Architecture 
        model = Sequential()
        model.add(Conv1D(filters=filters1_count, kernel_size=kernel1_size, activation='relu', input_shape=(input_Length, 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=filters2_count, kernel_size=kernel2_size, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(dense_neuron_count, activation='relu'))
        model.add(Dense(output_Length, activation='linear'))  # Need to test out a linear activation vs RELUw

        #Model Compilation:
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        return model 
    
    @staticmethod
    def build_s2p_model(window_len) -> tf.keras.Model:

        inputs = Input(shape=(window_len,), name="window_flat")

        # add the “image” dimension so Conv2D can work
        x = Reshape((1, window_len, 1), name="to_2d")(inputs)

        # five Conv2D blocks with identical specs to the reference
        x = Conv2D(30, (10, 1), strides=(1, 1), padding="same",
                   activation="relu", name="conv_1")(x)
        x = Conv2D(30, (8, 1),  strides=(1, 1), padding="same",
                   activation="relu", name="conv_2")(x)
        x = Conv2D(40, (6, 1),  strides=(1, 1), padding="same",
                   activation="relu", name="conv_3")(x)
        x = Conv2D(50, (5, 1),  strides=(1, 1), padding="same",
                   activation="relu", name="conv_4")(x)
        x = Conv2D(50, (5, 1),  strides=(1, 1), padding="same",
                   activation="relu", name="conv_5")(x)

        x = Flatten(name="flatten")(x)
        x = Dropout(0.2, name="dropout")(x)
        x = Dense(1024, activation="relu", name="dense_1024")(x)
        outputs = Dense(1, activation="linear", name="power")(x)

        model = Model(inputs, outputs, name=f"S2P_exact_{window_len}")
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    def build_model2(input_Length, filters1_count = 64, filters2_count=128, kernel1_size = 7, kernel2_size = 5, dense_neuron_count = 128,output_Length = None):
        #Here the input length is the amount of timesteps the model looks at to predict

        #if the output length is not defined set it to the input length
        if output_Length is None:
            output_Length = input_Length

        #Model Architecture 
        model = Sequential()
        model.add(Conv1D(filters = filters1_count, kernel_size = kernel1_size,  activation='relu', padding = 'same', input_shape=(input_Length, 1) ))
        model.add(Conv1D(filters=filters1_count, kernel_size=kernel1_size, activation='relu', padding = 'same'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=filters2_count, kernel_size=kernel2_size, activation='relu', padding = 'same'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(dense_neuron_count, activation='relu'))
        model.add(Dense(output_Length, activation='linear'))  # Need to test out a linear activation vs RELUw

        #Model Compilation:
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        return model 
    

    