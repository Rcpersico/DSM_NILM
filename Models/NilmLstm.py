from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Reshape
import tensorflow as tf



def build_lstm1(input_length):
    model = Sequential()
    model.add(Input(shape=(input_length, 1)))
    model.add(LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model





def build_lstm2(input_length):
    model = Sequential()
    model.add(Input(ser=(input_length, 1)))
    model.add(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    model.add(LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model




def build_lstm3(input_length):
    model = Sequential()
    model.add(Input(shape=(input_length, 1)))
    model.add(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    model.add(LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))
    model.add(LSTM(16, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model



de
