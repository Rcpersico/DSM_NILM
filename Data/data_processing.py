import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



def normalize(data, columns):
    scalers = {}
    normalized_data = pd.DataFrame()

    for col in columns:
        scaler = MinMaxScaler()
        normalized_data[col] = scaler.fit_transform(data[[col]]).flatten()
        scalers[col] = scaler

    return normalized_data, scalers


def prepareData(data ,sequence_length):

    data = np.array(data)

    # Trim to nearest full sequence
    trimmed_len = (len(data) // sequence_length) * sequence_length
    trimmed_data = data[:trimmed_len]

    # Reshape to 3D for Conv1D: (samples, time steps, features)
    reshaped_data = trimmed_data.reshape(-1, sequence_length, 1)

    return reshaped_data


def prepare_s2p_data(inputData, outputData, sequence_length, stride=5):
    X, y = [], []
    half = sequence_length // 2
    for i in range(half, len(inputData) - half, stride):
        window = inputData[i - half : i + half + 1]
        X.append(window)
        y.append(outputData[i])
    return np.array(X), np.array(y)





def prepare_balanced_s2p_data(
    inputData,
    outputData,
    sequence_length,
    stride=1,
    on_threshold=10,
    off_keep_prob=0.1,
    normalize=True
):

    X, y = [], []
    half = sequence_length // 2

    # Normalize if needed
    input_mean, input_std = inputData.mean(), inputData.std()
    output_max = outputData.max()

    if normalize:
        inputData = (inputData - input_mean) / input_std
        outputData = outputData / output_max

    for i in range(half, len(inputData) - half, stride):
        center_val = outputData[i]
        is_on = center_val > (on_threshold / output_max if normalize else on_threshold)

        if is_on or np.random.rand() < off_keep_prob:
            window = inputData[i - half: i + half + 1]
            if len(window) == sequence_length:  # Skip incomplete edge cases
                X.append(window)
                y.append(center_val)

    X = np.array(X).reshape(-1, sequence_length, 1)
    y = np.array(y)

    stats = {
        "input_mean": input_mean,
        "input_std": input_std,
        "output_max": output_max,
        "normalize": normalize
    }

    return X, y, stats
