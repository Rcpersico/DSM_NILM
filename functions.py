# my_experiments.py
import h5py
import numpy as np
import pandas as pd

def load_data_for_im2seq(h5_path, appliance, building='building2', max_ratio=0.3):
    with h5py.File(h5_path, 'r') as f:
        mains = f[f'{building}/mains/power/active'][:]
        target = f[f'{building}/{appliance}/power/active'][:]

    min_len = min(len(mains), len(target))
    mains = mains[:min_len]
    target = target[:min_len]

    if max_ratio < 1.0:
        trim_len = int(min_len * max_ratio)
        mains = mains[:trim_len]
        target = target[:trim_len]

    return pd.DataFrame(mains, columns=['mains']), pd.DataFrame(target, columns=[appliance])
