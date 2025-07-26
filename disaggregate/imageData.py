import os
import numpy as np
import pandas as pd
import h5py
from pyts.image import GramianAngularField

import os

# This resolves the path relative to the file's location, NOT where you run the script from
script_dir = os.path.dirname(os.path.abspath(__file__))
h5_path = os.path.join(script_dir, '..', '..', 'src', 'data', 'REFIT_House2.h5')


# Existing loader
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

    return np.array(mains), np.array(target)

# Sliding window
def sliding_window(data, seq_len, stride=1):
    return np.array([
        data[i:i + seq_len]
        for i in range(0, len(data) - seq_len + 1, stride)
    ])

# --- Main usage ---
appliance = 'fridge_freezer'
sequence_length = 200
stride = 1

# Define output directory
output_dir = 'imageData'
os.makedirs(output_dir, exist_ok=True)

# Load and pad
mains, target = load_data_for_im2seq(h5_path, appliance, max_ratio=0.3)
pad = sequence_length // 2
mains_padded = np.pad(mains, (pad, pad), mode='constant')
target_padded = np.pad(target, (pad, pad), mode='constant')

# Sliding windows
X_seq = sliding_window(mains_padded, sequence_length, stride)
y_seq = sliding_window(target_padded, sequence_length, stride)

# Normalize
mains_mean = X_seq.mean()
mains_std = X_seq.std()
X_seq_norm = (X_seq - mains_mean) / mains_std

appl_mean = y_seq.mean()
appl_std = y_seq.std()
y_seq_norm = (y_seq - appl_mean) / appl_std

# Convert to GASF
gasf = GramianAngularField(image_size=sequence_length, method='summation')
X_imgs = gasf.fit_transform(X_seq_norm)
X_imgs = X_imgs[..., np.newaxis]

# File prefix
prefix = f"{appliance}_{sequence_length}"

# Save files
np.save(os.path.join(output_dir, f'{prefix}_X_gasf.npy'), X_imgs)
np.save(os.path.join(output_dir, f'{prefix}_Y_seq.npy'), y_seq_norm)
np.savez(os.path.join(output_dir, f'{prefix}_norm_params.npz'),
         mains_mean=mains_mean, mains_std=mains_std,
         appl_mean=appl_mean, appl_std=appl_std)



print(f"✅ Saved preprocessed data to folder '{output_dir}' as:")
print(f"  {prefix}_X_gasf.npy")
print(f"  {prefix}_Y_seq.npy")
print(f"  {prefix}_norm_params.npz")
