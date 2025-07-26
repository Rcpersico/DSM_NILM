
import h5py
import pandas as pd

#data loader for multiple appliances
def load_data_for_im2seq(h5_path, appliances, building='building2', max_ratio=1.0):
    with h5py.File(h5_path, 'r') as f:
        mains = f[f'{building}/mains/power/active'][:]
        app_dfs = {}
        min_len = len(mains)
        # Find minimum length across all appliances
        for app in appliances:
            target = f[f'{building}/{app}/power/active'][:]
            min_len = min(min_len, len(target))
        mains = mains[:min_len]
        if max_ratio < 1.0:
            cut = int(min_len * max_ratio)
            mains = mains[:cut]
            min_len = cut
        mains_df = pd.DataFrame(mains, columns=['mains'])
        # Build DataFrames for each appliance
        for app in appliances:
            target = f[f'{building}/{app}/power/active'][:min_len]
            if max_ratio < 1.0:
                target = target[:cut]
            app_dfs[app] = pd.DataFrame(target, columns=[app])
    return mains_df, app_dfs




def align(pred_series, true_series, half):
    pred_aligned = pred_series[half : -half]        # shift right by L//2
    true_aligned = true_series[:len(pred_aligned)]  # now same length
    return pred_aligned, true_aligned