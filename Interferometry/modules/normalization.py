"""
This module contains functions for normalization of data
"""
import numpy as np


def normalize(signal_data, time_step, time_samples, normalizing_width=10e-15, t_norm_start=None):
    """
    Normalizes an interferogram by subtracting the mean value of the signal_data and normalizing it to have a 1 : 8 ratio
    ---
    Args:
    ---
    signal_data: 1d ndarray
        timer series to normalize
    time_step: float
        temporal step the signal_data was recorded at
    time_samples: 1d ndarray
        time samples of the signal_data
    normalizing_width: float, optional
        the width of integration range to be used for signals' normalization, in seconds
    t_norm_start: float
        the start time of the normalization window, in seconds
        Default is None
    ---
    Returns:
    ---
    signal_norm: 1d ndarray
        normalized signal_data
    """
    if t_norm_start is not None:
        #
        # set integration range for normalization
        idx_norm_range = int(normalizing_width / time_step)
        idx_norm_start = int(np.argmin(np.abs(t_norm_start - time_samples)))
        #
        # compute the mean value of the signal_data's background for given integration range
        # and normalize the signal_data to have 1:8 ratio
        signal_mean_bg = np.mean(np.abs(np.array(signal_data[idx_norm_start : idx_norm_range + idx_norm_start])))
        signal_data -= signal_mean_bg
        signal_data -= signal_data.min()
        signal_norm = 8 * signal_data / signal_data.max()
    else:
        raise ValueError("starting value t_norm_start cannot be none! ")
    return signal_norm