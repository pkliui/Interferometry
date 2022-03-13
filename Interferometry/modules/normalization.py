"""
This module contains functions for normalization of data
"""
import numpy as np
from matplotlib import pyplot as plt

def normalize_by_background(signal_data, background_data):
    """
    Normalizes a signal by dividing it by a background
    ---
    Args:
    ---
    signal_data: 1d ndarray
        interferogram signal to normalize
    one_arm_data: 1d ndarray
        signal recorded at one arm of interferometer
    ---
    Returns:
    ---
    signal_norm: 1d ndarray
        normalized signal_data
    """
    signal_norm = signal_data / background_data
    return signal_norm

def normalize_by_value_at_infinity(signal_data, time_step, time_samples, normalizing_width=None, t_norm_start=None):
    """
    Normalizes a signal by dividing it by its value at infinity
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
    if t_norm_start is not None and normalizing_width is not None:
        #
        # set integration  width for normalization
        idx_norm_width = int(normalizing_width / time_step)
        # start value
        idx_norm_start = int(np.argmin(np.abs(time_samples - t_norm_start)))
        #
        # compute the mean value of the signal_data at infinity
        signal_mean_infinity = np.mean(np.abs(np.array(signal_data[idx_norm_start : idx_norm_width + idx_norm_start])))
        signal_data /= signal_mean_infinity
    else:
        raise ValueError("starting value t_norm_start cannot be none! ")
    return signal_data

def rescale_1_to_n(signal_data, time_step, time_samples, normalizing_width=10e-15, t_norm_start=None):
    """
    Rescales an interferogram to 1 : N base-level-to-total-height ratio
    so that the base level is at 1.
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
        signal_data -= signal_data.min()
        signal_mean_bg = np.mean(np.abs(np.array(signal_data[idx_norm_start : idx_norm_range + idx_norm_start])))
        signal_data /= signal_mean_bg

    else:
        raise ValueError("starting value t_norm_start cannot be none! ")
    return signal_data