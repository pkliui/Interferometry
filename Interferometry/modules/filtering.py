"""
This module contains functions for spectral filtering
"""
from scipy.signal import filtfilt, butter, savgol_filter
import numpy as np


def low_pass_filter(signal, time_step, filter_cutoff=30e12, filter_order=6):
    """
    Filters an interferogram by using a butterworth filter
    ---
    Args:
    ---
    signal: 1d ndarray
        timer series to filter
    time_step:
        temporal step the signal was recorded at
    filter_cutoff: float, optional
        the cutoff frequency of the filter, in Hz
    filter_order: int, optional
        the order of the filter
    ---
    Returns:
    ---
    signal_filtered: 1d ndarray
        filtered signal
    ft_freq: 1d ndarray
        frequency samples of the filtered signal
    """
    #
    # compute the filter's cutoff frequency
    freq_critical = filter_cutoff * 2 * time_step
    #
    # filter the signal
    b, a = butter(filter_order, freq_critical, btype='lowpass')
    signal_filtered = filtfilt(b, a, signal)
    #
    return signal_filtered

def savitzky_golay_filter(signal_data, time_shannon, time_step, window_size_shannon=1, window_size_pxls=None, order=2):
    """
    Applies a Savitzky-Golay filter to the interferogram
    ---
    Args:
    ---
    signal_data: 1d ndarray
        the interferogram to filter
    time_shannon: float
        Shannon's critical sampling interval, in seconds
    time_step: float
        the temporal step of the interferogram, in seconds
    order : int, optional
        Order of the polynomial used in the filter
        Must be less than window_size
    window_size_shannon: float, optional
        Size of the window in units of the Shannon's sampling time
        Must be between 0 and 1
        Default: 1
    window_size_pxls: int, optional
        Size of the window in pixels
        If None, the window size is computed from the Shannon's sampling time and the time step
        Default: None
    ---
    Returns:
    ---
    filtered_signal: 1d ndarray
        the filtered signal
    """
    if window_size_pxls is not None:
        window_size = window_size_pxls
    else:
        window_size = int(np.ceil(window_size_shannon * time_shannon / time_step))
    filtered_signal = savgol_filter(signal_data, window_size, order)
    return filtered_signal


