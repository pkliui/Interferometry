"""
This module contains functions for normalization of data
"""


def normalize(signal_data, background_data):
    """
    Normalizes a signal by dividing it by the background
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
