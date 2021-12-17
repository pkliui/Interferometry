"""
This module contains functions to deal with sampling of interferometric data
"""

import numpy as np

def get_time_step(time_samples):
    """
    Get a step from experimental time samples
    It is assumed that the samples are equally spaced!
    """
    return np.abs(time_samples[1] - time_samples[0])

def get_time_units(units):
    """
    Converts temporal units in string format to a float number
    ---
    Args:
    ---
    units: str
        temporal units in string format
    ---
    Return
    ---
    tau_units: float
        Time unit as a float number
    """
    units_dict = {"ps": 1e-12, "fs": 1e-15, "as": 1e-18}
    if units in units_dict:
        tau_units = units_dict[units]
    else:
        raise ValueError("Only the following units are allowed: {}".format(list(units_dict.keys())))
    return tau_units

def zoom_in_2d(signal, sampling_variable, zoom_in_value):
    """
    crop the signal distribution to the sampling variable's range of interest
    ---
    Args:
    ---
    signal: 2dndarray
        2d signal distribution
    sampling_variable: 1darray
        sampling variable (e.g. frequency samples)
    zoom_in_value: float
        zoom in value, within the sampling_variable range
    ---
    Return
    ---
    signal_zoomed: 2dndarray
        cropped signal distribution
    sampling_variable_zoomed: 1darray
        cropped sampling variable
    """
    if zoom_in_value is not None and zoom_in_value < sampling_variable[-1]:
        # compute the sampling step
        d_sampling_variable = np.abs(sampling_variable[1] - sampling_variable[0])
        # get the index of the zoom in value
        cutoff_idx = int(len(sampling_variable)/2 - zoom_in_value / d_sampling_variable)
        # zoom in to the signal and sampling variable
        signal_zoomed = np.copy(signal[cutoff_idx:-cutoff_idx, :])
        sampling_variable_zoomed = np.copy(sampling_variable[cutoff_idx:-cutoff_idx])
    elif zoom_in_value is None:
        signal_zoomed = np.copy(signal)
        sampling_variable_zoomed = np.copy(sampling_variable)
    else:
        raise ValueError('zoom_in_value cannot be larger than the max value of the sampling variable!')
    return signal_zoomed, sampling_variable_zoomed
