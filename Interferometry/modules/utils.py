"""
This module contains auxiliary functions for the Interferogram class
"""

import numpy as np

def sort_list_of_tuples(list_of_tuples, sort_by_idx=0, reverse=False):
    """
    Sorts elements in a list of tuples
    ---
    Parameters
    ---
    list_of_tuples: list
        List of tuples
    sort_by_idx: int, optional
        Number of index to sort by
        E.g. if a tuple consists of two elements and we would like to sort by the second, set to 1
        Default: 0
    reverse: bool, optional
        If True, the sorting is done in ascending order.
        If False - in descending.
        Default is True
    """
    # sort by the parameter_value
    # signal_and_parameter.sort(key=operator.itemgetter(1))
    list_of_tuples.sort(key=lambda x: x[sort_by_idx], reverse=reverse)
    # split it back into sorted
    return zip(*list_of_tuples)

def get_wavelength_units(units):
    """
    Converts wavelength  units in string format to a float number
    ---
    Return
    ---
    wav_units: float
        Wavelength unit as a float number
    """
    units_dict = {"nm": 1e-9, "um": 1e-6}
    if units in units_dict:
        wav_units = units_dict[units]
    else:
        raise ValueError("Only the following units are allowed: {}".format(list(units_dict.keys())))
    return wav_units

def get_minmax_indices(wav, wav_min, wav_max, units):
    """
    Converts the min and max wavelength values to indices
    ---
    Return
    ---
    wav_min_idx: int
        Index of wav array where it is equal to the min wavelength wav_min
    wav_max_idx: int
        Index of wav array where it is equal to the min wavelength wav_max
    """
    # search for the indices whose elements are closest to specified xmin and xmax, respectively
    wav_min_idx = min(range(len(wav)), key=lambda i: abs(wav[i] - units * wav_min))
    wav_max_idx = min(range(len(wav)), key=lambda i: abs(wav[i] - units * wav_max))
    # make sure that they are sorted: wav_min_idx < wav_max_idx
    wav_min_idx, wav_max_idx = sorted([wav_min_idx, wav_max_idx], reverse=False)
    return wav_min_idx, wav_max_idx

def random_noise(mu, sigma, number_of_samples):
    """
    Random noise sequence of length equal to number_of_samples
    ---
    Parameters
    ---
    mu: float
        Mean of the normal distribution
    sigma: float
        Standard deviation of the normal distribution
    number_of_samples: int
        Number of samples to generate
    """
    return np.random.normal(mu, sigma, number_of_samples)
