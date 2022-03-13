"""
This module contains centered 1D Fourier transforms
"""

import numpy as np


def ft_data(intensity, time_step, time_samples):
    """
    Computes the Fourier transform of an input sequence
    and the corresponding frequency samples, given the signal_data intensity samples, temporal samples and a discretization step
    ---
    Parameters
    ---
    intensity: numpy 1D array
        Signal intensity samples
    time_samples: numpy 1D array
        Time samples
        Assumed to be equally sampled
        Default is None
    time_step: float
        Discretization step at which the time samples were recorded
        Default is None
    ---
    Return
    ---
    ft: 1d numpy array
        Only positive frequencies of the Fourier transformed sequence
        Excludes the zeroth frequency
    freq: 1d numpy array
        Corresponding frequency samples
        Excludes zeroth frequency
    """
    #
    # begin from 1st element to avoid displaying the zero-th freq. component
    ft = np.fft.rfft(intensity)[1:]
    freq = np.fft.rfftfreq(len(time_samples), time_step)[1:]
    return ft, freq

def ift_data(ft_data):
    """
    Computes the inverse Fourier transform of an input sequence
    and the corresponding frequency samples, given the signal_data intensity samples, temporal samples and a discretization step
    ---
    Parameters
    ---
    ft_data: numpy 1D array
        Signal FT distribution samples
    ---
    Return
    ---
    ift: 1d numpy array
        Only positive frequencies of the inverse Fourier transformed sequence
        Excludes the zeroth frequency
    """
    #
    # begin from 1st element to avoid displaying the zero-th freq. component
    ift = np.fft.irfft(ft_data)[1:]
    return ift
