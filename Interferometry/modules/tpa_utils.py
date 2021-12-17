"""
This modules contains functions for extraction and processing of
two-photon absorption (TPA) signal from a specttrogram
"""

import numpy as np
import itertools as it

def closest_indicies(tpa_freq, freq_samples, freq_window_size):
    """
    Returns the indicies of the frequencies separated from the two-photon
    absorption frequency by the window_size
    ---
    Args
    ---
    tpa_freq: float
        The TPA frequency of the signal we would like to get the closest indicies to, in Hz
    freq_samples: numpy array
        The frequencies of the spectrogram
    freq_window_size: int
        The distance between the frequency of interest and the closest indicies we are looking for, in pixels
    """
    # get indicies of the frequencies closest to the tpa_freq
    tpa_idx = np.where((abs(tpa_freq - freq_samples) < abs(freq_samples[1] - freq_samples[0])))[0][0]
    tpa_idx_low = int(tpa_idx - freq_window_size)
    tpa_idx_high = int(tpa_idx + freq_window_size)

    return tpa_idx_low, tpa_idx_high


def wigner_ville_distribution_tpa(signal_wvd, tpa_idx_low, tpa_idx_high):
    """
    Returns the Wigner-Ville distribution of the signal at the two-photon absorption frequency
    ---
    Args
    ---
    signal_wvd: numpy array
        The Wigner-Ville distribution of the signal
    tpa_idx_low: int
        The lower index of the frequency range we are interested in
    tpa_idx_high: int
        The higher index of the frequency range we are interested in
    ---
    Returns
    ---
    signal_wvd_tpa: numpy array
        The Wigner-Ville distribution of the signal at the two-photon absorption frequency
    """
    # get the WVD at the TPA frequency (withing the range in the vicinity of the frequency)
    signal_wvd_tpa = np.zeros(signal_wvd.shape)
    signal_wvd_tpa[tpa_idx_low:tpa_idx_high, :] = np.copy(signal_wvd[tpa_idx_low:tpa_idx_high, :])

    signal_wvd_tpa = signal_wvd_tpa.sum(axis=0)
    signal_wvd_tpa = signal_wvd_tpa / signal_wvd_tpa.max()

    return signal_wvd_tpa

def loosely_thresholded_tpa_signal(signal_wvd_tpa, tpa_thresh):
    """
    Computes a binary mask of the TPA signal set by the laser pulse duration through the TPA threshold.
    The mask takes values of 1 when the signal is above the threshold and 0 when it is below.
    ---
    Args
    ---
    signal_wvd_tpa: numpy array
        The Wigner-Ville distribution of the signal at the two-photon absorption frequency
    tpa_thresh: float
        The simulated threshold of the TPA signal set by the laser pulse duration
    ---
    Returns
    ---
    tpa_signal_loose: numpy array
        The binary mask of the TPA signal set by the laser pulse duration through the TPA threshold
    max_idx: int
        The index of the maximum value in the input Wigner-Ville distribution
    """
    # # set the mask distribution to 1 within the FWHM of the laser pulse and to 0 elsewhere
    # # use a threshold from the simulation to set the FWHM of the pulse
    tpa_signal_loose = np.zeros(signal_wvd_tpa.shape)
    tpa_signal_loose[np.where(signal_wvd_tpa > tpa_thresh)] = 1

    #  get the index of the interferogram's maximum
    max_idx = np.argmax(signal_wvd_tpa)

    return tpa_signal_loose, max_idx

def tightly_thresholded_tpa_signal(tpa_signal_loose, max_idx, time_step, tpa_tolerance = 2e-15):
    """
    Returns a binary mask of the TPA signal set by the laser pulse duration through the TPA threshold.
    The mask takes values of 1 when the signal is above the threshold and 0 when it is below.
    Computes the tight support of the TPA signal by checking if those consecutive sub-regions of
    the input loose mask that have pixel values of 1 are spaced not too far apart
    ---
    Args:
    ---
    tpa_signal_loose: 2d ndarray
        Loose TPA mask
    max_idx: int
        Index of the maximum value of the TPA signal
    time_step: float
        Time step of the TPA signal
    tpa_subregions_distance: float
        Distance between the subregions of the TPA
    tpa_tolerance: float
        Tolerance for the TPA
    """
    # set the tight mask distribution corresponding to the TPA region
    tpa_signal_tight = np.zeros(tpa_signal_loose.shape)
    # set the mask's value to 1 at the interferogram's maximum value
    tpa_signal_tight[max_idx:max_idx+2] = 1
    tpa_signal_loose[max_idx:max_idx+2] = 1
    # initialise the value for the distance between non-zero-valued subregions of the tight mask
    tpa_subregions_distance = 0
    jump_to_next_subregion = 0
    #
    # explore the left and then the right side of the TPA signal
    for i in it.chain(range(max_idx-1, 0, -1), range(max_idx+1, len(tpa_signal_tight))):
        #print("i", i)
        # if we are below the tpa_tolerance, reassign the pixel values from loose mask to tight mask
        if tpa_signal_loose[i] == 1 and tpa_subregions_distance < tpa_tolerance:
            tpa_signal_tight[i] = 1
            # reset the sub-regions distance
            tpa_subregions_distance = 0
        elif tpa_signal_loose[i] == 0 and tpa_subregions_distance < tpa_tolerance:
            tpa_signal_tight[i] = 0
            # increment the sub-regions distance
            tpa_subregions_distance += time_step
        # in case, tpa_subregions_distance >= tpa_tolerance,
        else:
            tpa_signal_tight[i] = 0
            if i == 1 or i == len(tpa_signal_tight)-1:
                tpa_subregions_distance = 0

    return tpa_signal_tight

def tight_support_tpa(tpa_signal_tight):
    """
    Returns the tight support of the TPA signal
    ---
    Args:
    ---
    tpa_signal_tight: 2d ndarray
        Tightly thresholded TPA signal
    ---
    Returns:
    ---
    tpa_support: 2d ndarray
        Tight TPA support
    """
    idx_left = np.where(tpa_signal_tight == 1)[0][0]
    idx_right = np.where(tpa_signal_tight == 1)[0][-1]
    tight_support = np.zeros(tpa_signal_tight.shape)
    tight_support[idx_left:idx_right+1] = 1

    return tight_support

def tight_support_tpa_simulation(time_samples, pulse_duration, signal_wvd_tpa):
    """
    Returns the tight support of the simulated TPA signal and the corresponding threshold value
    ---
    Args:
    ---
    time_samples: numpy array
        Time samples of the TPA signal
    pulse_duration: float
        Duration of the laser pulse
    signal_wvd_tpa: numpy array
        Wigner-Ville distribution of the TPA signal
    ---
    Returns:
    ---
    tight_support: 2d ndarray
        Tight TPA support
    tpa_thresh: float
        TPA threshold
    """
    # get the indices of time samples within the FWHM of the laser pulse
    idx_t_fwhm_left = np.where(np.abs(time_samples+pulse_duration/2) < abs(time_samples[1] - time_samples[0]))[0][0]
    idx_t_fwhm_right = np.where(np.abs(time_samples-pulse_duration/2) < abs(time_samples[1] - time_samples[0]))[0][0]

    # set the mask distribution to 1 within the FWHM of the laser pulse and to 0 elsewhere
    # this is the region where the definition of the TPA is always valid
    tight_support = np.zeros(signal_wvd_tpa.shape)
    tight_support[idx_t_fwhm_left:idx_t_fwhm_right] = 1

    # get the corresponding threshold value of the TPA signal
    tpa_thresh = (signal_wvd_tpa[idx_t_fwhm_left] + signal_wvd_tpa[idx_t_fwhm_right])/2


    return tight_support, tpa_thresh