"""
This module contains functions to compute the second order correlation function g2
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt
from Interferometry.modules.filtering import low_pass_filter
from Interferometry.modules.filtering import savitzky_golay_filter
import itertools as it
import os
import matplotlib.backends.backend_pdf


def compute_g2(signal_data, time_step, filter_cutoff=15e12, filter_order=3):
    """
    Computes the second order correlation function

    ---
    Args:
    ---
    signal_data: 1d ndarray
        timer series to filter
    time_step: float
        temporal step the signal_data was recorded at
    filter_cutoff: float, optional
        the cutoff frequency of the filter, in Hz
    filter_order: int, optional
        the order of the filter
    ---
    Returns:
    ---
    g2: 1d ndarray
        Second order correlation function
    """
    #
    # low-pass filter the interferogram
    signal_filtered = low_pass_filter(signal_data, time_step, filter_cutoff=filter_cutoff, filter_order=filter_order)
    # compute the g2 function
    g2 = (signal_filtered - 1)*0.5
    return g2

def g2_vs_lowpass_cutoff(signal_data, time_samples, time_step,
                         cutoff_min=1e12, cutoff_max=30e12, cutoff_step=1e12,
                         filter_order=3,
                         g2_min=0.95, g2_max=1.05,
                         cbar_min=0, cbar_max=1,
                         plotting=True, title=None,
                         ax_num=None):
    """
    Computes the second-order correlation function as a function of the filter's cut-off frequency
    ---
    Args:
    ---
    signal_data: 1d ndarray
        Signal to be filtered
    time_samples: 1d ndarray
        Time samples of the signal_data
    time_step: float
        Temporal step of the signal_data
    cutoff_min: float, optional
        The minimum cutoff frequency of the filter, in Hz
        Default is 1e12
    cutoff_max: float, optional
        The maximum cutoff frequency of the filter, in Hz
        Default is 30e12
    cutoff_step: float, optional
        The step of the cutoff frequency of the filter, in Hz
        Default is 1e12
    filter_order: int, optional
        The order of the filter
        Default is 3
    g2_min: float, optional
        If the maximum value of the computed g2 is below this value, the whole g2 distribution is set to -1
        Default is 0.95
    g2_max: float, optional
        Pixel values of the computed g2 that exceed g2_max are set to -1
        Default is 1.05
    cbar_min: float, optional
        Minimum value of the colorbar
        Default is 0
    cbar_max: float, optional
        Maximum value of the colorbar
        Default is 1
    plotting: bool, optional
        If True, the g2 distribution is plotted
        Default is True
    title: str, optional
        Title of the plot
        Default is None
    ax_num: int, optional
        The number of the axis to plot on
        Default is None
    ---
    Returns:
    ---
    g2_vs_freq: 2d ndarray
        The second order correlation function as a function of the filter's cut-off frequency
    """
    #
    # range of cutoff frequencies to test
    filter_cutoff_range = np.linspace(cutoff_min, cutoff_max, int(abs(cutoff_max - cutoff_min)/cutoff_step))
    #
    # initialise a list to keep g2 at each filter cutoff frequency
    g2_vs_freq = []
    #
    # loop over the filter cutoff frequencies
    for fc in filter_cutoff_range:
        g2 = compute_g2(signal_data, time_step, filter_cutoff=fc, filter_order=filter_order)
        #
        # threshold from below and append
        # if max g2 value is below the minimum, set all values of g2 to -1
        g2[g2.max() < g2_min] = -1
        g2_vs_freq.append(g2)
    g2_vs_freq = np.array(g2_vs_freq)
    #
    # threshold from above
    # set g2 at time delays where the maximum value of g2 is above 5 % of the max g2_max to -1
    g2_vs_freq[g2_vs_freq > g2_max] = -1
    #
    # set the max colorbar value to the max value of the g2 function
    cbar_max = (g2_vs_freq).max()
    #
    # get the average value of the g2 across the range of frequencies
    idx_thresholded = np.where(g2_vs_freq!=-1)[0]
    g2_average = np.sum(g2_vs_freq[idx_thresholded, :], axis=0)/len(idx_thresholded)
    #
    #plot
    if plotting:
        if ax_num is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax_num = ax
        im = ax_num.imshow(g2_vs_freq, aspect='auto', origin='upper',
                   cmap=plt.get_cmap("viridis"), vmin=cbar_min, vmax=cbar_max,
                   extent=(min(time_samples), max(time_samples),
                           cutoff_max, cutoff_min))
        ax_num.set_xlabel("Temporal delay, s")
        ax_num.set_ylabel("Cut-off freq., Hz")
        if title is not None:
            ax_num.set_title(title)
    else:
        im = None
    return g2_vs_freq, g2_average, im

def g2_vs_savitsky_golay(signal_data, time_shannon, time_step, time_samples,
                                 keep_shannon_sampling=True,
                                 sg_window_min=1, sg_window_max=3, sg_window_step=2,
                                 sg_order_min=1, sg_order_max=6, sg_order_step=1,
                                 bw_filter_order = 3, bw_filter_cutoff = 1e12,
                                 g2_min=0.95, g2_max=1.05,
                                 plotting=False):
    """
    Computes the second-order correlation function as a function of the filter's cut-off frequency
    ---
    Args:
    ---
    signal_data: 1d ndarray
        Signal to be filtered
    time_shannon: 1d ndarray
        Inverse of the Shannon's sampling rate (for the 2nd harmonic)
    time_step: float
        Temporal step of the signal_data
    time_samples: 1d ndarray
        Temporal samples of the signal_data
    keep_shannon_sampling: bool, optional
        If True, limits the max window size of the filter to the one set by the Shannon's sampling rate
        Default is True
    sg_window_min: int, optional
        The minimum window size of the filter, in samples
        Default is 1
    sg_window_max: int, optional
        The maximum window size of the filter, in samples
        Default is 3
    sg_window_step: int, optional
        The step of the window size of the filter, in samples
        Default is 2
    sg_order_min: int, optional
        The minimum order of the filter
        Default is 1
    sg_order_max: int, optional
        The maximum order of the filter
        Default is 6
    sg_order_step: int, optional
        The step of the order of the filter
        Default is 1
    bw_filter_order: int, optional
        The order of the Butterworth filter
        Default is 3
    bw_filter_cutoff: float, optional
        The cut-off frequency of the Butterworth filter
        Default is 1e12
    g2_min: float, optional
        The minimum value of the g2 function to threshold its distribution at
        Default is 0.95
    g2_max: float, optional
        The maximum value of the g2 function to threshold its distribution at
        Default is 1.05
    plotting: bool, optional
        If True, plots the g2 function at each filter cutoff frequency
    """
    # if it is desired to keep the filter's window size below the Shannon's sampling time,
    # set the max window size to the Shannon's sampling time
    if keep_shannon_sampling:
        sg_window_max =  int(np.ceil(time_shannon / time_step))
    # otherwise, use the provided range of window sizes
    sg_window_range = np.linspace(sg_window_min, sg_window_max,
                                  1 + int(abs(sg_window_max - sg_window_min)/sg_window_step))
    # range of orders to test
    sg_filter_order = np.linspace(sg_order_min, sg_order_max, 1 + int(abs(sg_order_max - sg_order_min)/sg_order_step))
    #
    for fo in sg_filter_order:
        #
        # initialise a list to keep g2 at each filter cutoff frequency
        g2_vs_window_range = []
        # loop over the filter cutoff frequencies
        for wr in sg_window_range:
            # filter with SG filter
            signal_data = savitzky_golay_filter(signal_data, time_shannon, time_step, window_size_shannon=wr, window_size_pxls=int(wr), order=int(fo))
            # compute the g2 function
            g2 = compute_g2(signal_data, time_step, time_samples, filter_cutoff=bw_filter_cutoff, filter_order=bw_filter_order)
            g2[g2.max() < g2_min] = -1
            g2_vs_window_range.append(g2)
        g2_vs_window_range = np.array(g2_vs_window_range)
        # threshold from above
        # set g2 at time delays where the maximum value of g2 is above g2_max to -1
        g2_vs_window_range[g2_vs_window_range > g2_max] = -1

        #plot
        if plotting:
            fig, ax = plt.subplots(1, figsize=(15, 5))
            plt.imshow(g2_vs_window_range, aspect='auto', origin='lower',
                       cmap=plt.get_cmap("viridis"), #vmin=0, vmax=1,
                       extent=(min(time_samples), max(time_samples),
                               min(sg_window_range), max(sg_window_range)))
            plt.title("g2 function")
            ax.set_xlabel("Time, s")
            ax.set_ylabel("SG filter window size")
            plt.colorbar()
            plt.show()

    return g2
