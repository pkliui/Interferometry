"""
This module contains functions to compute the second order correlation function g2
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt
from Interferometry.modules.filtering import low_pass_filter
from Interferometry.modules.filtering import savitzky_golay_filter


def compute_g2(signal_data, time_step, time_samples, filter_cutoff=30e12, filter_order=6, plotting=True):
    """
    Computes the second order correlation function

    ---
    Args:
    ---
    signal_data: 1d ndarray
        timer series to filter
    time_step: float
        temporal step the signal_data was recorded at
    time_samples: 1d ndarray
        time samples of the signal_data
    filter_cutoff: float, optional
        the cutoff frequency of the filter, in Hz
    filter_order: int, optional
        the order of the filter
    plotting: bool, optional
        whether to plot the g2 function
        Default: True
    ---
    Returns:
    ---
    g2: 1d ndarray
        Second order correlation function
    """
    #
    # low-pass filter the input signal_data
    #
    signal_filtered = low_pass_filter(signal_data, time_step, filter_cutoff=filter_cutoff, filter_order=filter_order)
    # Subtract the background and divide by 2 to get the correlation function
    g2 = (signal_filtered - 1) / 2
    #
    if plotting:
        fig, ax = plt.subplots(1, figsize=(15, 5))
        ax.plot(time_samples, g2)
        ax.set_xlabel("Time, s")
        plt.title("g2")
        plt.grid()
        plt.show()

    return g2


def g2_vs_low_pass_cutoff(signal_data, time_samples, time_step,
                         cutoff_min=1e12, cutoff_max=30e12, cutoff_step=1e12,
                         order_min=1, order_max=6, order_step=1,
                         g2_min=0.95, g2_max=1.05,
                         to_plot=True):
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
    order_min: int, optional
        The minimum order of the filter, Default is 1
    order_max: int, optional
        The maximum order of the filter, Default is 6
    order_step: int, optional
        The step of the order of the filter, Default is 1
    g2_min: float, optional
        If the maximum value of the computed g2 is below this value, the whole g2 distribution is set to -1
        Default is 0.95
    g2_max: float, optional
        Pixel values of the computed g2 that exceed g2_max are set to -1
        Default is 1.05
    to_plot: bool, optional
        If True, the g2 distribution is plotted
    ---
    Returns:
    ---
    g2_vs_freq: 2d ndarray
        The second order correlation function as a function of the filter's cut-off frequency
    """
    #
    # range of cutoff frequencies to test
    filter_cutoff_range = np.linspace(cutoff_min, cutoff_max, int(abs(cutoff_max - cutoff_min)/cutoff_step))
    # range of orders to test
    filter_order = np.linspace(order_min, order_max, 1 + int(abs(order_max - order_min)/order_step))
    #
    for fo in filter_order:
        #
        # initialise a list to keep g2 at each filter cutoff frequency
        g2_vs_freq = []
        # loop over the filter cutoff frequencies
        for fc in filter_cutoff_range:
            g2 = compute_g2(signal_data, time_step, filter_cutoff=fc, filter_order=fo)
            #
            # threshold from below and append
            # if max g2 value is below the minimum, set all values of g2 to -1
            g2[g2.max()<g2_min] = -1
            g2_vs_freq.append(g2)
        g2_vs_freq = np.array(g2_vs_freq)
        # threshold from above
        # set g2 at time delays where the maximum value of g2 is above g2_max to -1
        g2_vs_freq[g2_vs_freq>g2_max] = -1
        #
        #plot
        if to_plot:
            fig, ax = plt.subplots(1, figsize=(15, 5))
            plt.imshow(g2_vs_freq, aspect='auto', origin='lower',
                       cmap=plt.get_cmap("viridis"), vmin=0, vmax=1,
                       extent=(min(time_samples), max(time_samples),
                               min(filter_cutoff_range), max(filter_cutoff_range)))
            plt.title("g2 function, filter order = {}".format(fo))
            ax.set_xlabel("Time, s")
            ax.set_ylabel("Cut-off frequency, Hz")
            #ax.invert_yaxis()
            plt.colorbar()
            plt.show()

    return g2_vs_freq

def g2_vs_savitsky_golay(signal_data, time_shannon, time_step, time_samples,
                                 keep_shannon_sampling=True,
                                 sg_window_min=1, sg_window_max=3, sg_window_step=2,
                                 sg_order_min=1, sg_order_max=6, sg_order_step=1,
                                 bw_filter_order = 3, bw_filter_cutoff = 1e12,
                                 g2_min=0.95, g2_max=1.05,
                                 to_plot=True):
    """
    Computes the second-order correlation function as a function of the filter's cut-off frequency
    !!! CHECK THE ARG IN SAV GOLAY FILTER _ THERE IS A NEW WINDOW SIZE IN PXLS AND WINDOW SIZE IN SHANNON UNITS!!! NOT THE SAME HERE IN COMPUTE G"....`
    ---
    Args:
    ---
    """
    #
    # range of cutoff frequencies to test
    sg_window_range = np.linspace(sg_window_min, sg_window_max,
                                  1 + int(abs(sg_window_max - sg_window_min)/sg_window_step))
    # range of orders to test
    sg_filter_order = np.linspace(sg_order_min, sg_order_max, 1 + int(abs(sg_order_max - sg_order_min)/sg_order_step))
    #
    for fo in sg_filter_order:
        #print("fo", fo)
        #
        # initialise a list to keep g2 at each filter cutoff frequency
        g2_vs_window_range = []
        # loop over the filter cutoff frequencies
        for wr in sg_window_range:
            #print("wr ", wr)
            #print(sg_window_range)
            # filter with SG filter
            signal_data = savitzky_golay_filter(signal_data, time_shannon, time_step, window_size_shannon=wr, window_size_pxls=int(wr), order=int(fo))

            g2 = compute_g2(signal_data, time_step=time_step, filter_cutoff=bw_filter_cutoff, filter_order=bw_filter_order)
            g2[g2.max()<g2_min] = -1
            g2_vs_window_range.append(g2)
        g2_vs_window_range = np.array(g2_vs_window_range)
        # threshold from above
        # set g2 at time delays where the maximum value of g2 is above g2_max to -1
        g2_vs_window_range[g2_vs_window_range>g2_max] = -1

        #plot
        if to_plot:
            fig, ax = plt.subplots(1, figsize=(15, 5))
            plt.imshow(g2_vs_window_range, aspect='auto',
                       cmap=plt.get_cmap("viridis"), #vmin=0, vmax=1,
                       extent=(min(time_samples), max(time_samples),
                               min(sg_window_range), max(sg_window_range)))
            plt.title("g2 function")
            ax.set_xlabel("Time, s")
            ax.set_ylabel("SG filter window size")
            plt.colorbar()
            plt.show()

    return g2

