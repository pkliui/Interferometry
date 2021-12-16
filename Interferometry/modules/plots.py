"""
This module contains functions for plotting
"""


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors

def plot_1dsignal(sampling_variable, signal, xlabel, ylabel):
    """
    Plots the samples of an input 1d signal
    ---
    Args:
    ---
    sampling variable: 1d numpy array
        Samples of the domain's variable (time, frequency, etc.)
        the signal is represented in
    signal: 1d numpy array
        Signal's samples
    """
    fig, ax = plt.subplots(1, figsize=(15, 5))
    ax.plot(sampling_variable, signal)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    plt.show()

def plot_two_1dsignals(sampling_variable_1, signal_1, xlabel_1, ylabel_1,
                      sampling_variable_2, signal_2, xlabel_2, ylabel_2):
    """
    Plots the samples of two input 1d signals
    ---
    Args:
    ---
    sampling variable_1: 1d numpy array
        Samples of the domain's variable (time, frequency, etc.)
        signal 1 is represented in
    signal_1: 1d numpy array
        Samples of signal 1
    sampling variable_2: 1d numpy array
        Samples of the domain's variable (time, frequency, etc.)
        signal 2 is represented in
    signal_2: 1d numpy array
        Samples of signal 2
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5), constrained_layout=True)
    ax1.plot(sampling_variable_1, signal_1,)
    ax1.set_xlabel(xlabel_1)
    ax1.set_ylabel(ylabel_1)
    ax1.grid()
    ax2.plot(sampling_variable_2, signal_2)
    ax2.set_xlabel(xlabel_2)
    ax2.set_ylabel(ylabel_2)
    ax2.grid()

def plot_2dspectrogram(sampling_variable_1, label_1, sampling_variable_2, label_2, signal, title, vmin, vmax):
    f, axx = plt.subplots(1)
    d1 = np.abs(sampling_variable_1[1] - sampling_variable_1[0])
    d2 = np.abs(sampling_variable_2[1] - sampling_variable_2[0])
    print("signal min", signal.min())
    print("signal max", signal.max())
    im = axx.imshow(signal,
                    interpolation=None, origin='lower', aspect="auto",
                    extent=(
                        sampling_variable_2[0] - d2 / 2, sampling_variable_2[-1] + d2 / 2,
                        sampling_variable_1[0] - d1 / 2, sampling_variable_1[-1] + d1 / 2),
                    norm=colors.Normalize(vmin=vmin, vmax=vmax),
                    cmap="bwr"
                    )
    axx.set_xlabel(label_2)
    axx.set_ylabel(label_1)
    plt.colorbar(im, ax=axx)
    axx.set_title(title)
    plt.show()