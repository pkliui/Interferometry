"""
This module contains functions for plotting
"""


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import os
from skimage import io

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
    xlabel: str
        Label of the x axis
    ylabel: str
        Label of the y axis
    """
    fig, ax = plt.subplots(1, figsize=(15, 5))
    ax.plot(sampling_variable, signal)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    plt.show()

def plot_multiple_1dsignals(sampling_variable, xlabel, ylabel, *kwargs, save_figure=False, pathtosave=None, save_name=None):
    """
    Plots multiple samples of input 1d signals
    ---
    Args:
    ---
    sampling variable: 1d numpy array
        Samples of the domain's variable (time, frequency, etc.)
        the signal is represented in
    xlabel: str
        Label of the x axis
    ylabel: str
        Label of the y axis
    *kwargs:
        Samples of the signals to be plotted and their labels provided as tuples
        (sampling_variable, label)
    save_figure: bool
        If True, the figure will be saved
        Default: False
    pathtosave: str
        Path to save the figure
        Default: None
    save_name: str
        Name of the saved figure
        Default: None
    """
    fig, ax = plt.subplots(1, figsize=(15, 5))
    for ii in range(len(kwargs)):
        ax.plot(sampling_variable, kwargs[ii][0], label=kwargs[ii][1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()

    if save_figure:
        if os.path.exists(pathtosave):
            fig.savefig(pathtosave + "/" + save_name + ".eps")
        else:
            raise ValueError("The path to save the figure does not exist")

def plot_subplots_1dsignals(sampling_variable_1, signal_1, xlabel_1, ylabel_1,
                      sampling_variable_2, signal_2, xlabel_2, ylabel_2, signal2_cutoff_arg=None, title=None,
                            save_figure=False, pathtosave=None, save_name=None):
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
    signal2_cutoff_arg: float
        Signal2's argument value sampling_variable_2 to be used to cut off it whilst printing
    title: str
        Title of the plot
    save_figure: bool
        If True, the figure will be saved
    pathtosave: str
        Path to save the figure
    save_name: str
        Name of the saved figure

    """
    if signal2_cutoff_arg is not None:
        idx_signal2_plot_boundary=signal2_cutoff_arg
        #idx_signal2_plot_boundary = np.argwhere(sampling_variable_2 == signal2_cutoff_arg)[0]
        #print(idx_signal2_plot_boundary)
    else:
        idx_signal2_plot_boundary = len(sampling_variable_2)
        print("Displaying the full range of arguments in Fourier domain")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    plt.suptitle(title)
    ax1.plot(sampling_variable_1, signal_1)
    ax1.set_xlabel(xlabel_1)
    ax1.set_ylabel(ylabel_1)
    ax1.grid()
    ax2.plot(sampling_variable_2[0:idx_signal2_plot_boundary], signal_2[0:idx_signal2_plot_boundary])
    ax2.set_xlabel(xlabel_2)
    ax2.set_ylabel(ylabel_2)
    ax2.grid()

    if save_figure:
        if os.path.exists(pathtosave):
            plt.savefig(pathtosave + "/" + save_name + ".PNG")
        else:
            raise ValueError("The path to save the figure does not exist")

def plot_2dspectrogram(sampling_variable_1, label_1, sampling_variable_2, label_2,
                       signal, vmin, vmax, save_figure, pathtosave, save_name):
    """
    Plots the 2d spectrogram of the input signal
    ---
    Args:
    ---
    sampling_variable_1: 1d numpy array
        Samples of the domain's variable (time, frequency, etc.)
        signal 1 is represented in
    label_1: str
        Label of the x axis
    sampling_variable_2: 1d numpy array
        Samples of the domain's variable (time, frequency, etc.)
        signal 2 is represented in
    label_2: str
        Label of the y axis
    signal: 2d numpy array
        Samples of the signal
    vmin: float
        Minimum value of the colorbar
    vmax: float
        Maximum value of the colorbar
    save_figure: bool
        If True, the figure will be saved
    pathtosave: str
        Path to save the figure
    save_name: str
        Name of the saved figure
    """
    fig, axx = plt.subplots(1)
    d1 = np.abs(sampling_variable_1[1] - sampling_variable_1[0])
    d2 = np.abs(sampling_variable_2[1] - sampling_variable_2[0])
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
    plt.tight_layout()
    #plt.show()
    if save_figure:
        if os.path.exists(pathtosave):
            fig.savefig(pathtosave + "/" + save_name + ".PNG")
        else:
            raise ValueError("The path to save the figure does not exist")

def plot2dsubplots(axrow, sampling_variable_1, sampling_variable_2, signal, label_x, label_y):
    """
    Plots the 2d subplots of the input signal
    ---
    Args:
    ---
    axrow: int
        Row of the subplots
    sampling_variable_1: 1d numpy array
        Samples of the domain's variable (time, frequency, etc.)
    sampling_variable_2: 1d numpy array
        Samples of the domain's variable (time, frequency, etc.)
    signal: 2d numpy array
        Samples of the signal
    label_x: str
        Label of the x axis
    label_y: str
        Label of the y axis
    """
    d1 = np.abs(sampling_variable_1[1] - sampling_variable_1[0])
    d2 = np.abs(sampling_variable_2[1] - sampling_variable_2[0])
    vmin = np.abs(signal).min()
    vmax = np.abs(signal).max()
    im = axrow.imshow(np.abs(signal),
                    interpolation=None, origin='lower', aspect="auto",
                    extent=(
                        sampling_variable_2[0] - d2 / 2, sampling_variable_2[-1] + d2 / 2,
                        sampling_variable_1[0] - d1 / 2, sampling_variable_1[-1] + d1 / 2),
                    norm=colors.Normalize(vmin=vmin, vmax=vmax),
                    cmap="bwr"
                    )
    axrow.set_xlabel(label_x)
    axrow.set_ylabel(label_y)
    plt.colorbar(im, ax=axrow)