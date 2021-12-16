"""
This module contains functions to compute and plot spectrograms with given parameters.
"""
import numpy as np
import tftb
from matplotlib import pyplot as plt
from scipy.signal import stft, hilbert, detrend
from scipy.signal.windows import gaussian


from Interferometry.modules import plots, sampling


def stft_spectrogram(signal, time_step, zoom_in_freq, nperseg=2**6, plotting=False):
    """
    Compute a spectrogram of a time-series signal by short time Fourier transform (STFT)
    ---
    Args:
    ---
    signal: a 1D numpy array of floats
        time series to compute the spectrogram of
        normalized so that the interferogram's base level is 1
    time_step: float
        Sampling interval in time domain, s
    zoom_in_freq: float, optional
        frequency value to zoom in at, Hz
    nperseg: int, optional
        window size of the STFT
    plotting: bool, optional
        If True, generates a plot of the spectrogram
        Default: False
    ---
    Returns:
    ---
    signal_stft: 2ddarray
        STFT of signal_data
    t_stft_samples: 1darray
        Time samples
    f_stft_samples: 1darray
        Frequency samples
    """
    #
    # compute the short time fourier transform (SFTF) of the signal,
    # frequency and time samples
    f_stft_samples, t_stft_samples, signal_stft = stft(signal, 1/time_step, nperseg=nperseg,
                                                       noverlap=nperseg-1, return_onesided=False)
    # and shift the zero frequency to the center
    signal_stft = np.fft.fftshift(signal_stft, axes=0)
    f_stft_samples = np.fft.fftshift(f_stft_samples)
    #
    # crop the signal to the frequency range of interest
    signal_stft, f_stft_samples = sampling.zoom_in_2d(signal_stft, f_stft_samples, zoom_in_freq)
    #
    if plotting:
        plots.plot_2dspectrogram(f_stft_samples, 'frequency, Hz', t_stft_samples, "time, s", np.abs(signal_stft), 'spectrogram - amplitude of STFT',
                                 vmin=np.abs(signal_stft).min(), vmax=np.abs(signal_stft).max())
    return signal_stft, t_stft_samples, f_stft_samples

def wigner_ville_distribution(time_samples, signal_data, zoom_in_freq, plotting=False, vmin=0, vmax=0):
    """
    Compute Wigner-Ville distribution (time-frequency representation) of a time-series signal_data
    ---
    Args:
    ---
    time_samples: ndarray
        Array of time domain samples
    signal_data: ndarray
        timer series to compute the WVD of
    zoom_in_freq: float, optional
        frequency value to zoom in at, Hz
    plotting: bool, optional
        If True, generates a plot of the WVD
        Default: False
    vmin: float, optional
        Minimum value of the colorbar
        Default: 0
    vmax: float, optional
        Maximum value of the colorbar
        Default: 0
    ---
    Returns:
    ---
    signal_stft: ndarray
        STFT of signal_data
    t_stft_samples: ndarray
        Array of time samples
    f_stft_samples: ndarray
        Array of frequency samples
    """
    #
    if len(time_samples) == len(signal_data):
        # check if the number of samples is even, if not - delete the last entry
        if (len(time_samples) % 2 != 0) and (len(signal_data) % 2 != 0):
            time_samples = time_samples[:-1]
            signal_data = signal_data[:-1]
        #
        # compute WVD, time and frequency samples
        wvd = tftb.processing.WignerVilleDistribution(signal_data, timestamps=time_samples)
        signal_wvd, t_wvd_samples, f_wvd_samples = wvd.run()
        #
        # # because of the way WignerVilleDistribution is implemented,
        # # the maximum frequency is half of the sampling Nyquist frequency,
        # # (e.g. 1 Hz instead of 2 Hz, and the sampling is 2 * dt instead of dt
        # # hence generate new freq. samples and shift them to the center of the spectrum
        f_wvd_samples = np.fft.fftshift(np.fft.fftfreq(signal_wvd.shape[0], d=2*(time_samples[1] - time_samples[0])))
        signal_wvd = np.fft.fftshift(signal_wvd, axes=0)
        #
        # crop the signal to the frequency range of interest
        signal_wvd, f_wvd_samples = sampling.zoom_in_2d(signal_wvd, f_wvd_samples, zoom_in_freq)
        if plotting:
            plots.plot_2dspectrogram(f_wvd_samples, 'frequency, Hz', t_wvd_samples, "time, s", signal_wvd, 'spectrogram - amplitude of WVD', vmin=vmin, vmax=vmax)
    else:
        raise ValueError("Time_samples and signal_data must have the same length!")
    return signal_wvd, t_wvd_samples, f_wvd_samples
