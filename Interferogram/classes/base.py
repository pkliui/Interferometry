import numpy as np
from scipy.signal import stft
import tftb


from matplotlib import pyplot as plt

"""
This module contains the base class to work with interferometric data.
Implements compute_spectrogram and compute_wigner_ville_distribution
common to both experimental and simulated interferometric data
"""

class BaseInterferometry:
    def __init__(self):
        super().__init__()

    def compute_spectrogram(self, signal, delta_f, nperseg=2**6, **kwargs):
        """
        Compute a spectrogram of a time-series signal by short time Fourier transform (STFT)
        ---
        Parameters
        ---
        signal: a 1D numpy array of floats
            time series to compute the spectrogram of
        delta_f: float
            Inverse of the sampling interval in time domain  (1 / self.delta_t), 1 / s
        nperseg: int, optional
            window size of the STFT
        ---
        Return
        ---
        signal_stft: ndarray
            STFT of signal
        t_stft_samples: ndarray
            Array of time samples
        f_stft_samples: ndarray
            Array of frequency samples
        """
        #
        # first looking at the power of the short time fourier transform (SFTF):
        f_stft_samples, t_stft_samples, signal_stft = stft(signal, delta_f, nperseg=nperseg, **kwargs)
        #
        # shift the frequency axis
        signal_stft = np.fft.fftshift(signal_stft, axes=0)
        f_stft_samples = np.fft.fftshift(f_stft_samples)
        #
        return signal_stft, t_stft_samples, f_stft_samples

    def compute_wigner_ville_distribution(self, time_samples, signal):
        """
        Compute Wigner-Ville distribution (time-frequency representation) of a time-series signal
        ---
        Parameters
        ---
        time_samples: ndarray
            Array of time domain samples
        signal: ndarray
            timer series to compute the WVD of
        ---
        Return
        ---
        signal_stft: ndarray
            STFT of signal
        t_stft_samples: ndarray
            Array of time samples
        f_stft_samples: ndarray
            Array of frequency samples
        """
        #
        # compute WVD
        wvd = tftb.processing.WignerVilleDistribution(signal, timestamps=time_samples)
        #
        # t_wvd is equivalent to time_samples
        # f_wvd are the "normalized frequencies" which I override below
        signal_wvd, t_wvd_samples, f_wvd_samples = wvd.run()
        #
        # because of the way WignerVilleDistribution is implemented,
        # the maximum frequency is half of the sampling Nyquist frequency,
        # (e.g. 1 Hz instead of 2 Hz set by , and the sampling is 2 * dt instead of dt
        delta_t = time_samples[1] - time_samples[0]
        f_wvd_samples = np.fft.fftshift(np.fft.fftfreq(signal_wvd.shape[0], d=2*delta_t))

        return signal_wvd, t_wvd_samples, f_wvd_samples

