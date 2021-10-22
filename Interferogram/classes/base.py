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

    def compute_spectrogram(self, signal, delta_f, nperseg=2**6, plotting=False, **kwargs):
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
        plotting: bool, opional
            If True, generates a plot of the spectrogram
            Default: False
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
        f_stft_samples, t_stft_samples, signal_stft = stft(signal, delta_f, nperseg=nperseg,
                                                           noverlap=nperseg-1, return_onesided=False, **kwargs)
        #
        # shift the frequency axis
        signal_stft = np.fft.fftshift(signal_stft, axes=0)
        f_stft_samples = np.fft.fftshift(f_stft_samples)
        #
        # generate frequency and time steps - must be replaced by  base class variables!
        df1 = f_stft_samples[1] - f_stft_samples[0]
        dt = t_stft_samples[1] - t_stft_samples[0]
        #print("t_stft_samples[0] - dt / 2 ", t_stft_samples[0] - dt / 2)
        #print("t_stft_samples[-1] + dt / 2", t_stft_samples[-1] + dt / 2)
        #print("f_stft_samples[0] - df1 / 2 ", f_stft_samples[0] - df1 / 2)
        #print("f_stft_samples[-1] + df1 / 2", f_stft_samples[-1] + df1 / 2)
        #
        #print(signal_stft.shape)
        #
        if plotting:
            f, axx = plt.subplots(1)
            im = axx.imshow(np.abs(signal_stft), aspect='auto',
                               interpolation=None, origin='lower',
                               extent=(t_stft_samples[0] - dt / 2, t_stft_samples[-1] + dt / 2,
                                       f_stft_samples[0] - df1 / 2, f_stft_samples[-1] + df1 / 2))
            axx.set_ylabel('frequency [Hz]')
            plt.colorbar(im, ax=axx)
            axx.set_title('spectrogram - amplitude of STFT')
        #
        return signal_stft, t_stft_samples, f_stft_samples

    def compute_wigner_ville_distribution(self, time_samples, signal, plotting=False):
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
        #print("signal ", signal.shape)
        #print("time sample", time_samples.shape)
        #
        # t_wvd_samples is equivalent to time_samples
        # f_wvd_samples are the "normalized frequencies" which I override below

        #wvd.plot(threshold=1.9, show_tf=True)
        #wvd.run()
        signal_wvd, t_wvd_samples, f_wvd_samples = wvd.run()
        print("sign  ", signal_wvd.shape)
        print("t ", t_wvd_samples.shape)
        print("f  ", f_wvd_samples.shape)

        # #
        # # generate time steps - must be replaced by  base class variables!
        # delta_t = time_samples[1] - time_samples[0]
        # #
        # # because of the way WignerVilleDistribution is implemented,
        # # the maximum frequency is half of the sampling Nyquist frequency,
        # # (e.g. 1 Hz instead of 2 Hz set by , and the sampling is 2 * dt instead of dt
        # # hence generate new freq. samples
        # f_wvd_samples = np.fft.fftshift(np.fft.fftfreq(signal_wvd.shape[0], d=2*delta_t))
        # #
        # # generate frequency steps - must be replaced by  base class variables!
        # df1 = f_wvd_samples[1] - f_wvd_samples[0]
        # #
        # if plotting:
        #     f, axx = plt.subplots(1)
        #     im = axx.imshow(np.abs(signal_wvd), aspect='auto',
        #                        interpolation=None, origin='lower',
        #                        extent=(time_samples[0] - delta_t / 2, time_samples[-1] + delta_t / 2,
        #                                f_wvd_samples[0] - df1 / 2, f_wvd_samples[-1] + df1 / 2))
        #     axx.set_ylabel('frequency [Hz]')
        #     plt.colorbar(im, ax=axx)
        #     axx.set_title("amplitude of Wigner-Ville distr.")

        return signal_wvd, t_wvd_samples, f_wvd_samples

