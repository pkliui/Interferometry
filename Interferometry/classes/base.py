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

    def compute_spectrogram(self, signal, time_step, nperseg=2**6, plotting=False, **kwargs):
        """
        Compute a spectrogram of a time-series signal by short time Fourier transform (STFT)
        ---
        Parameters
        ---
        signal: a 1D numpy array of floats
            time series to compute the spectrogram of
            normalized so that the interferogram's base level is 1
        time_step: float
            Sampling interval in time domain, s
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
        f_stft_samples, t_stft_samples, signal_stft = stft(signal, 1/time_step, nperseg=nperseg,
                                                           noverlap=nperseg-1, return_onesided=False, **kwargs)
        #
        # shift the frequency axis
        signal_stft = np.fft.fftshift(signal_stft, axes=0)
        f_stft_samples = np.fft.fftshift(f_stft_samples)
        #
        # generate frequency and time steps - must be replaced by  base class variables!
        df1 = f_stft_samples[1] - f_stft_samples[0]
        dt = t_stft_samples[1] - t_stft_samples[0]
        #
        if plotting:
            f, axx = plt.subplots(1)
            im = axx.imshow(np.abs(signal_stft),
                               interpolation=None, origin='lower', aspect="auto",
                               extent=(
                                       t_stft_samples[0] - dt / 2, t_stft_samples[-1] + dt / 2,
                                       f_stft_samples[0] - df1 / 2, f_stft_samples[-1] + df1 / 2),
                            cmap="gnuplot2"
                            )
            axx.set_ylabel('frequency [Hz]')
            plt.colorbar(im, ax=axx)
            axx.set_title('spectrogram - amplitude of STFT')
        #
        return signal_stft, t_stft_samples, f_stft_samples

    def compute_wigner_ville_distribution(self, time_samples, signal, plotting=False, vmin=0, vmax=0):
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
        if len(time_samples) == len(signal):
            # check if the number of samples is even, if not - delete the last entry
            if (len(time_samples) % 2 != 0) and (len(signal) % 2 != 0):
                time_samples = time_samples[:-1]
                signal = signal[:-1]
            else:
                pass
            #
            # compute WVD
            wvd = tftb.processing.WignerVilleDistribution(signal, timestamps=time_samples)
            #
            # t_wvd_samples is equivalent to time_samples
            # f_wvd_samples are the "normalized frequencies" which I override below
            signal_wvd, t_wvd_samples, f_wvd_samples = wvd.run()
            # #
            # # generate time steps - must be replaced by base class variables!
            delta_t = time_samples[1] - time_samples[0]
            # #
            # # because of the way WignerVilleDistribution is implemented,
            # # the maximum frequency is half of the sampling Nyquist frequency,
            # # (e.g. 1 Hz instead of 2 Hz set by , and the sampling is 2 * dt instead of dt
            # # hence generate new freq. samples
            f_wvd_samples = np.fft.fftshift(np.fft.fftfreq(signal_wvd.shape[0], d=2*delta_t))
            # #
            # # generate frequency steps - must be replaced by  base class variables!
            delta_f = f_wvd_samples[1] - f_wvd_samples[0]
            # #
            if plotting:
                 f, axx = plt.subplots(1)
                 im = axx.imshow(np.fft.fftshift((signal_wvd)**1.0, axes=0),
                                 aspect='auto', origin='lower',
                                 extent=(time_samples[0] - delta_t / 2, time_samples[-1] + delta_t / 2,
                                            f_wvd_samples[0] - delta_f / 2, f_wvd_samples[-1] + delta_f / 2),
                                 cmap = plt.get_cmap("hsv"), vmin=vmin, vmax=vmax)
                 axx.set_ylabel('frequency [Hz]')
                 plt.colorbar(im, ax=axx)
                 axx.set_title("amplitude of Wigner-Ville distr.")
        else:
            raise ValueError("Time_samples and signal must have the same length!")

    def normalize(self, signal, time_step, time_samples, normalizing_width=10e-15, t_norm_start=None):
        """
        Normalizes an interferogram
        ---
        Parameters
        ---
        signal: ndarray
            timer series to normalize
        time_step:
            temporal step the signal was recorded at
        normalizing_width: float, optional
            the width of integration range to be used for signals' mormalization, in seconds
        ---
        Return
        ---
        signal_norm: ndarray
            normalized signal
        """
        if t_norm_start is not None:
            #
            # set integration range for normalization
            idx_norm_range = int(normalizing_width / time_step)
            idx_norm_start = int(np.argmin(np.abs(t_norm_start - time_samples)))
            #
            # compute the mean value of the signal's background for given integration range
            # and normalize the signal to have 1:8 ratio
            signal_mean_bg = np.mean(np.abs(np.array(signal[idx_norm_start : idx_norm_range + idx_norm_start])))
            signal -= signal_mean_bg
            signal -= - signal.min()
            signal = 8 * signal / signal.max()
        else:
            raise ValueError("starting value t_norm_start cannot be none! ")
        return signal

    def gen_g2(self, time_samples, plotting=False):
        """
        Generates the second order correlation function of the input signal
        ---
        Parameters
        ---
        signal: ndarray
            timer series to normalize
        time_step:
            temporal step the signal was recorded at
        normalizing_width: float, optional
            the width of integration range to be used for signals' mormalization, in seconds
        ---
        Return
        ---
        signal_norm: ndarray
            normalized signal
        """
        #
        # iniitalise g2
        g2 = np.zeros(len(time_samples))
        #
        # initialise electric field and its envelope at delay = 0
        e_t, a_t = self.gen_e_field(delay=0)
        #
        # compute the g2
        for idx, delay in enumerate(time_samples):
            #
            # compute the field and its envelope at current delay
            e_t_tau, a_t_tau = self.gen_e_field(delay=delay)
            #
            # compute an interferogram value at current delay
            g2[idx] = np.mean(np.conj(e_t) * np.conj(e_t_tau) * e_t_tau * e_t) / np.mean((np.conj(e_t) * e_t)**2)
        #self.g2 /= np.mean(self.g2[1500:2500])
        #
        if plotting:
            fig, ax = plt.subplots(1, figsize=(15, 5))
            ax.plot(time_samples, g2)
            ax.set_xlabel("Time, s")
            plt.show()

    def ft_data(self, intensity, time, time_step):
        """
        Computes the Fourier transform of an input sequence
        and the corresponding frequency samples, given the signal intensity samples, temporal samples and a discretization step
        ---
        Parameters
        ---
        intensity: numpy 1D array
            Signal intensity samples
        time: numpy 1D array
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
        freq = np.fft.rfftfreq(len(time), time_step)[1:]
        return ft, freq

    def plot_data(self, time_samples, signal):
        fig, ax = plt.subplots(1, figsize=(15, 5))
        ax.plot(time_samples * 10e15, signal)
        ax.set_xlabel("Time, fs")
        plt.show()