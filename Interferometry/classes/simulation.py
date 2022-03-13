import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

from Interferometry.classes.base import BaseInterferometry
from Interferometry.modules import filtering, fourier_transforms, g2_function, normalization, plots, sampling, \
    spectrograms, utils, tpa_utils, minimization
from Interferometry.modules.filtering import low_pass_filter
import os

import matplotlib as mpl
mpl.rcdefaults()
mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.unicode'] = True
mpl.rcParams['ps.usedistiller'] = 'xpdf'
mpl.rcParams['ps.papersize'] = 'a4'
plt.rcParams.update({'font.size': 19})


class Simulation(BaseInterferometry):
    """
    class for simulating 1D interferometric data
    """
    def __init__(self, lambd0=800e-9, t_fwhm0=100e-15, gvd=0e-30,  t_tpa0=None, t_phase=0, t_start=-200e-15, t_end=200e-15,
                 delta_t=0.15e-15, tau_start=0, tau_end=100e-15, tau_step=0.15e-15, interferogram=None, g2_analytical=None,
                 g2=None, freq=None, ft=None):
        """
        Initializes the class to simulate interferometric data

        ---
        Args:
        ---
        lambd: float, optional
            Wavelegth of light, m
            Default: 800e-9
        t_fwhm0: float, optional
            Pulse duration, full width at half maximum definition, s
            Default: 100e-15
        gvd: float, optional
            Group velocity dispersion, fs^2
            Default: 0e-30
        t_tpa0: float, optional
            User-set temporal width of the TPA region, s
            If None, the value of the TPA region used to compute the interferogram is set to the whole temporal range
            Default: None
        t_phase: float, optional
            Phase of a Gaussian envelope
            Default: 0
        _t_start: float, optional
            Start position in time domain
            Default: -200e-15
        _t_end: float, optional
            End position in time domain (including this value)
            Default: -200e-15
        _delta_t: float, optional
            Sampling interval in time domain
            Default: 0.15e-15
        tau_start: float, optional
            Start of delay in time domain
            Default: 0e-15
        tau_end: float, optional
            End of delay in time domain (including this value)
            Default: 100e-15
        tau_step: float, optional
            Sampling of delay in time domain
            Default: 0.15e-15
        interferogram: 1D array of floats
            Samples of simulated interferogram
        g2_analytical: 1D array of floats
            Samples of the second-order correlation function, simulated
            Default: None
        g2: 1D array of floats
            Samples of the second-order correlation function,
            extracted from the simulated interferogram
            Default: None
        freq: 1D array of floats
            Samples of the frequency domain
            Default: None
        ft: 1D array of floats
            Samples of the frequency domain, extracted from the simulated interferogram
            Default: None
        """
        super().__init__()
        self.lambd0 = lambd0
        self.freq0 = 1 / (lambd0 / 3e8)
        """float: Frequency of the fundamental signal"""

        self.t_fwhm0 = t_fwhm0
        self.gvd = gvd
        self.t_fwhm = self.t_fwhm0 * np.sqrt(1 + (8 * self.gvd * np.log(2) / self.t_fwhm0**2)**2)
        """float: FWHM pulse duration, non-zero-valued GVD """

        self.t_phase = t_phase
        self._t_start = t_start
        self._t_end = t_end
        self._delta_t = delta_t

        self._t_nsteps = int((t_end - t_start) / delta_t) + 1
        """int: Number of samples in time domain, 1 to include the end value"""

        self._time_samples = np.linspace(t_start, t_end, self._t_nsteps)
        """1d numpy array of floats: time domain samples"""

        self.tau_start = tau_start
        self.tau_end = tau_end # (including this value)
        self.tau_step = tau_step

        self.tau_nsteps = int((tau_end - tau_start) / tau_step) + 1
        """int: Number of temporal delay samples"""

        self.tau_samples = np.linspace(tau_start, tau_end, self.tau_nsteps)
        """1d numpy array of floats: Temporal delay samples"""

        self.interferogram = interferogram
        self.interferogram_unnorm = None
        """Normalized version of the interferogram"""
        # compute the interferogram if it is None
        if self.interferogram is None:
            self.gen_interferogram()

        self.freq = freq
        self.ft = ft
        # compute the Fourier transform and the frequency samples if they are None
        if self.freq is None and self.ft is None:
            if self.interferogram is not None and self.tau_step is not None and self.tau_samples is not None:
                self.ft, self.freq = fourier_transforms.ft_data(self.interferogram, self.tau_step, self.tau_samples)
            else:
                raise ValueError("Cannot compute Fourier transform without interferogram and tau_step and tau_samples")

        self.g2_analytical = g2_analytical
        self.g2 = g2
        self.g2_support = None

        self.normalizing_width = None
        self.t_norm_start = None
        """g2 support"""
        self.tau_shannon = 1 / (2 * (3e8 / self.lambd0) * 2)
        """Shannon's sampling time"""

        self.t_tpa0 = t_tpa0
        self.t_tpa = None
        """TPA width for the computation of the interferogram"""
        if self.t_tpa0 is None:
            self.t_tpa = self.tau_step * len(self.tau_samples)
        else:
            self.t_tpa = self.t_tpa0
        if self.tau_samples is not None and self.tau_step is not None:
            self.idx_low_tpa = int(0.5 * len(self.tau_samples) - 0.5 * self.t_tpa / self.tau_step)
            """Lower boundary index of the TPA region"""
            self.idx_high_tpa = int(0.5 * len(self.tau_samples) + 0.5 * self.t_tpa / self.tau_step)
            """High boundary index of the TPA region"""
        else:
            raise ValueError("tau_samples and tau_step must be defined")


    def gen_e_field(self, delay=0, plotting=False):
        """
        Computes the electric field of a Gaussian laser pulse given its parameters
        ---
        Args:
        ---
        delay: float, optional
            Temporal delay to compute the electric field of a pulse shifted in time,
            in femtoseconds
            Default is 0 fs
        plotting: bool, optional
            If True, displays a plot of the computed field
            Default is False
        ---
        Returns:
        ---
        self.e_field: 1D array of floats
            Samples of electric field
        envelope: 1D array of floats
            Samples of the pulse envelope
        """
        #
        # of the following pulse properties are defined
        if all(x is not None for x in [self._time_samples, self.t_fwhm, self.t_phase, self.freq0]):
            #
            # compute the envelope of a Gaussian pulse
            envelope = np.exp(-np.sqrt(4 * np.log(2)) * (self._time_samples - delay)**2 / self.t_fwhm**2) \
                            * np.exp(1j * self.t_phase * (self._time_samples - delay)**2)
            # compute the electric field of a Gaussian pulse
            e_field = envelope * np.exp(-1j * 2 * np.pi * self.freq0 * (self._time_samples - delay))
            #
            # plot
            if plotting:
                fig, ax = plt.subplots(1, figsize=(15, 5))
                ax.plot(self._time_samples, e_field)
                ax.set_xlabel("Time, s")
                plt.show()
        else:
            raise ValueError("Check input variables, they cannot be None")
        return e_field, envelope

    def gen_interferogram(self, temp_shift=0, add_noise=False, noise_percentage=0.05, normalize=True, plotting=False,
                          save_figure=True, pathtosave=None, save_name=None):
        """
        Computes an interferometric autocorrelation (an interferogram)
        Does not take into account the TPA region
        ---
        Args:
        ---
        temp_shift: float, optional
            Arbitrary temporal shift (e.g. to simulate non-centered experimental data),
            in femtoseconds
            Default is 0 fs
        add_noise: bool, optional
            If True, adds noise to the interferogram
            Default is False
        noise_percentage: float, optional
            Percentage of noise to add to the interferogram (as a fraction of the maximum value)
            Default is 0.05 (5%)
        normalize: bool, optional
            If True, normalizes the interferogram
        plotting: bool, optional
            If True, displays a plot of the computed interferogram
            Default is False
        save_fig: bool, optional
            If True, saves the plot of the computed interferogram
            Default is True
        pathtosave: str, optional
            Path to save the plot of the computed interferogram
            Default is None
        save_name: str, optional
            Name of the file to save the plot of the computed interferogram
            Default is None
        ---
        Modifies:
        ---
        self.interferogram: 1D array of floats
            Samples of the interferogram
        self.interferogram_norm: 1D array of floats
            Samples of the normalized interferogram

        """
        if self.tau_samples is not None:
            #
            # iniitalise interferogram
            self.interferogram = np.zeros(len(self.tau_samples))
            #
            # initialise electric field and its envelope at delay = 0
            e_t, a_t = self.gen_e_field(delay=0)
            #
            # compute the temporal shift in pixels
            idx_temp_shift = int(temp_shift / self.tau_step)
            #
            # compute the interferogram values at different temporal delays (taking into account the temporal shift)
            for idx, tau_sample in enumerate(self.tau_samples + self.tau_step * idx_temp_shift):
                # compute the field and its envelope at current tau_sample delay + additional temporal delay
                e_t_tau, a_t_tau = self.gen_e_field(delay=tau_sample)
                # compute the interferogram and its normalized version
                self.interferogram[idx] = np.sum(np.abs((e_t + e_t_tau) ** 2) ** 2)
                if normalize:
                    self.interferogram[idx] = np.sum(np.abs((e_t + e_t_tau) ** 2) ** 2) / (2*np.sum(np.abs(e_t)**4))

            # add noise to the interferogram
            if add_noise:
                self.add_noise_to_interferogram(noise_percentage)
            # update FT
            self.ft, self.freq = fourier_transforms.ft_data(self.interferogram, self.tau_step, self.tau_samples)

            self.interferogram_unnorm = self.interferogram * 2*np.sum(np.abs(e_t)**4)

            if plotting:

                fig, ax = plt.subplots(1, figsize=(10, 5))
                ax.plot(self.tau_samples, self.interferogram_unnorm)
                ax.set_xlabel("Time delay, s")
                ax.grid()
                fig.tight_layout()
                plt.title("Before normalisation")
                plt.show()

                fig, ax = plt.subplots(1, figsize=(10, 5))
                ax.plot(self.tau_samples, self.interferogram)
                ax.set_xlabel("Time delay, s")
                ax.grid()
                ax.set_yticks([0,2,4,6,8])
                fig.tight_layout()
                plt.show()

                if save_figure is True and os.path.exists(pathtosave):
                        fig.savefig(pathtosave + "/" + save_name + ".PNG")
        else:
            raise ValueError("self.tau_samples variable cannot be None")

    def rescale_interferogram(self, normalizing_width=10e-15, t_norm_start=None):
        """
        Rescales interferogram to 1 : N base-level-to-total-height ratio
        so that the base level is at 1.
        ---
        Parameters
        ---
        normalizing_width: float, optional
            the width of integration range to be used for signals' normalization, in seconds
        t_norm_start: float
            the start time of the integration range to be used for normalization, in seconds
        ---
        Modifies:
        ---
        self.interferogram
        """
        if t_norm_start is not None:
            self.interferogram = normalization.rescale_1_to_n(self.interferogram, self.tau_step, self.tau_samples,
                                                              normalizing_width=normalizing_width, t_norm_start=t_norm_start)
        else:
            raise ValueError("starting value start_at cannot be none! ")

    def normalize_interferogram_by_infinity(self, normalizing_width=None, t_norm_start=None):
        """
        Normalizes interferogram data by its value at infinity
        ---
        Parameters
        ---
        normalizing_width: float, optional
            the width of integration range to be used for signals' normalization, in seconds
        t_norm_start: float
            the start time of the integration range to be used for normalization, in seconds
        ---
        Modifies:
        ---
        self.interferogram
        """
        if t_norm_start is not None and normalizing_width is not None:
            self.interferogram = normalization.normalize_by_value_at_infinity(self.interferogram, self.tau_step, self.tau_samples,
                                                                              normalizing_width=normalizing_width, t_norm_start=t_norm_start)
        else:
            raise ValueError("starting value start_at cannot be none! ")

    def gen_complex_interferogram(self, field_ac_weight=0.5, interferometric_ac_weight=0.5, temp_shift=0, add_noise=False, noise_percentage=0.15,
                                  normalize=True, plotting=False, save_figure=True, pathtosave=None, save_name=None):
        """
        Computes a weighted sum of the field autocorrelation and the interferometric autocorrelation
        Sets the interferometric autocorrelation to zero  outside of the TPA region
        ---
        Args:
        ---
        field_ac_weight: float, optional
            Weight of the field autocorrelation in the final signal
            Default is 0.5
        interferometric_ac_weight: float, optional
            Weight of the interferometric autocorrelation in the final signal
            Default is 0.5
        temp_shift: float, optional
            Arbitrary temporal shift (e.g. to simulate non-centered experimental data),
            in femtoseconds
            Default is 0 fs
        add_noise: bool, optional
            If True, adds Gaussian noise to the final signal
            Default is False
        noise_percentage: float, optional
            Percentage of the signal to be added as noise (as a fraction of the signal amplitude)
            Default is 0.15 (15%)
        normalize: bool, optional
            If True, normalizes the final signal
        plotting: bool, optional
            If True, displays a plot of the computed interferogram
            Default is False
        save_figure: bool, optional
            If True, saves the plot of the computed interferogram
            Default is True
        pathtosave: str, optional
            Path to save the plot of the computed interferogram
            Default is None
        save_name: str, optional
            Name of the plot to be saved
            Default is None
        ---
        Modifies:
        ---
        self.interferogram: 1D array of floats
            Samples of the interferogram

        """
        if self.tau_samples is not None:
            #
            # iniitalise interferogram
            self.interferogram = np.zeros(len(self.tau_samples))
            #
            # initialise electric field and its envelope at delay = 0
            e_t, a_t = self.gen_e_field(delay=temp_shift)
            #
            # compute the temporal shift in pixels
            idx_temp_shift = int(temp_shift / self.tau_step)
            #
            # compute the interferogram values at different temporal delays (taking into account the temporal shift)
            for idx, tau_sample in enumerate(self.tau_samples + self.tau_step * idx_temp_shift):
                # compute the field and its envelope at current tau_sample delay + additional temporal delay
                e_t_tau, a_t_tau = self.gen_e_field(delay=tau_sample+temp_shift)
                # compute the interferogram and its normalized version
                # compute complex  interferogram trace composed of interferometric and field autocorrelations
                # before the detector
                interferometric_part = np.abs((e_t + e_t_tau)**2)**2
                field_part = np.abs(e_t + e_t_tau)**2
                # set the limits where the TPa happens in the detector
                idx_low_tpa = int(0.5 * len(self.tau_samples) - 0.5 * self.t_tpa / self.tau_step)
                idx_high_tpa = int(0.5 * len(self.tau_samples) + 0.5 * self.t_tpa / self.tau_step)
                # and set interferometric autocorrelation to the field autocorrelation outside of the TPA region
                # with appropriate weights
                if idx < idx_low_tpa:
                    interferometric_part = field_part
                    self.interferogram[idx] = interferometric_ac_weight * np.sum(interferometric_part) + \
                                              field_ac_weight * np.sum(field_part)
                elif idx > idx_high_tpa:
                    interferometric_part = field_part
                    self.interferogram[idx] = interferometric_ac_weight * np.sum(interferometric_part) + \
                                              field_ac_weight * np.sum(field_part)
                else:
                    self.interferogram[idx] = interferometric_ac_weight * np.sum(interferometric_part) + \
                                              field_ac_weight * np.sum(field_part)
                if normalize:
                    # and its normnalized version
                    self.interferogram[idx] = interferometric_ac_weight * np.sum(interferometric_part) / (2*np.sum(np.abs(e_t)**4)) + \
                                              field_ac_weight * np.sum(np.abs(e_t + e_t_tau)**2) / (2*np.sum(np.abs(e_t)**2))

            # add noise to the interferogram
            if add_noise:
                self.add_noise_to_interferogram(noise_percentage)
            #update FT
            self.ft, self.freq = fourier_transforms.ft_data(self.interferogram, self.tau_step, self.tau_samples)

            if plotting:
                fig, ax = plt.subplots(1, figsize=(15, 5))
                plt.plot(self.tau_samples, self.interferogram)
                plt.xlabel("Time delay, s")
                plt.grid()
                plt.tight_layout()
                plt.show()
                if save_figure:
                    if os.path.exists(pathtosave):
                        fig.savefig(pathtosave + "/" + save_name + ".PNG")
                    else:
                        raise ValueError("The path to save the figure does not exist")
        else:
            raise ValueError("self.tau_samples variable cannot be None")

        return self.interferogram

    def add_noise_to_interferogram(self, percentage):
        """
        Adds Gaussian noise to the interferogram
        ---
        Args:
        ---
        percentage: float
            percentage of the interferogram to be added as noise
        """
        # add noise
        if percentage is not None:
            self.interferogram += percentage * np.random.normal(0, self.interferogram.std(), self.interferogram.size)
        else:
            raise ValueError("percentage variable cannot be None")

    def add_noise(self, signal, noise_percentage):
        """
        Adds Gaussian noise to a signal
        ---
        Args:
        ---
        """
        # add noise
        if noise_percentage is not None:
            signal += noise_percentage * np.random.normal(0, signal.std(), signal.size)
        else:
            raise ValueError("percentage variable cannot be None")
        return signal

    def display_temporal_and_ft(self, ft_cutoff_freq=None, title=None, save_figure=False, pathtosave=None, save_name=None):
        """
        Displays the temporal and FT of the interferogram
        ---
        Args:
        ---
        ft_cutoff_freq: float
            frequency cutoff for the FT
        title: string
            title of the figure
        save_figure: bool
            if True, saves the figure
        pathtosave: string
            path to save the figure
        save_name: string
            name of the figure
        """
        plots.plot_subplots_1dsignals(self.tau_samples, self.interferogram, "time, fs", "Normalized interferogram intensity, a.u.",
                                      self.freq, np.abs(self.ft), "freq, Hz", "FT amplitude, a.u.",
                                      signal2_cutoff_arg=ft_cutoff_freq, title=title, save_figure=save_figure, pathtosave=pathtosave,
                                      save_name=save_name)

    def gen_g2_analytical(self, plotting=False):
        """
        Compute the second order correlation function g2 analytically
        ---
        Args:
        ---
        plotting: bool, optional
            If True, displays a plot of the computed g2
            Default is False
        ---
        Modifies:
        ---
        self.g2_analytical: 1D array of floats
            Samples of the analytically computed g2
        """
        # iniitalise g2
        self.g2_analytical = np.zeros(len(self.tau_samples))
        # initialise electric field and its envelope at delay = 0
        e_t, a_t = self.gen_e_field(delay=0)
        #
        # compute the g2 values at different temporal delays
        for idx, delay in enumerate(self.tau_samples):
            # compute the field and its envelope at current delay
            e_t_tau, a_t_tau = self.gen_e_field(delay=delay)
            # compute the g2 value at current delay
            self.g2_analytical[idx] = np.mean(e_t * np.conj(e_t) * e_t_tau * np.conj(e_t_tau) / np.mean((e_t * np.conj(e_t))**2))

        if plotting:
            fig, ax = plt.subplots(1, figsize=(15, 5))
            ax.plot(self.tau_samples, self.g2_analytical)
            ax.set_xlabel("Time, s")
            plt.grid()
            plt.show()

        return self.g2_analytical

    def gen_g2(self, filter_cutoff=30e12, filter_order=6, apply_support=False, plotting=False):
        """
        Compute the second order correlation function from the simulated interferogram
        ---
        Args:
        ---
        filter_cutoff: float, optional
            The cutoff frequency of the filter, in Hz
            Default is 30e12
        filter_order: int, optional
            The order of the filter, Default is 6
        plotting: bool, optional
            If True, displays a plot of the computed g2
        ---
        Modifies:
        ---
        self.g2: 1D array of floats
            Samples of the g2 computed from the simulated interferogram
        """
        self.g2 = g2_function.compute_g2(self.interferogram, self.tau_step, filter_cutoff=filter_cutoff, filter_order=filter_order)
        # apply support if needed
        if apply_support:
            self.g2 = self.g2 * self.g2_support
        if plotting:
            fig, ax = plt.subplots(1, figsize=(15, 5))
            ax.plot(self.tau_samples, self.g2)
            ax.set_xlabel("Time, s")
            ax.set_ylabel("g2 value, a.u.")
            plt.grid()
            plt.show()
        return self.g2

    def gen_g2_vs_cutoff(self, cutoff_min=1e12, cutoff_max=30e12, cutoff_step=1e12,
                              filter_order=3, g2_min=0.95, g2_max=1.05,
                              cbar_min= 0, cbar_max=1,
                              plotting=True,
                         ax_num=None, title=None):
        """
        Compute the second order correlation function from the experimental interferogram
        for different cutoff frequencies and orders of the Butterworth filter
        ---
        Args:
        ---
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
            The order of the filter, Default is 3
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
        ax_num: int, optional
            The number of the axis to plot on
            Default is None
        title: str, optional
            The title of the plot
            Default is None
        ---
        Returns:
        ---
        g2_vs_cutoff: 2d ndarray
            The second order correlation function as a function of the filter's cut-off frequency
        """
        g2_vs_cutoff = g2_function.g2_vs_lowpass_cutoff(self.interferogram, self.tau_samples, self.tau_step,
                                    cutoff_min=cutoff_min, cutoff_max=cutoff_max, cutoff_step=cutoff_step,
                                    filter_order=filter_order,
                                    g2_min=g2_min, g2_max=g2_max, cbar_min=cbar_min, cbar_max=cbar_max, plotting=plotting,
                                                        ax_num=ax_num, title=title)

        return g2_vs_cutoff

    def compute_stft_spectrogram(self, nperseg=2**6, plotting=False, zoom_in_freq=None, save_figure=False, pathtosave=None, save_name=None):
        """
        Computes and displays simulated spectrogram by short time Fourier transform
        ---
        Parameters
        ---
        nperseg: int, optional
            window size of the STFT
        plotting: bool, opional
            If True, generates a plot of the spectrogram
            Default: False
        zoom_in_freq: float, optional
            If not None, the spectrogram is zoomed in to the specified frequency range
            Default: None
        save_figure: bool, optional
            If True, the figure is saved
            Default: False
        pathtosave: str, optional
            The path to save the figure
            Default: None
        save_name: str, optional
            The name of the figure to save
            Default: None
        """
        signal_stft, t_stft_samples, f_stft_samples = spectrograms.stft_spectrogram(self.interferogram, self.tau_step, zoom_in_freq, nperseg=nperseg, plotting=plotting,
                                      save_figure=save_figure, pathtosave=pathtosave, save_name=save_name)
        return signal_stft, t_stft_samples, f_stft_samples

    def compute_wigner_ville_distribution(self, zoom_in_freq=None, plotting=False, vmin=0, vmax=0, save_figure=False, pathtosave=None, save_name=None):
        """
        Computes and displays simulated Wigner-Ville distribution
        ---
        Parameters
        ---
        zoom_in_freq: float, optional
            If not None, the Wigner-Ville distribution is zoomed in to the specified frequency range
            Default: None
        plotting: bool, opional
            If True, generates a plot of the Wigner-Ville distribution
            Default: False
        vmin: float, optional
            The minimum value of the colorbar
            Default: 0
        vmax: float, optional
            The maximum value of the colorbar
            Default: 0
        save_figure: bool, optional
            If True, the figure is saved
            Default: False
        pathtosave: str, optional
            The path to save the figure
            Default: None
        save_name: str, optional
            The name of the figure to save
            Default: None
        """
        spectrograms.wigner_ville_distribution(self.tau_samples, self.interferogram, zoom_in_freq,  plotting=plotting, vmin=vmin, vmax=vmax,
                                               save_figure=save_figure, pathtosave=pathtosave, save_name=save_name)

    def g2_support_simulated(self, tpa_freq=3e8 / 440e-9, freq_window_size = 3, vmin=-550, vmax=550,
                               plotting=True):
        """
        Plots the cross-section of the WVD
        ---
        Args:
        ---
        tpa_freq: float, optional
            The frequency of the TPA, in Hz
            Default is 3e8 / 440e-9
        freq_window_size: int
            The distance between the frequency of interest and the closest indicies we are looking for, in pixels
            Default is 3
        vmin: float, optional
            The minimum value of the colorbar
            Default: -550
        vmax: float, optional
            The maximum value of the colorbar
            Default: 550
        plotting: bool, optional
            If True, generates a plot of the Wigner-Ville distribution
            Default: False
        ---
        Returns:
        ---
        g2_support: ndarray
            The cross-section of the WVD
        tpa_threshold: float
            The threshold of the TPA
        """
        #
        # get the cross-section of the WVD
        signal_wvd, t_wvd_samples, f_wvd_samples = spectrograms.wigner_ville_distribution(self.tau_samples, self.interferogram,
                                                                                          None, plotting=False, vmin=vmin, vmax=vmax);
        # get indicies of the frequencies closest to the tpa_freq
        tpa_idx_low, tpa_idx_high = tpa_utils.closest_indicies(tpa_freq, f_wvd_samples, freq_window_size)

        # get the WVD at the two-photon absorption frequency
        signal_wvd_tpa = tpa_utils.wigner_ville_distribution_tpa(signal_wvd, tpa_idx_low, tpa_idx_high)
        #
        # get the tight siupport of the g2 function and the corresponding threshold value of the WVD distribution at TPA frequency
        if self.t_tpa0 is None:
            tight_support, tpa_thresh = tpa_utils.tight_support_tpa_simulation(self.tau_samples, self.t_fwhm, signal_wvd_tpa)
        else:
            tight_support, tpa_thresh = tpa_utils.tight_support_tpa_simulation(self.tau_samples, self.t_tpa0, signal_wvd_tpa)

        self.g2_support = tight_support

        if plotting:
            plots.plot_multiple_1dsignals(t_wvd_samples*1e15, "time, fs", "Intensity, a.u.",
                                  (signal_wvd_tpa, "TPA signal"), (self.g2_support, "TPA support"))

        # return the support and the TPA threshold value
        print("Threshold value of the WVD at TPA frequency is ", tpa_thresh)
        return self.g2_support, tpa_thresh

    def apply_savitzky_golay_filter(self, window_size_shannon=1, window_size_pxls=None,  order=2):
        """
        Apply a Savitzky-Golay filter to the interferogram
        ---
        Args:
        ---
        window_size_shannon: float, optional
            The window size of the Savitzky-Golay filter, relative to the Shannon's sampling interval
            Default is 1
        window_size_pxls: int, optional
            The window size of the Savitzky-Golay filter, in pixels
            Default is None
        order: int, optional
            The order of the Savitzky-Golay filter
            Default is 2
        """
        self.interferogram = filtering.savitzky_golay_filter(self.interferogram, self.tau_shannon, self.tau_step,
                                                             window_size_shannon=window_size_shannon, window_size_pxls=window_size_pxls, order=order)
