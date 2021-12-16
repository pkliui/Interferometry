import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize


from Interferometry.classes.base import BaseInterferometry
from Interferometry.modules import filtering, fourier_transforms, g2_function, normalization, plots, sampling, spectrograms, utils

class Simulation(BaseInterferometry):
    """
    class for simulating 1D interferometric data
    """
    def __init__(self, lambd0=800e-9, t_fwhm=100e-15, t_phase=0, t_start=-200e-15, t_end=200e-15, delta_t=0.15e-15,
                 tau_start=0, tau_end=100e-15, tau_step=0.15e-15, interferogram=None, g2_analytical=None, g2=None, freq=None, ft=None):
        """
        Initializes the class to simulate interferometric data

        ---
        Args:
        ---
        lambd: float, optional
            Wavelegth of light, m
            Default: 800e-9
        t_fwhm: float, optional
            Pulse duration, full width at half maximum definition, s
            Default: 100e-15
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
        """
        super().__init__()
        self.lambd0 = lambd0
        self.freq0 = 1 / (lambd0 / 3e8)
        """float: Frequency of the fundamental signal"""

        self.t_fwhm = t_fwhm
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
        #print("tau samples", self.tau_samples)
        """1d numpy array of floats: Temporal delay samples"""

        self.interferogram = interferogram
        # compute the interferogram if it is None
        if self.interferogram is None:
            #print("infgm in progress ")
            #print(self.interferogram)
            self.gen_interferogram_simulation()
            #print(self.interferogram)
            #print("  shape ",self.interferogram.shape)

        self.freq = freq
        self.ft = ft
        # compute the Fourier transform and the frequency samples if they are None
        if self.freq is None and self.ft is None and \
                self.interferogram is not None and self.tau_step is not None and self.tau_samples is not None:
            #print("ft in progress ")
            self.ft, self.freq = fourier_transforms.ft_data(self.interferogram, self.tau_step, self.tau_samples)

        self.g2_analytical = g2_analytical
        self.g2 = g2


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
        e_field: 1D array of floats
            Samples of electric field
        envelope: 1D array of floats
            Samples of the pulse envelope
        """
        #
        # of the following pulse properties are defined
        if all(x is not None for x in [self._time_samples, self.t_fwhm, self.t_phase, self.freq0]):
            #
            # compute the envelope of a Gaussian pulse
            envelope = np.exp(-4 * np.log(2) * (self._time_samples - delay)**2 / self.t_fwhm**2) \
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

    def gen_interferogram_simulation(self, temp_shift=0, plotting=False):
        """
        Computes an interferometric autocorrelation (an interferogram)

        ---
        Args:
        ---
        temp_shift: float, optional
            Arbitrary temporal shift (e.g. to simulate non-centered experimental data),
            in femtoseconds
            Default is 0 fs
        plotting: bool, optional
            If True, displays a plot of the computed interferogram
            Default is False
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
            print("ifgm shape ", self.interferogram.shape)
            #
            # initialise electric field and its envelope at delay = 0
            e_t, a_t = self.gen_e_field(delay = 0)
            print("field shape ", e_t.shape)
            #
            # compute the temporal shift in pixels
            idx_temp_shift = int(temp_shift / self.tau_step)
            #
            # compute the interferogram values at different temporal delays (taking into account the temporal shift)
            for idx, tau_sample in enumerate(self.tau_samples + self.tau_step * idx_temp_shift):
                # compute the field and its envelope at current tau_sample delay + additional temporal delay
                e_t_tau, a_t_tau = self.gen_e_field(delay=tau_sample)
                # compute the interferogram
                self.interferogram[idx] = np.sum(np.abs((e_t + e_t_tau) ** 2) ** 2)
                # interferogram with an additional field autocorrelation term
                # self.interferogram[idx] = np.sum(np.abs((e_t + e_t_tau)**2)**2) + 2 * np.sum(np.abs(e_t + e_t_tau)**2)

            print(self.interferogram.shape)
            if plotting:
                fig, ax = plt.subplots(1, figsize=(15, 5))
                ax.plot(self.tau_samples, self.interferogram)
                ax.set_xlabel("Time, s")
                plt.show()
        else:
            raise ValueError("self.tau_samples variable cannot be None")

    def normalize_interferogram_simulation(self, normalizing_width=10e-15, t_norm_start=None):
        """
        Normalizes interferogram simulation
        ---
        Args:
        ---
        normalizing_width: float, optional
            the width of integration range to be used for normalization, in seconds
        t_norm_start: float, optional
            the start time of the integration range to be used for normalization, in seconds
        ---
        Modifies:
        ---
        self.interferogram: 1D array of floats
            Samples of the normalized interferogram
        """
        if t_norm_start is not None:
            self.interferogram = normalization.normalize(self.interferogram, self.tau_step, self._time_samples,
                                                normalizing_width=normalizing_width, t_norm_start=t_norm_start)
        else:
            raise ValueError("starting value start_at cannot be none! ")

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
            #
            #e_t = np.real(e_t)
            #e_t_tau = np.real(e_t_tau)
            self.g2_analytical[idx] = np.mean(e_t * np.conj(e_t) * e_t_tau * np.conj(e_t_tau) / np.mean((e_t * np.conj(e_t))**2))
            #self.g2_analytical[idx] = np.real(np.mean(e_t * np.conj(e_t) * e_t_tau * np.conj(e_t_tau)) / np.mean((e_t * np.conj(e_t))**2))

    #
        if plotting:
            fig, ax = plt.subplots(1, figsize=(15, 5))
            ax.plot(self.tau_samples, self.g2_analytical)
            ax.set_xlabel("Time, s")
            plt.grid()
            plt.show()

    def gen_g2(self, filter_cutoff=30e12, filter_order=6, plotting=False):
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
        if (self.interferogram is not None) and (self.tau_step is not None):
            # compute the g2
            self.g2 = self.compute_g2(self.interferogram, self.tau_step, filter_cutoff=filter_cutoff, filter_order=filter_order)
        else:
            raise ValueError("Temporal delay sample and interferogram samples cannot be None!")
        #
        if plotting:
            fig, ax = plt.subplots(1, figsize=(15, 5))
            ax.plot(self.tau_samples, self.g2)
            ax.set_xlabel("Time, s")
            plt.title("g2")
            plt.grid()
            plt.show()

    def gen_g2_vs_cutoff(self, cutoff_min=1e12, cutoff_max = 30e12, cutoff_step = 1e12,
                              order_min = 1, order_max = 6, order_step = 1,
                              g2_min = 0.95, g2_max = 1.05,
                              to_plot = True):
        """
        Compute the second order correlation function from the experimental interferogram
        for different cutoff frequencies and orders of the Butterworth filter
        ---
        Args:
        ---
        signal: 1d ndarray
            Signal to be filtered
        time_samples: 1d ndarray
            Time samples of the signal
        time_step: float
            Temporal step of the signal
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
        g2_vs_cutoff = self.compute_g2_vs_cutoff(self.interferogram, self.tau_samples, self.tau_step,
                                    cutoff_min=cutoff_min, cutoff_max=cutoff_max, cutoff_step=cutoff_step,
                                    order_min=order_min, order_max=order_max, order_step=order_step,
                                    g2_min=g2_min, g2_max=g2_max, to_plot=to_plot)

        return g2_vs_cutoff

    def compute_wigner_ville_distribution(self, zoom_in_freq=None, plotting=False, vmin=0, vmax=0):
        spectrograms.wigner_ville_distribution(self.tau_samples, self.interferogram, zoom_in_freq,  plotting=plotting, vmin=vmin, vmax=vmax)

    def plot_cross_section_wvd(self, tpa_freq=3e8 / 440e-9, tpa_thresh=0.5, vmin=-550, vmax=550):
        """
        Plots the cross-section of the WVD
        """
        #
        # plot the cross-section of the WVD
        signal_wvd, t_wvd_samples, f_wvd_samples = spectrograms.wigner_ville_distribution(self.tau_samples, self.interferogram,
                                                                                          None, plotting=False, vmin=vmin, vmax=vmax);
        # get indicies of the frequencies closest to the tpa_freq
        tpa_idx = np.where((abs(tpa_freq - self.freq) < abs(self.freq[1] - self.freq[0])))[0][0]
        tpa_idx_low = int(np.ceil(len(f_wvd_samples)/2 + tpa_idx*2)-3)
        tpa_idx_high = int(np.ceil(len(f_wvd_samples)/2 + tpa_idx*2)+3)

        # get the WVD at the TPA frequency (in the vicinity of the TPA frequency)
        signal_wvd_tpa = np.zeros(signal_wvd.shape)
        signal_wvd_tpa[tpa_idx_low:tpa_idx_high, :] = np.copy(signal_wvd[tpa_idx_low:tpa_idx_high, :])

        signal_wvd_tpa = signal_wvd_tpa.sum(axis=0)
        signal_wvd_tpa = signal_wvd_tpa / signal_wvd_tpa.max()

        # get the indices of time samples within the FWHM of the laser pulse
        idx_t_fwhm_left = np.where(np.abs(self.tau_samples+self.t_fwhm/2) < abs(self.tau_samples[1] - self.tau_samples[0]))[0][0]
        idx_t_fwhm_right = np.where(np.abs(self.tau_samples-self.t_fwhm/2) < abs(self.tau_samples[1] - self.tau_samples[0]))[0][0]

        # set the mask distribution to 1 within the FWHM of the laser pulse and to 0 elsewhere
        # this is the region where the definition of the TPA is always valid
        tpa_region_mask = np.zeros(signal_wvd_tpa.shape)
        tpa_region_mask[idx_t_fwhm_left:idx_t_fwhm_right] = 1

        tpa_threshold_value = (signal_wvd_tpa[idx_t_fwhm_left] + signal_wvd_tpa[idx_t_fwhm_right])/2

        # plot the cross-section
        f, axx = plt.subplots(1)
        im = axx.plot(self.tau_samples, signal_wvd_tpa)
        axx.plot(self.tau_samples,tpa_region_mask)
        axx.legend(("{} ".format(100*tpa_thresh)),
                   )
        axx.set_ylabel('TPA signal, a.u.')
        axx.set_title("Wigner-Ville distribution at TPA frequency")
        plt.show()

        # return the mask distribution
        return tpa_region_mask, tpa_threshold_value