import numpy as np
from matplotlib import pyplot as plt

from Interferometry.classes.base import BaseInterferometry


class Simulation(BaseInterferometry):
    """
    class for simulating 1D interferometric data
    """
    def __init__(self, lambd=800e-9, t_fwhm=100e-15, t_phase=0, t_start=-200e-15, t_end=200e-15, delta_t=0.15e-15,
                 tau_start=0, tau_end=100e-15, delta_tau=0.15e-15,
                 e_field=None, envelope=None, interferogram=None, g2=None):
        """
        Initializes the class to simulate interferometric data

        ---
        Parameters
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
        t_start: float, optional
            Start position in time domain
            Default: -200e-15
        t_end: float, optional
            End position in time domain (including this value)
            Default: -200e-15
        delta_t: float, optional
            Sampling interval in time domain
            Default: 0.15e-15
        tau_start: float, optional
            Start of delay in time domain
            Default: 0e-15
        tau_end: float, optional
            End of delay in time domain (including this value)
            Default: 100e-15
        delta_tau: float, optional
            Sampling of delay in time domain
            Default: 0.15e-15
        e_field: 1D array
            Samples of electric field
        envelope: 1D array
            Samples of the pulse envelope
        interferogram: 1D array
            Samples of interferogram
        g2: 1D array
            Samples of the second-order correlation function

        """
        super().__init__()
        #
        # wavelength, period and frequency of the fundamental signal
        self.lambd = lambd
        self.freq = 1 / (lambd / 3e8)
        #
        # FWHM pulse duration
        self.t_fwhm = t_fwhm
        #
        # temporal phase of a pulse
        self.t_phase = t_phase
        #
        # sampling parameters
        # start, end  and sampling interval in time domain
        self.t_start = t_start
        self.t_end = t_end # (including this value)
        self.delta_t = delta_t
        # number of samples in time domain, 1 to include the end value
        self.t_nsteps = int((t_end - t_start) / delta_t) + 1
        # time domain samples
        self.time_samples = np.linspace(t_start, t_end, self.t_nsteps)
        #
        # temporal delay parameters
        # start, end and sampling interval for temporal delay
        self.tau_start = tau_start
        self.tau_end = tau_end # (including this value)
        self.delta_tau = delta_tau
        # number of temporal delay samples
        self.tau_nsteps = int((tau_end - tau_start) / delta_tau) + 1
        # temporal delay samples
        self.tau_samples = np.linspace(tau_start, tau_end, self.tau_nsteps)
        #
        # e-field, pulse envelope and interferogram of the signal
        self.e_field = e_field
        self.envelope = envelope
        self.interferogram = interferogram
        self.g2 = g2

    def gen_e_field(self, delay=0, plotting=False):
        """
        Computes the electric field of a Gaussian laser pulse given its parameters
        ---
        Parameters
        ---
        delay: float, optional
            Temporal delay to computed the electric field of a pulse shifted in time,
            in femtoseconds
            Default is 0 fs
        plotting: bool, optional
            If True, displays a plot of the computed field
            Default is False
        ---
        Returns and sets the following class variables:
        ---
        self.e_field, self.envelope
        """
        #
        #
        if all(x is not None for x in [self.time_samples, self.t_fwhm, self.t_phase, self.freq]):
            # compute the envelope of a Gaussian pulse
            self.envelope = np.exp(-4 * np.log(2) * (self.time_samples + delay)**2 / self.t_fwhm**2) \
                            * np.exp(1j * self.t_phase * (self.time_samples + delay)**2)
            #
            # compute the electric field of a Gaussian pulse
            self.e_field = self.envelope * np.exp(-1j * 2 * np.pi * self.freq * (self.time_samples + delay))
            #
            if plotting:
                fig, ax = plt.subplots(1, figsize=(15, 5))
                ax.plot(self.time_samples, self.e_field)
                ax.set_xlabel("Time, s")
                plt.show()
        else:
            raise ValueError("Check input variables, they cannot be None")

        return self.e_field, self.envelope

    def gen_interferogram(self, plotting=False):
        """
        Computes an interferometric autocorrelation (an interferogram)
        ---
        Parameters
        ---
        plotting: bool, optional
            If True, displays a plot of the computed interferogram
            Default is False
        ---
        """
        if self.tau_samples is not None:
            #
            # iniitalise interferogram
            self.interferogram = np.zeros(len(self.tau_samples))
            # initialise electric field and its envelope at delay = 0
            e_t, a_t = self.gen_e_field(delay=0)
            #
            # compute the interferogram
            for idx, delay in enumerate(self.tau_samples):
                #
                # compute the field and its envelope at current delay
                e_t_tau, a_t_tau = self.gen_e_field(delay=delay)
                #
                # compute an interferogram value at current delay
                #self.interferogram[idx] = np.sum(np.abs((e_t + e_t_tau) ** 2) ** 2)
                #self.interferogram[idx] = np.sum((e_t + e_t_tau) ** 2)
                self.interferogram[idx] = np.sum(np.abs((e_t + e_t_tau) ** 2) ** 2)
            #
            if plotting:
                fig, ax = plt.subplots(1, figsize=(15, 5))
                ax.plot(self.tau_samples, self.interferogram)
                ax.set_xlabel("Time, s")
                plt.show()
        else:
            raise ValueError("self.tau_samples variable cannot be None")
