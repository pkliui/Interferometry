import pandas as pd
from scipy import signal
import numpy as np
import os
from matplotlib import pyplot as plt
import glob
from parse import parse
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors

from Interferometry.classes.base import BaseInterferometry
from Interferometry.modules import filtering, fourier_transforms, g2_function, normalization, plots, sampling, spectrograms, utils, tpa_utils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Interferogram(BaseInterferometry):
    """
    class for 1D interferometric data
    """
    def __init__(self, pathtodata=None, filetoread=None, lambda0 = 880e-9, tau_samples=None, tau_step=None, tau_units="fs", interferogram=None,
                 freq_samples=None, ft=None, g2=None):
        """
        Initializes the class

        ---
        Parameters
        ---
        pathtodata: str
            Path to a directory with data
            Default is None
        filetoread: str
            Filename to read, with extension
            Default is None
        lambda0: float
            Wavelength of the laser, in meters
            Default is 880e-9
        tau_samples: numpy 1D array
            Time samples
            Assumed to be equally sampled
            Default is None
        tau_step: float
            Step at which the tau samples were recorded
            Default is None
        tau_units: str, optional
            Units of temporal samples
            Possible units are "as", "fs", "ps"
            Default is femtosecond ("fs")
        interferogram: numpy 1D array
            Samples of interferogram
            Default is None
        freq_samples: 1D numpy array
            Frequency samples as set by the discrete Fourier transform
        ft: 1D numpy array
            Samples of the discrete  Fourier transform of the signal interferogram data
        g2: 1D array of floats
            Samples of the second-order correlation function,
            extracted from the interferogram
            Default: None
        """
        super().__init__()
        self.pathtodata = pathtodata
        self.filetoread = filetoread
        self.interferogram = interferogram
        self.tau_samples = tau_samples
        self.tau_step = tau_step
        self.tau_units = tau_units
        self.lambda0 = lambda0
        #
        # read interferogram and time samples
        if self.interferogram is None and \
                self.pathtodata is not None and self.filetoread is not None:
            self.read_data()
        self.freq_samples = freq_samples
        self.ft = ft
        #
        # compute the Fourier transform and the frequency samples
        if self.freq_samples is None and self.ft is None and \
                self.interferogram is not None and self.tau_step is not None and self.tau_samples is not None:
            self.ft, self.freq_samples = fourier_transforms.ft_data(self.interferogram, self.tau_step, self.tau_samples)
        #
        if self.freq_samples is not None:
            self.wav = 3e8 / self.freq_samples
        """ wavelength samples """
        self.g2 = g2
        self.g2_left_idx = None
        self.g2_right_idx = None
        self.g2_support = None
        self.tau_shannon = 1 / (2 * (3e8 / self.lambda0) * 2)

    def read_data(self):
        """
        Reads an interferogram vs. tau data saved in two tabulated columns
        with no header
        ---
        Modifies:
        ---
        self.tau_samples, self.tau_step, self.interferogram
        """
        pathtofile = os.path.join(self.pathtodata, self.filetoread)
        if os.path.exists(pathtofile):
            #
            # read data from a csv or a txt file, drop nan values if any
            data = pd.read_csv(pathtofile, delimiter='\t', header=None)
            data.dropna(inplace=True)
            data = np.array(data)
            #
            # extract temporal delay and interferogram samples
            self.tau_samples = data[:, 0]
            self.interferogram = data[:, 1]
            #
            # make sure tau is in SI units
            if self.tau_step is None:
                if sampling.get_time_step(self.tau_samples) >= sampling.get_time_units(self.tau_units):
                    self.tau_step = sampling.get_time_step(self.tau_samples) * sampling.get_time_units(self.tau_units)
                    self.tau_samples = self.tau_samples * sampling.get_time_units(self.tau_units)
                else:
                    self.tau_step = sampling.get_time_step(self.tau_samples)
            else:
                self.tau_step = sampling.get_time_step(self.tau_samples) * sampling.get_time_units(self.tau_units)
                self.tau_samples = self.tau_samples * sampling.get_time_units(self.tau_units)
            #
            # make sure tau samples are sorted in ascending order
            # and the corresponding signal values too!
            self.tau_samples = np.sort(self.tau_samples)
            self.interferogram = np.flip(self.interferogram)
        else:
            raise ValueError("File path does not exist! Please enter a valid path")

    def zero_pad_interferogram(self, pad_width=100e-15):
        """
        Zero-pads the interferogram to the same length as the tau samples
        ---
        Args
        ---
        pad_width: float, optional
            The width of the zero-valued region to pad the interferogram with, in seconds
            Default is 100e-15
        ---
        Modifies:
        ---
        self.interferogram, self.tau_samples
        """
        idx_pad_width = int(pad_width / self.tau_step)
        self.interferogram = np.pad(self.interferogram, (idx_pad_width, idx_pad_width),
                                    'constant', constant_values=(self.interferogram.mean(), self.interferogram.mean()))
        new_tau_samples_left = np.linspace(self.tau_samples[0] - pad_width, self.tau_samples[0], idx_pad_width)
        new_tau_samples_right = np.linspace(self.tau_samples[-1], self.tau_samples[-1] + pad_width, idx_pad_width)
        self.tau_samples = np.concatenate((new_tau_samples_left, self.tau_samples, new_tau_samples_right))

    def normalize_interferogram(self, normalizing_width=10e-15, t_norm_start=None):
        """
        Normalizes interferogram data
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
            self.interferogram = normalization.normalize(self.interferogram, self.tau_step, self.tau_samples,
                                                         normalizing_width=normalizing_width, t_norm_start=t_norm_start)
        else:
            raise ValueError("starting value start_at cannot be none! ")

    def display_temporal_and_ft(self, vs_wavelength=False, plot_type="both",
                                wav_min=400, wav_max=800, wav_units="nm"):
        """
        Plots interferogram in temporal and frequency domains
        ---
        Parameters
        ---
        vs_wavelength: binary
            if True plots FT amplitude vs. wavelength
            if False plots FT amplitude vs. frequency (default)
        plot_type: str
            "both" plots both temporal and Fourier domain
            "temporal" plots only temporal domain
            "fourier" plots only Fourier domain data vs. frequency or wavelength
            Default is "both"
        wav_min : float
            min wavelength to plot, units set in wav_units
            default is set to the max input freq. value
        wav_max: float
            max wavelength to plot, units set in wav_units
            default is set to the min input freq. value
        wav_units: str
            units of wav_min and wav_max
            must be one of the following: nm, um
        """
        # plot FT amplitude vs. freq. or wavelength
        if vs_wavelength is False:
            ft_samples = self.freq_samples
            ft_abs = 2.0 / len(np.abs(self.ft)) * np.abs(self.ft)
            xlabel = "Frequency, Hz"
        else:
            # get units of min and max wavelengths' boundaries and their indicies
            wav_min_idx, wav_max_idx = utils.get_minmax_indices(self.wav, wav_min, wav_max, utils.get_wavelength_units(wav_units))
            ft_samples = self.wav[wav_min_idx:wav_max_idx] * (1/utils.get_wavelength_units(wav_units))
            ft_abs = 2.0 / len(np.abs(self.ft)) * np.abs(self.ft[wav_min_idx:wav_max_idx])
            xlabel = "Wavelength, {}".format(wav_units)
        #
        if plot_type == "both":
            plots.plot_two_1dsignals(self.tau_samples, self.interferogram, "Time delay, {}".format(self.tau_units), "Signal interferogram, a.u.",
                                        ft_samples, ft_abs, xlabel, "FT amplitude, a.u.")
        elif plot_type == "fourier":
            plots.plot_1dsignal(ft_samples, ft_abs, xlabel, "Signal interferogram, a.u.")
        elif plot_type == "temporal":
            plots.plot_1dsignal(self.tau_samples, self.interferogram, "Time delay, {}".format(self.tau_units), "Signal interferogram, a.u.")
        plt.suptitle(self.filetoread[:-4])
        plt.show()

    def compute_stft_spectrogram(self, nperseg=2**6, plotting=False, zoom_in_freq=None, **kwargs):
        """
        Computes and displays experimental spectrogram by short time Fourier transform
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
        """
        spectrograms.stft_spectrogram(self.interferogram, self.tau_step, zoom_in_freq, nperseg=nperseg, plotting=plotting)

    def compute_wigner_ville_distribution(self, zoom_in_freq=None, plotting=False, vmin=0, vmax=0):
        spectrograms.wigner_ville_distribution(self.tau_samples, self.interferogram, zoom_in_freq,  plotting=plotting, vmin=vmin, vmax=vmax)

    def compute_g2(self, filter_cutoff=30e12, filter_order=6, plotting=False):
        """
        Computes the second order correlation function from the experimental interferogram
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
            Default is False
        ---
        Modifies:
        ---
        self.g2
        """
        if self.interferogram.any() and self.tau_step.any() is not None:
            self.g2 = self.g2_support * g2_function.compute_g2(self.interferogram, self.tau_step, self.tau_samples,
                                             filter_cutoff=filter_cutoff, filter_order=filter_order, plotting=plotting)[:-1]
        else:
            raise ValueError("Temporal delay sample and interferogram samples cannot be None!")

    def compute_g2_vs_lowpass_cutoff(self, cutoff_min = 1e12, cutoff_max = 30e12, cutoff_step = 1e12,
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
        g2_vs_cutoff = g2_function.g2_vs_lowpass_cutoff(self.interferogram, self.tau_samples, self.tau_step,
                                                         cutoff_min=cutoff_min, cutoff_max=cutoff_max, cutoff_step=cutoff_step,
                                                         order_min=order_min, order_max=order_max, order_step=order_step,
                                                         g2_min=g2_min, g2_max=g2_max, to_plot=to_plot)
        return g2_vs_cutoff

    def display_temporal_vs_parameter(self, parameter=None, normalizing_width=10e-15, title="Some title"):
        """
        Plots the heat maps of all interferograms in a directory 
        as a function of a parameter
        ---
        Parameters
        ---
        parameter: str
            What kind of parameter to plot the data against
            Must be one of the following: "intrange", "power"
            "intrange" stands for integration range of an electron pulse
            "power" stands for average laser power
            It is assumed that the filename contains the parameter keyword and that it is followed by
            the corresponding value
            Default is None
        normalizing_width: float, optional
            Temporal window over which to compute the mean value for interferogram normalization
            Default : 10e-15, m
        title: str, optional
            Figure title
            Default: "Some title"
        """
        #
        # initialise lists to keep data and parameters
        signal_and_parameter = []
        parameter_values = []
        parameter_names = {"intrange": "Electron pulse's integration range ", "power": "Average power "}
        #
        if parameter in parameter_names:
            #
            for f in glob.glob(os.path.join(self.pathtodata, "*.txt")):
                #
                # extract base name
                base_name = os.path.basename(f)
                #
                # extract temporal sampling step
                extracted_tau_step = parse("{prefix}-step-{step_size}fs-{suffix}.txt", base_name)["step_size"]
                tau_step = float(extracted_tau_step)
                #
                # extract parameter value
                if parameter == "intrange":
                    extracted_parameter_value = parse("{prefix}-intrange-{parameter_value}-{suffix}.txt", base_name)
                elif parameter == "power":
                    extracted_parameter_value = parse("{prefix}-power-{parameter_value}-{suffix}.txt", base_name)
                else:
                    raise ValueError("Parameter must be set to one of : 'intrange', 'power' ! ")
                #
                parameter_value = float(re.findall(r'\D+|\d+', extracted_parameter_value["parameter_value"])[0])
                parameter_unit = str(re.findall(r'\D+|\d+', extracted_parameter_value["parameter_value"])[1])
                parameter_values.append(parameter_value)
                #
                # read interferogram and plot data
                ifgm = Interferogram(pathtodata=self.pathtodata, filetoread=base_name, tau_units=self.tau_units, tau_step=tau_step)
                ifgm.read_data()
                #
                # normalize and add parameter
                # by dividing by the mean value far away from tau zero
                # and then subtracting 1 to have 0 at backgorund level
                signal_norm = self.normalize(ifgm.interferogram, tau_step * sampling.get_tau_units(self.tau_units), normalizing_width=normalizing_width)
                signal_and_parameter.append((parameter_value,  signal_norm))
            #
            # sort by parameter values in descending order so that the signal recorded at the largest parameter is on top
            parameters_sorted, data_sorted = utils.sort_list_of_tuples(signal_and_parameter, reverse=True)
            data_sorted = np.array(data_sorted)
            # sort the parameters back to be in ascending order
            # this is because the parameter ticks value will be highest at the top
            # and lowest at the bottom (as set by matplotlib)
            parameters_sorted = sorted(parameters_sorted, reverse=False)
            #1
            # plot
            fig, ax = plt.subplots(figsize=(10, 5))
            im = ax.imshow(data_sorted, interpolation=None, 
                           extent=(ifgm.tau[0] - ifgm.tau_step / 2, ifgm.tau[-1] + ifgm.tau_step / 2, 0, 1),
                           cmap = plt.get_cmap("bwr"), vmin = data_sorted.min(), vmax = data_sorted.max(), norm=MidpointNormalize(midpoint=0)) #
            # set the ticks and ticks labels (0,1 is because I set 0,1 in imshow(extent=(...0,1))
            ticks = np.linspace(0, 1, len(parameter_values))
            ticklabels = ["{:6.2f}".format(i) for i in parameters_sorted]
            print(ticklabels)
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticklabels)
            ax.set_xlabel("Temporal delay, {}".format(self.tau_units))
            ax.set_ylabel(parameter_names[parameter] + ", " + parameter_unit)
            ax.set_title(title)
            ax.set_aspect('auto')
            #
            # set the colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            clb = fig.colorbar(im, cax=cax, orientation='vertical')
            clb.ax.get_yaxis().labelpad = 15
            clb.ax.set_ylabel("Interferogram's interferogram, a.u.", rotation=270)
            plt.show()

        else:
            raise ValueError("Parameter cannot be None!")

    def display_ft_vs_parameter(self, parameter=None, wav_fund=800e-9, log_scale=False, title="Some title"):
        """
        Plots all data in a directory as interferogram heat maps
        as a function of a parameter
        ---
        Parameters
        ---
        parameter: str
            What kind of parameter to plot the data against
            Must be one of the following: "intrange", "power"
            "intrange" stands for integration range of an electron pulse
            "power" stands for average laser power
            It is assumed that the filename contains the parameter keyword and that it is followed by
            the corresponding value
            Default is None
        wav_fund: float
            Fundamental wavelength present in data, m
            Needed to scale the parameter axis if ft_data=True
            Default: 800e-9 m
        log_scale: bool
            If True, log of data is plotted
            Default is False
        title: str, optional
            Figure title
            Default: "Some title"
        """
        #
        # initialise lists to keep data and parameters
        signal_and_parameter = []
        parameter_values = []
        parameter_names = {"intrange": "Electron pulse's integration range ", "power": "Average power "}
        #
        if parameter in parameter_names:
            #
            for f in glob.glob(os.path.join(self.pathtodata, "*.txt")):
                #
                # extract base name
                base_name = os.path.basename(f)
                #
                # extract temporal sampling step
                extracted_tau_step = parse("{prefix}-step-{step_size}fs-{suffix}.txt", base_name)
                tau_step = float(extracted_tau_step["step_size"])
                #
                # extract parameter values
                if parameter == "intrange":
                    extracted_parameter_value = parse("{prefix}-intrange-{parameter_value}-{suffix}.txt", base_name)
                elif parameter == "power":
                    extracted_parameter_value = parse("{prefix}-power-{parameter_value}-{suffix}.txt", base_name)
                else:
                    raise ValueError("Parameter must be set to one of : 'intrange', 'power' ! ")
                #
                parameter_value = float(re.findall(r'\D+|\d+', extracted_parameter_value["parameter_value"])[0])
                parameter_unit = str(re.findall(r'\D+|\d+', extracted_parameter_value["parameter_value"])[1])
                parameter_values.append(parameter_value)
                #print("parameter valiues", parameter_values)
                #
                # read interferogram and plot data
                ifgm = Interferogram(pathtodata=self.pathtodata, filetoread=base_name, tau_samples=self.tau_samples,
                                     tau_units=self.tau_units, tau_step=tau_step)
#ifgm.(self, pathtodata=None, filetoread=None, tau_samples=None, tau_step=None, tau_units="fs", interferogram=None,
#      freq=None, ft=None, g2=None)
                ifgm.read_data()
                # make sure all data have the same length!
                # #
                # FT the signal and normalise it by its max value
                ft, freq = fourier_transforms.ft_data(ifgm.interferogram, ifgm.tau_samples,
                                        tau_step * sampling.get_tau_units(self.tau_units))
                #print("freq len", len(freq))
                signal_and_parameter.append((parameter_value, np.abs(np.array(ft)) / np.max(np.abs(np.array(ft)))))
            #
            # sort by parameter values in descending order so that the signal recorded at the largest parameter is on top
            parameters_sorted, data_sorted = utils.sort_list_of_tuples(signal_and_parameter, sort_by_idx=0, reverse=True)
            if log_scale:
                data_sorted = np.log(np.array(data_sorted)**2)
            else:
                data_sorted = np.array(data_sorted)**2
            # sort the parameters back to be in ascending order
            # this is because the parameter ticks value will be highest at the top
            # and lowest at the bottom (as set by matplotlib)
            parameters_sorted = sorted(parameters_sorted, reverse=False)
            #
            # frequency spacing
            freq_scaled = freq / (3e8 / wav_fund)
            d_freq_scaled = freq_scaled[1] - freq_scaled[0]
            #
            fig, ax = plt.subplots(figsize=(10, 5))
            im = ax.imshow(data_sorted,
                           extent=(freq_scaled[0] - d_freq_scaled/ 2, freq_scaled[-1] + d_freq_scaled / 2, 0, 1),
                           cmap = plt.get_cmap("viridis"))
            #
            # set the ticks and ticks labels (0,1 is because I set 0,1 in imshow(extent=(...0,1))
            ticks = np.linspace(0, 1, len(parameter_values))
            ticklabels = ["{:6.2f}".format(i) for i in parameters_sorted]
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticklabels)
            ax.set_xlabel("Frequency, in units of fundamental")
            ax.set_ylabel(parameter_names[parameter] + ", " + parameter_unit)
            ax.set_title(title)
            ax.set_aspect('auto')
            # set the colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            clb = fig.colorbar(im, cax=cax, orientation='vertical')
            clb.ax.get_yaxis().labelpad = 15
            clb.ax.set_ylabel("Intensity of interferogram's FT, a.u.", rotation=270)
            plt.show()
        else:
            raise ValueError("Parameter cannot be None!")

    def display_temporal_and_ft_batch(self, vs_wavelength=True, wav_min=None, wav_max=None, wav_units=None):
        """
        Plots all interferograms  contained in a given directory in temporal and frequency domains
        """
        for f in glob.glob(os.path.join(self.pathtodata, "*.txt")):
            base_name = os.path.basename(f)
            print("base name", base_name)
            #
            # read interferograms and plot data
            ifgm = Interferogram(pathtodata=self.pathtodata, filetoread=base_name, tau_units=self.tau_units)
            ifgm.read_data()
            ifgm.display_temporal_and_ft(vs_wavelength=vs_wavelength, wav_min=wav_min, wav_max=wav_max, wav_units=wav_units)

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

    def gen_g2_vs_savitsky_golay(self, sg_window_min=0.1, sg_window_max=1, sg_window_step=0.1,
                                 keep_shannon_sampling=True,
                                 sg_order_min=1, sg_order_max=6, sg_order_step=1,
                                 bw_filter_order=3, bw_filter_cutoff=1e12,
                                 g2_min=0.95, g2_max=1.05,
                                 plotting=True):

        g2_sg_window = g2_function.g2_vs_savitsky_golay(self.interferogram, self.tau_shannon, self.tau_step, self.tau_samples,
                                                        keep_shannon_sampling=keep_shannon_sampling,
                                     sg_window_min=sg_window_min, sg_window_max=sg_window_max, sg_window_step=sg_window_step,
                                     sg_order_min=sg_order_min, sg_order_max=sg_order_max, sg_order_step=sg_order_step,
                                     bw_filter_order = bw_filter_order, bw_filter_cutoff = bw_filter_cutoff,
                                     g2_min=g2_min, g2_max=g2_max,
                                     plotting=plotting)
        return g2_sg_window

    def plot_cross_section_wvd(self, tpa_freq=3e8 / 440e-9, freq_window_size=3, tpa_thresh=0.5,
                               tpa_tolerance=2e-15, vmin=-550, vmax=550, plotting=True):
        """
        Plots the cross-section of the WVD
        ---
        Args:
        ---
        tpa_freq: float, optional
            The frequency of the TPA, in Hz
            Default is 3e8 / 440e-9
        freq_window_size: int, optional
            The distance between the frequency of interest and the closest indicies we are looking for, in pixels
            Default is 3
        tpa_thresh: float, optional
            The simulated threshold of the TPA signal set by the laser pulse duration
            Default is 0.5
        tpa_tolerance: float, optional
            The maximal acceptable distance between different sub-regions of the detected TPA signal, in seconds
            Default is 2e-15
        vmin: float, optional
            The minimum value of the colorbar
            Default is -550
        vmax: float, optional
            The maximum value of the colorbar
            Default is 550
        plotting: bool, optional
            Whether to plot the cross-section or not
            Default is True
        """
        # compute Wigner-Ville distribution
        signal_wvd, t_wvd_samples, f_wvd_samples = spectrograms.wigner_ville_distribution(self.tau_samples, self.interferogram,
                                                                                          None, plotting=False, vmin=vmin, vmax=vmax)
        # get the indicies of the frequencies separated from the two-photon absorption frequency by the window_size
        tpa_idx_low, tpa_idx_high = tpa_utils.closest_indicies(tpa_freq, f_wvd_samples, freq_window_size)
        # get the WVD at the two-photon absorption frequency
        signal_wvd_tpa = tpa_utils.wigner_ville_distribution_tpa(signal_wvd, tpa_idx_low, tpa_idx_high)
        # get a loose support of the TPA signal set by the laser pulse duration through the TPA threshold
        tpa_signal_loose, max_idx = tpa_utils.loosely_thresholded_tpa_signal(signal_wvd_tpa, tpa_thresh)
        # get the tight support of the TPA signal
        tpa_signal_tight = tpa_utils.tightly_thresholded_tpa_signal(tpa_signal_loose, max_idx, self.tau_step,
                                                              tpa_tolerance=tpa_tolerance)

        tight_support = tpa_utils.tight_support_tpa(tpa_signal_tight)

        self.g2_support = tight_support

        if plotting:
            # pick only the fully connected central region of the mask
            n = 0#700
            m = len(signal_wvd_tpa) #1500

            plt.figure(figsize=(18, 6))
            plt.plot(self.tau_samples[n:m-1]*1e15, signal_wvd_tpa[n:m-1])
            plt.show()

            plt.figure(figsize=(18, 6))
            plt.plot(self.tau_samples[n:m-1]*1e15, signal_wvd_tpa[n:m-1])
            plt.plot(self.tau_samples[n:m-1]*1e15, tpa_signal_loose[n:m-1])
            plt.plot(self.tau_samples[n:m-1]*1e15, tpa_signal_tight[n:m-1])
            plt.show()

            plt.figure(figsize=(18, 6))
            plt.plot(self.tau_samples[n:m-1]*1e15, tight_support[n:m-1] * signal_wvd_tpa[n:m-1])
            plt.show()

            plt.figure(figsize=(18, 6))
            plt.plot(self.tau_samples[n:m-1]*1e15, tight_support[n:m-1])
            plt.show()

            return self.g2_support

    def plot_cross_section_stft(self, tpa_freq=3e8 / 440e-9, nperseg=2**5):
        """
        Plots the cross-section of the spectrogram
        """
        #
        # plot the cross-section of the WVD
        plt.figure(figsize=(8, 6))
        signal_stft, t_stft_samples, f_stft_samples = spectrograms.stft_spectrogram(self.interferogram, self.tau_step,
                                                                                    None, nperseg=nperseg, plotting=True)
        # get the index of the frequency closest to the tpa_freq
        tpa_idx = np.where((abs(tpa_freq - self.freq_samples) < abs(self.freq_samples[1] - self.freq_samples[0])))[0][0]
        print(tpa_idx)
        # because of the sampling peculiarities of stft, we need to shift the index by scaling it
        tpa_idx = int(tpa_idx * 0.5 * nperseg / len(self.freq_samples))

        print(tpa_idx)
        print("len(self.freq) ", len(self.freq_samples))

        tpa_idx_low = int(len(f_stft_samples)/2) + int(tpa_idx)-5
        tpa_idx_high = int(len(f_stft_samples)/2) + int(tpa_idx)+5

        print(tpa_idx_low)
        print(tpa_idx_high)

        print("len(signal_stft) ", signal_stft.shape)


        signal_stft_tpa = np.zeros(signal_stft.shape)

        print("len(signal_stft_tpa) ", signal_stft_tpa.shape)


        signal_stft_tpa[tpa_idx_low:tpa_idx_high, :] = signal_stft[tpa_idx_low:tpa_idx_high, :]

        print("len(signal_stft_tpa) ", signal_stft_tpa.shape)

        plt.plot(signal_stft_tpa.sum(axis=0))
        plt.show()

        plt.plot(signal_stft_tpa.sum(axis=1))
        plt.show()

        #
        df1 = f_stft_samples[1] - f_stft_samples[0]
        # f, axx = plt.subplots(1)
        # im = axx.imshow(np.abs(signal_stft_tpa),
        #                 interpolation=None, origin='lower', aspect="auto",
        #                 extent=(self.tau_samples[0] - self.tau_step / 2, self.tau_samples[-1] + self.tau_step / 2,
        #                         f_stft_samples[0] - df1 / 2, f_stft_samples[-1] + df1 / 2),
        #                 cmap="viridis")
        # axx.set_ylabel('frequency [Hz]')
        # plt.colorbar(im, ax=axx)
        # axx.set_title('spectrogram - amplitude of STFT')
        # plt.show()

        fig, ax = plt.subplots()
        #signal_stft_tpa = np.ma.masked_where(signal_stft_tpa == 0, signal_stft_tpa)
        ax.imshow(np.abs(signal_stft_tpa), cmap=cm.gray)
        ax.imshow(np.abs(signal_stft), cmap=cm.jet, interpolation='none')
        plt.show()

        print("hez ")

class MidpointNormalize(colors.Normalize):
    """
    Adjust color values to be symmetric with respect to zero
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
