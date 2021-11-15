import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import glob
from parse import parse
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors

from Interferometry.classes.base import BaseInterferometry


class Interferogram(BaseInterferometry):
    """
    class for 1D interferometric data
    """
    def __init__(self, pathtodata=None, filetoread=None, tau_samples=None, tau_step=None, tau_units="fs", interferogram=None,
                 freq=None, ft=None, g2=None):
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
        freq: 1D numpy array
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
        self.tau_samples = tau_samples
        self.tau_step = tau_step
        self.tau_units = tau_units
        self.interferogram = interferogram
        self.freq = freq
        self.ft = ft
        self.g2 = g2

    def read_data(self):
        """
        Reads an interferogram vs. tau data saved in two tabulated columns
        with no header

        ---
        Modifies:
        ---
        self.tau_samples, self.interferogram
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
                if self.get_tau_step() >= self.get_tau_units(self.tau_units):
                    self.tau_step = self.get_tau_step() * self.get_tau_units(self.tau_units)
                    self.tau_samples = self.tau_samples * self.get_tau_units(self.tau_units)
                else:
                    self.tau_step = self.get_tau_step()
            else:
                self.tau_step = self.get_tau_step() * self.get_tau_units(self.tau_units)
                self.tau_samples = self.tau_samples * self.get_tau_units(self.tau_units)
            #
            # make sure tau samples are sorted in ascending order
            # and the corresponding signal values too
            self.tau_samples = np.sort(self.tau_samples)
            self.interferogram = np.flip(self.interferogram)
        else:
            raise ValueError("File path does not exist! Please enter a valid path")

    def display_temporal_and_ft(self, vs_wavelength=False, plot_type="both", wav_min=400, wav_max=800, wav_units="nm"):
        """
        Plots interferogram in temporal and frequency domains
        ---
        Parameters
        ---
        vs_wavelength: binary
            if True plots FT amplitude vs. wavelength
            if False plots FT amplitude vs. frequency (default)
        plot_type: str
            "both" plots both temporal and frequency domain
            "temporal" plots only temporal domain
            "frequency" plots only frequency domain
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
        ---
        Modifies the following class variables:
        ---
        self.ft: numpy 1D array
             Samples of the discrete  Fourier transform of the signal interferogram data
        self.freq: numpy 1D array
            Frequency samples as set by the discrete Fourier transform
        ---
        Returns
        ---
        Nothing
        """
        #
        # FT datatensity
        self.ft, self.freq = self.ft_data(self.interferogram, self.tau_samples, self.tau_step)
        #
        # get wavelengths samples
        wav = self.convert_to_wavelength()
        #
        # get units of min and max wavelengths' boundaries and their indicies
        wav_min_idx, wav_max_idx = self.get_minmax_indices(wav, wav_min, wav_max, self.get_wavelength_units(wav_units))
        #
        # plot
        #
        if vs_wavelength is False:
            x = self.freq
            ft_abs = 2.0 / len(np.abs(self.ft)) * np.abs(self.ft)
            xlabel = "Frequency, Hz"
        else:
            x = wav[wav_min_idx:wav_max_idx] * (1/self.get_wavelength_units(wav_units))
            ft_abs = 2.0 / len(np.abs(self.ft)) * np.abs(self.ft[wav_min_idx:wav_max_idx])
            xlabel = "Wavelength, {}".format(wav_units)
        #
        if plot_type == "both":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5),  constrained_layout=True)
            ax1.plot(self.tau_samples, self.interferogram)
            ax1.set_xlabel("Time delay, {}".format(self.tau_units))
            ax1.set_ylabel("Signal interferogram, a.u.")
            ax2.plot(x, ft_abs)
            ax2.set_xlabel(xlabel)
            ax2.set_ylabel("FT amplitude, a.u.")
        elif plot_type == "frequency":
            fig, ax = plt.subplots(constrained_layout=True)
            ax.plot(x, ft_abs)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Signal interferogram, a.u.")
        elif plot_type == "temporal":
            fig, ax = plt.subplots(constrained_layout=True)
            ax.plot(self.tau_samples, self.interferogram)
            ax.set_xlabel("Time delay, {}".format(self.tau_units))
            ax.set_ylabel("Signal interferogram, a.u.")
        plt.suptitle(self.filetoread[:-4])
        plt.show()

    def display_all(self, vs_wavelength=True, wav_min=None, wav_max=None, wav_units=None):
        """
        Plots all interferograms that are contained in a directory in temporal and frequency domains
        """
        for f in glob.glob(os.path.join(self.pathtodata, "*.txt")):
            base_name = os.path.basename(f)
            #
            # read interferograms and plot data
            ifgm = Interferogram(pathtodata=self.pathtodata, filetoread=base_name, tau_units=self.tau_units)
            ifgm.read_data()
            ifgm.display_temporal_and_ft(vs_wavelength=vs_wavelength, wav_min=wav_min, wav_max=wav_max, wav_units=wav_units)

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
                signal_norm = self.normalize(ifgm.interferogram, tau_step * self.get_tau_units(self.tau_units), normalizing_width=normalizing_width)
                signal_and_parameter.append((parameter_value,  signal_norm))
            #
            # sort by parameter values in descending order so that the signal recorded at the largest parameter is on top
            parameters_sorted, data_sorted = self.sort_list_of_tuples(signal_and_parameter, reverse=True)
            data_sorted = np.array(data_sorted)
            # sort the parameters back to be in ascending order
            # this is because the parameter ticks value will be highest at the top
            # and lowest at the bottom (as set by matplotlib)
            parameters_sorted = sorted(parameters_sorted, reverse=False)
            #
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
                #
                # read interferogram and plot data
                ifgm = Interferogram(pathtodata=self.pathtodata, filetoread=base_name, tau_units=self.tau_units, tau_step=tau_step)
                ifgm.read_data()
                #
                # FT the signal and normalise it by its max value
                ft, freq = self.ft_data(ifgm.interferogram, ifgm.tau, tau_step * self.get_tau_units(self.tau_units))
                signal_and_parameter.append((parameter_value, np.abs(np.array(ft)) / np.max(np.abs(np.array(ft)))))
            #
            # sort by parameter values in descending order so that the signal recorded at the largest parameter is on top
            parameters_sorted, data_sorted = self.sort_list_of_tuples(signal_and_parameter, sort_by_idx=0, reverse=True)
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
        Return
        ---
        Normalized interferogram
        """
        if t_norm_start is not None:
            self.interferogram = self.normalize(self.interferogram, self.tau_step, self.tau_samples,
                                            normalizing_width=normalizing_width, t_norm_start=t_norm_start)
        else:
            raise ValueError("starting value start_at cannot be none! ")

    def compute_spectrogram_of_interferogram(self, nperseg=2**6, plotting=False, **kwargs):
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
        """
        self.compute_spectrogram(self.interferogram, self.tau_step, nperseg=nperseg, plotting=plotting, **kwargs)

    def gen_g2(self, filter_cutoff=30e12, filter_order=6, plotting=False):
        """
        Compute the second order correlation function from the experimental interferogram
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
            Samples of the g2 computed from the experimental interferogram
        """
        if self.interferogram.any() and self.tau_step.any() is not None:
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
            plt.show()

    def gen_g2_vs_cutoff_freq(self, cutoff_min = 1e12, cutoff_max = 30e12, cutoff_step = 1e12,
                              order_min = 1, order_max = 6, order_step = 1,
                              g2_min = 0.95, g2_max = 1.05,
                              to_plot = True):
        """
        Compute the second order correlation function from the experimental interferogram
        for different cutoff frequencies and orders of the Butterworth filter
        :param cutoff_min:
        :param cutoff_max:
        :param cutoff_step:
        :param order_min:
        :param order_max:
        :param order_step:
        :param g2_min:
        :param g2_max:
        :param to_plot:
        :return:
        """
        self.plot_g2_vs_cutoff_freq(self.interferogram, self.tau_samples, self.tau_step,
                              filter_cutoff_range=np.linspace(cutoff_min, cutoff_max, 1+int((cutoff_max - cutoff_min)/cutoff_step)),
                              filter_order=np.linspace(order_min, order_max, 1+int(abs((order_max - order_min))/order_step)),
                              g2_min = g2_min, g2_max = g2_max, to_plot = to_plot)


    def convert_to_wavelength(self):
        """
        Converts frequency samples to wavelength samples
        ---
        Return
        ---
        wav: 1D numpy array
            Wavelength samples, in meters
        """
        # compute the walength's samples
        wav = 3e8 / self.freq
        return wav

    def get_tau_step(self):
        """
        Get a step from experimental tau_samples
        It is assumed that the samples are equally sampled!
        """
        return np.abs(self.tau_samples[1] - self.tau_samples[0])

    @staticmethod
    def get_wavelength_units(units):
        """
        Converts wavelength  units in string format to a float number
        ---
        Return
        ---
        wav_units: float
            Wavelength unit as a float number
        """
        units_dict = {"nm": 1e-9, "um": 1e-6}
        if units in units_dict:
            wav_units = units_dict[units]
        else:
            raise ValueError("Only the following units are allowed: {}".format(list(units_dict.keys())))
        return wav_units

    @staticmethod
    def get_minmax_indices(wav, wav_min, wav_max, units):
        """
        Converts the min and max wavelength values to indices
        ---
        Return
        ---
        wav_min_idx: int
            Index of wav array where it is equal to the min wavelength wav_min
        wav_max_idx: int
            Index of wav array where it is equal to the min wavelength wav_max
        """
        # search for the indices whose elements are closest to specified xmin and xmax, respectively
        wav_min_idx = min(range(len(wav)), key=lambda i: abs(wav[i] - units * wav_min))
        wav_max_idx = min(range(len(wav)), key=lambda i: abs(wav[i] - units * wav_max))
        # make sure that they are sorted: wav_min_idx < wav_max_idx
        wav_min_idx, wav_max_idx = sorted([wav_min_idx, wav_max_idx], reverse=False)
        return wav_min_idx, wav_max_idx

    @staticmethod
    def get_tau_units(units):
        """
        Converts temporal units in string format to a float number
        ---
        Return
        ---
        tau_units: float
            Time unit as a float number
        """
        units_dict = {"ps": 1e-12, "fs": 1e-15, "as": 1e-18}
        if units in units_dict:
            tau_units = units_dict[units]
        else:
            raise ValueError("Only the following units are allowed: {}".format(list(units_dict.keys())))
        return tau_units

    @staticmethod
    def ft_data(interferogram, tau, tau_step):
        """
        Computes the Fourier transform of an input sequence
        and the corresponding frequency samples, given the signal interferogram samples,
        temporal samples and a discretization step
        ---
        Parameters
        ---
        interferogram: numpy 1D array
            Signal interferogram samples
        tau: numpy 1D array
            Time samples
            Assumed to be equally sampled
            Default is None
        tau_step: float
            Discretization step at which the tau samples were recorded
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
        ft = np.fft.rfft(interferogram)[1:]
        freq = np.fft.rfftfreq(len(tau), tau_step)[1:]
        return ft, freq

    def sort_list_of_tuples(self, list_of_tuples, sort_by_idx=0, reverse=False):
        """
        Sorts elements in a list of tuples
        ---
        Parameters
        ---
        list_of_tuples: list
            List of tuples
        sort_by_idx: int, optional
            Number of index to sort by
            E.g. if a tuple consists of two elements and we would like to sort by the second, set to 1
            Default: 0
        reverse: bool, optional
            If True, the sorting is done in ascending order.
            If False - in descending.
            Default is True
        """

        # sort by the parameter_value
        # signal_and_parameter.sort(key=operator.itemgetter(1))
        list_of_tuples.sort(key=lambda x: x[sort_by_idx], reverse=reverse)
        # split it back into sorted
        return zip(*list_of_tuples)


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
