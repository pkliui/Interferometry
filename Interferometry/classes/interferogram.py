import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import glob
from parse import parse
import re
import operator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors

from Interferogram.classes.base import BaseInterferometry


class Interferogram(BaseInterferometry):
    """
    class for 1D interferometric data
    """
    def __init__(self, pathtodata=None, filetoread=None, time=None, time_step=None, time_units="fs", intensity=None,
                 freq=None, ft=None):
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
        time: numpy 1D array
            Time samples
            Assumed to be equally sampled
            Default is None
        time_step: float
            Step at which the time samples were recorded
            Default is None
        time_units: str
            Units of time samples
            Possible units are "as", "fs", "ps"
            Default is femtosecond ("fs")
        intensity: numpy 1D array
            Signal intensity samples
            Default is None
        freq: 1D numpy array
            Frequency samples as set by the discrete Fourier transform
        ft: 1D numpy array
            Samples of the discrete  Fourier transform of the signal intensity data
        """
        super().__init__()
        self.pathtodata = pathtodata
        self.filetoread = filetoread
        self.time = time
        self.time_units = time_units
        self.time_step = time_step
        self.intensity = intensity
        self.freq = freq
        self.ft = ft

    def read_data(self):
        """
        reads an intensity vs. time data saved in two tabulated columns
        with no header
        ---
        Modifies the following class variables:
        ---
        self. time: numpy 1D array
            Time samples, assumed to be equally sampled
        self. intensity: numpy 1D array
            Signal intensity samples
        ---
        Returns
        ---
        Nothing
        """
        pathtofile = os.path.join(self.pathtodata, self.filetoread)
        if os.path.exists(pathtofile):
            #
            # read data from a csv or a txt file
            data = pd.read_csv(pathtofile, delimiter='\t', header=None)
            #
            # drop nan values if any
            data.dropna(inplace=True)
            #
            # convert to array
            data = np.array(data)
            self.time = data[:, 0]
            self.intensity = data[:, 1]
        else:
            raise ValueError("File path does not exist! Please enter a valid path")

    def display(self, vs_wavelength=False, temporal_data = True, wav_min=400, wav_max=800, wav_units="nm"):
        """
        Plots input data in time and frequency domains
        ---
        Parameters
        ---
        vs_wavelength: binary
            if True plots FT amplitude vs. wavelength
            if False plots FT amplitude vs. frequency (default)
        temporal_data: binary
            if True, displays both the input interferogram and its Fourier transform
            if False, only FT of the interferogram is plotted
            Default is True
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
             Samples of the discrete  Fourier transform of the signal intensity data
        self.freq: numpy 1D array
            Frequency samples as set by the discrete Fourier transform
        ---
        Returns
        ---
        Nothing
        """
        #
        # FT datatensity
        self.ft, self.freq = self.ft_data(self.intensity, self.time, self.time_step * self.get_time_units(self.time_units))
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
        if temporal_data is True:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5),  constrained_layout=True)
            ax1.plot(self.time, self.intensity)
            ax1.set_xlabel("Time delay, {}".format(self.time_units))
            ax1.set_ylabel("Signal intensity, a.u.")
            ax2.plot(x, ft_abs)
            ax2.set_xlabel(xlabel)
            ax2.set_ylabel("FT amplitude, a.u.")
        else:
            fig, ax = plt.subplots(constrained_layout=True)
            ax.plot(x, ft_abs)
            ax.set_xlabel(xlabel)
        plt.suptitle(self.filetoread[9:-4])
        plt.show()


    def display_all(self, vs_wavelength=True, wav_min=None, wav_max=None, wav_units=None):
        """
        Plots all input data that are contained in a directory  in time and frequency domains
        """
        for f in glob.glob(os.path.join(self.pathtodata, "*.txt")):
            base_name = os.path.basename(f)
            extracted_time_step = parse("{prefix}-step-{step_size}fs-{suffix}.txt", base_name)
            time_step = float(extracted_time_step["step_size"])
            #
            # read interferograms and plot data
            ifgm = Interferogram(pathtodata=self.pathtodata, filetoread=base_name, time_units=self.time_units, time_step=time_step)
            ifgm.read_data()
            ifgm.display(vs_wavelength=vs_wavelength, wav_min=wav_min, wav_max=wav_max, wav_units=wav_units)

    def display_interferogram_vs_parameter(self, parameter=None, normalizing_width=10e-15, title="Some title"):
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
                extracted_time_step = parse("{prefix}-step-{step_size}fs-{suffix}.txt", base_name)
                time_step = float(extracted_time_step["step_size"])
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
                ifgm = Interferogram(pathtodata=self.pathtodata, filetoread=base_name, time_units=self.time_units, time_step=time_step)
                ifgm.read_data()
                #
                # normalize and add parameter
                # by dividing by the mean value far away from time zero
                # and then subtracting 1 to have 0 at backgorund level
                signal = np.abs(np.array(ifgm.intensity))
                signal_mean_bg = np.mean(np.abs(np.array(ifgm.intensity[0:int(normalizing_width/(time_step * self.get_time_units(self.time_units)))])))
                signal_norm = signal / signal_mean_bg - 1
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
            im = ax.imshow(data_sorted,
                           extent=(ifgm.time[0] - ifgm.time_step / 2, ifgm.time[-1] + ifgm.time_step / 2, 0, 1),
                           cmap = plt.get_cmap("bwr"), vmin = data_sorted.min(), vmax = data_sorted.max(), norm=MidpointNormalize(midpoint=0)) #
            # set the ticks and ticks labels (0,1 is because I set 0,1 in imshow(extent=(...0,1))
            ticks = np.linspace(0, 1, len(parameter_values))
            ticklabels = ["{:6.2f}".format(i) for i in parameters_sorted]
            print(ticklabels)
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticklabels)
            ax.set_xlabel("Temporal delay, {}".format(self.time_units))
            ax.set_ylabel(parameter_names[parameter] + ", " + parameter_unit)
            ax.set_title(title)
            ax.set_aspect('auto')
            #
            # set the colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            clb = fig.colorbar(im, cax=cax, orientation='vertical')
            clb.ax.get_yaxis().labelpad = 15
            clb.ax.set_ylabel("Interferogram's intensity, a.u.", rotation=270)
            plt.show()

        else:
            raise ValueError("Parameter cannot be None!")

    def display_ft_vs_parameter(self, parameter=None, wav_fund=800e-9, log_scale=False, title="Some title"):
        """
        Plots all data in a directory as intensity heat maps
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
                extracted_time_step = parse("{prefix}-step-{step_size}fs-{suffix}.txt", base_name)
                time_step = float(extracted_time_step["step_size"])
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
                ifgm = Interferogram(pathtodata=self.pathtodata, filetoread=base_name, time_units=self.time_units, time_step=time_step)
                ifgm.read_data()
                #
                # FT the signal and normalise it by its max value
                ft, freq = self.ft_data(ifgm.intensity, ifgm.time, time_step * self.get_time_units(self.time_units))
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
    def get_time_units(units):
        """
        Converts temporal units in string format to a float number
        ---
        Return
        ---
        time_units: float
            Time unit as a float number
        """
        units_dict = {"ps": 1e-12, "fs": 1e-15, "as": 1e-18}
        if units in units_dict:
            time_units = units_dict[units]
        else:
            raise ValueError("Only the following units are allowed: {}".format(list(units_dict.keys())))
        return time_units

    @staticmethod
    def ft_data(intensity, time, time_step):
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
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
#####