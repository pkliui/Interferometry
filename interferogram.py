import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt


class Interferogram:
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
        self.pathtodata = pathtodata
        self.filetoread = filetoread
        self.time = time
        self.time_units = time_units
        self.time_step = time_step
        self.intensity = intensity
        self.freq = freq
        self.ft = ft

        fullpath = os.path.join(self.pathtodata, self.filetoread)
        if self.time is None and os.path.exists(fullpath):
            self.read_data()
        else:
            raise ValueError("The path {} is {} ".format(fullpath, os.path.exists(fullpath)))

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

    def display(self, by_wavelength=False, temporal_data = True, wav_min=400, wav_max=800, wav_units="nm"):
        """
        Plots input data in time and frequency domains
        ---
        Parameters
        ---
        by_wavelength: binary
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

        if by_wavelength is False:
            freq = self.freq
            ft_abs = 2.0 / len(np.abs(self.ft)) * np.abs(self.ft)
            xlabel = "Frequency, Hz"
        else:
            freq = wav[wav_min_idx:wav_max_idx] * (1/self.get_wavelength_units(wav_units))
            ft_abs = 2.0 / len(np.abs(self.ft)) * np.abs(self.ft[wav_min_idx:wav_max_idx])
            xlabel = "Wavelength, {}".format(wav_units)

        if temporal_data is True:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5),  constrained_layout=True)
            ax1.plot(self.time, self.intensity)
            ax1.set_xlabel("Time delay, {}".format(self.time_units))
            ax1.set_ylabel("Signal intensity, a.u.")
            ax2.plot(freq, ft_abs)
            ax2.set_xlabel(xlabel)
            ax2.set_ylabel("FT amplitude, a.u.")
        else:
            fig, ax = plt.subplots(constrained_layout=True)
            ax.plot(freq, ft_abs)
            ax.set_xlabel(xlabel)
        plt.suptitle(self.filetoread[9:-4])
        plt.show()



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
        ft = np.fft.rfft(np.sqrt(intensity))[1:]
        freq = np.fft.rfftfreq(len(time), time_step)[1:]
        return ft, freq
