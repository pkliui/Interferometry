import unittest

import numpy as np
import shutil, tempfile

from ddt import ddt

from Interferogram.classes.interferogram import Interferogram

@ddt
class TestInterferogramClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        #import class
        self.ifgm = Interferogram()
        #create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        #remove temporary directory after the test
        shutil.rmtree(self.test_dir)

    def test_empty_arguments(self):
        """
        test the input arguments are existing and are all None
        :return:
        """
        for var in ["pathtodata", "filetoread", "time", "time_step", "intensity",
                 "freq", "ft"]:
            self.assertIn(var, self.ifgm.__dict__)
            self.assertEqual(self.ifgm.__dict__[var], None)

    def test_read_data(self):
        """
        test missing positional arguments
        test to read some non-existing data
        :return:
        """
        with self.assertRaises(TypeError):
            self.ifgm = Interferogram()
            self.ifgm.read_data()
        with self.assertRaises(ValueError):
            self.ifgm = Interferogram(pathtodata='some_non_existing_path',
                                  filetoread="some_non_existing file")
            self.ifgm.read_data()

    def test_convert_to_wavelength(self):
        """
        test converting frequencies to wavelegnths
        """
        self.ifgm = Interferogram()
        self.ifgm.freq = np.array([1, 2, 3.2, 1e17, 1.25e17])
        self.assertTrue(np.array_equal(np.array([3e8, 1.5e8, 0.9375e8, 3e-9, 2.4e-9]), self.ifgm.convert_to_wavelength()))

    def test_get_wavelength_units(self):
        """
        test getting wavelength units
        """
        #
        # test allowed units
        self.ifgm = Interferogram()
        unit = self.ifgm.get_wavelength_units("nm")
        self.assertEqual(1e-9, unit)
        #
        unit = self.ifgm.get_wavelength_units("um")
        self.assertEqual(1e-6, unit)
        #
        # test non-allowed unit
        with self.assertRaises(ValueError):
            unit = self.ifgm.get_wavelength_units("cm")

    def test_get_minmax_indices(self):
        """
        tests getting indices of the min and max specified wav values
        :return:
        """
        self.ifgm = Interferogram()
        wav_min_idx, wav_max_idx = self.ifgm.get_minmax_indices(wav=np.array([3, 4, 5, 6, 7, 8, 9]), wav_min=4, wav_max=6, units=1)
        self.assertEqual((1, 3), (wav_min_idx, wav_max_idx))
        #
        wav_min_idx, wav_max_idx = self.ifgm.get_minmax_indices(wav=1e-9*np.array([3, 4, 5, 6, 7, 8, 9]), wav_min=4, wav_max=6, units=1e-9)
        self.assertEqual((1, 3), (wav_min_idx, wav_max_idx))

    def test_get_time_units(self):
        """
        test getting time units
        """
        #
        # test allowed units
        self.ifgm = Interferogram()
        unit = self.ifgm.get_time_units("ps")
        self.assertEqual(1e-12, unit)
        #
        unit = self.ifgm.get_time_units("fs")
        self.assertEqual(1e-15, unit)
        #
        unit = self.ifgm.get_time_units("as")
        self.assertEqual(1e-18, unit)
        #
        # test non-allowed unit
        with self.assertRaises(ValueError):
            _ = self.ifgm.get_time_units("ns")

if __name__ == '__main__':
    unittest.main()