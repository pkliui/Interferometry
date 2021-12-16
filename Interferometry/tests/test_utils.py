import unittest

import numpy as np
import shutil, tempfile

from ddt import ddt

from Interferometry.classes.interferogram import Interferogram
from Interferometry.modules import utils


@ddt
class TestUtilsModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        #create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        #remove temporary directory after the test
        shutil.rmtree(self.test_dir)

    def test_get_minmax_indices(self):
        """
        tests getting indices of the min and max specified wav values
        :return:
        """
        wav_min_idx, wav_max_idx = utils.get_minmax_indices(wav=np.array([3, 4, 5, 6, 7, 8, 9]), wav_min=4, wav_max=6, units=1)
        self.assertEqual((1, 3), (wav_min_idx, wav_max_idx))
        #
        wav_min_idx, wav_max_idx = utils.get_minmax_indices(wav=1e-9*np.array([3, 4, 5, 6, 7, 8, 9]), wav_min=4, wav_max=6, units=1e-9)
        self.assertEqual((1, 3), (wav_min_idx, wav_max_idx))

    def test_get_wavelength_units(self):
        """
        test getting wavelength units
        """
        #
        # test allowed units
        unit = utils.get_wavelength_units("nm")
        self.assertEqual(1e-9, unit)
        #
        unit = utils.get_wavelength_units("um")
        self.assertEqual(1e-6, unit)
        #
        # test non-allowed unit
        with self.assertRaises(ValueError):
            unit = utils.get_wavelength_units("cm")

    def test_sort_list_of_tuples(self):
        """
        test sort a list of tuples
        """
        given_list = [(5, 10), (3, 4), (1, 6)]
        list1, list2 = utils.sort_list_of_tuples(given_list, sort_by_idx=0, reverse=True)
        self.assertEqual((5, 3, 1), list1)
        self.assertEqual((10, 4, 6), list2)
        #
        given_list = [(5, 10), (3, 4), (1, 6)]
        list1, list2 = utils.sort_list_of_tuples(given_list, sort_by_idx=1, reverse=True)
        self.assertEqual((5, 1, 3), list1)
        self.assertEqual((10, 6, 4), list2)

















