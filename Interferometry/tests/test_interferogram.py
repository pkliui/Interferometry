import unittest

import numpy as np
import shutil, tempfile

from ddt import ddt

from Interferometry.classes.interferogram import Interferogram
from Interferometry.modules import sampling


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
        for var in ["pathtodata", "filetoread", "tau_samples", "tau_step", "interferogram", "freq_samples", "ft", "g2"]:
            self.assertIn(var, self.ifgm.__dict__)
            self.assertEqual(self.ifgm.__dict__[var], None)

    def test_read_data(self):
        """
        test to read some non-existing data
        :return:
        """
        with self.assertRaises(ValueError):
            self.ifgm = Interferogram(pathtodata='some_non_existing_path',
                                  filetoread="some_non_existing file")

    def test_convert_to_wavelength(self):
        """
        test converting frequencies to wavelegnths
        """
        self.ifgm = Interferogram(freq_samples=np.array([1, 2, 3.2, 1e17, 1.25e17]))
        self.assertTrue(np.array_equal(np.array([3e8, 1.5e8, 0.9375e8, 3e-9, 2.4e-9]), self.ifgm.wav))

    def test_get_time_units(self):
        """
        test getting time units
        """
        #
        # test allowed units
        self.ifgm = Interferogram()
        unit = sampling.get_time_units("ps")
        self.assertEqual(1e-12, unit)
        #
        unit = sampling.get_time_units("fs")
        self.assertEqual(1e-15, unit)
        #
        unit = sampling.get_time_units("as")
        self.assertEqual(1e-18, unit)
        #
        # test non-allowed unit
        with self.assertRaises(ValueError):
            _ = sampling.get_time_units("ns")

    def test_get_time_step(self):
        """
        test getting a temporal step
        """
        #
        self.ifgm = Interferogram()
        self.ifgm.tau_samples = np.array([1, 3])
        time_step = sampling.get_time_step(self.ifgm.tau_samples)
        self.assertEqual(2, time_step)
        #
        self.ifgm = Interferogram()
        self.ifgm.tau_samples = np.array([1e-15, 3e-15])
        time_step = sampling.get_time_step(self.ifgm.tau_samples)
        self.assertAlmostEqual(2e-15, time_step)

if __name__ == '__main__':
    unittest.main()