import unittest

import numpy as np
import shutil, tempfile

from ddt import ddt

from Interferometry.classes.simulation import Simulation

from matplotlib import pyplot as plt

@ddt
class TestSimulationClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        #import class
        self.sim = Simulation()
        #create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        #remove temporary directory after the test
        shutil.rmtree(self.test_dir)

    def test_empty_arguments(self):
        """
        test the input arguments are existing
        """
        for var in ["interferogram", "g2_analytical", "g2", "freq", "ft"]:
            self.assertIn(var, self.sim.__dict__)

    def test_empty_arguments(self):
        """
        test the input arguments None
        """
        for var in ["g2_analytical", "g2"]:
            self.assertEqual(self.sim.__dict__[var], None)

    def test_frequency(self):
        """
        test if frequency is computed correctly
        """
        self.sim = Simulation(lambd0=3e8)
        self.assertEqual(self.sim.freq0, 1)
        #
        self.sim = Simulation(lambd0=3e-8)
        self.assertEqual(np.float32(self.sim.freq0), np.float32(1e16))
        #
        self.sim = Simulation(lambd0=300e-9)
        self.assertEqual(np.float32(self.sim.freq0), np.float32(1e15))

    def test_t_nsteps(self):
        """
        test number of samples in time domain t_nsteps
        """
        self.sim = Simulation(t_start=-1, t_end=1, delta_t=1)
        self.assertEqual(self.sim._t_nsteps, 3)
        #
        self.sim = Simulation(t_start=0, t_end=5, delta_t=1)
        self.assertEqual(self.sim._t_nsteps, 6)
        #
        self.sim = Simulation(t_start=0, t_end=3, delta_t=0.5)
        self.assertEqual(self.sim._t_nsteps, 7)

    def test_time_samples(self):
        """
        test samples in time domain time_samples
        """
        self.sim = Simulation(t_start=0, t_end=5, delta_t=1)
        self.assertTrue(np.array_equal(self.sim._time_samples, [0, 1, 2, 3, 4, 5]))
        #
        self.sim = Simulation(t_start=-3, t_end=3, delta_t=1)
        self.assertTrue(np.array_equal(self.sim._time_samples, [-3, -2, -1, 0, 1, 2, 3]))
        #
        self.sim = Simulation(t_start=-1, t_end=1, delta_t=0.5)
        self.assertTrue(np.array_equal(self.sim._time_samples, [-1, -0.5, 0, 0.5, 1]))

    def test_tau_samples(self):
        """
        test delay samples tau_samples
        """
        self.sim = Simulation(tau_start=0, tau_end=5, tau_step=1)
        self.assertTrue(np.array_equal(self.sim.tau_samples, [0, 1, 2, 3, 4, 5]))
        #
        self.sim = Simulation(tau_start=-3, tau_end=3, tau_step=1)
        self.assertTrue(np.array_equal(self.sim.tau_samples, [-3, -2, -1, 0, 1, 2, 3]))
        #
        self.sim = Simulation(tau_start=-1, tau_end=1, tau_step=0.5)
        self.assertTrue(np.array_equal(self.sim.tau_samples, [-1, -0.5, 0, 0.5, 1]))

    def test_get_interferogram(self):
        """
        test computing the interferogram using its analytic expression
        successful testing should also prove that the self.e_field and self.envelope variables are defined and computed correctly
        :return:
        """
        self.sim = Simulation(lambd0=800e-9, t_fwhm=10e-15,
                              t_start=-15e-15, t_end=15e-15, delta_t=0.01e-15,
                              tau_start=0, tau_end=30e-15, tau_step=0.15e-15)
        # initialise expected interferogram array
        expected_interferogram = np.zeros(len(self.sim.tau_samples))
        #
        # generate electric field and its envelope at delay=0
        e_t, a_t = self.sim.gen_e_field(delay=0)
        #
        # compute expected interferogram by its analytic formula
        for idx, delay in enumerate(self.sim.tau_samples):
            #
            # generate electric field and its envelope at a current delay
            e_t_tau, a_t_tau = self.sim.gen_e_field(delay=delay)
            #
            # compute interferogram value at a current delay
            expected_interferogram[idx] = np.sum(np.abs(a_t) ** 4) + \
                                      np.sum(np.abs(a_t_tau) ** 4) + \
                                      4 * np.sum(np.abs(a_t) ** 2 * np.abs(a_t_tau) ** 2) + \
                                      4 * np.sum((np.abs(a_t) ** 2 + np.abs(a_t_tau) ** 2) * np.real(
                a_t * np.conj(a_t_tau) * np.exp(1j * 2 * np.pi * self.sim.freq0 * delay))) + \
                                      2 * np.real(
                np.sum(a_t ** 2 * np.conj(a_t_tau) ** 2 * np.exp(2 * 1j * 2 * np.pi * self.sim.freq0 * delay)))
        #
        # compute interferogram using gen_interferogram method
        self.sim.gen_interferogram_simulation()
        # compare with analytic result
        self.assertTrue(np.array_equal(np.round(expected_interferogram), np.round(self.sim.interferogram)))

