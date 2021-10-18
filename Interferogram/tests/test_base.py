import unittest
import shutil, tempfile

from ddt import ddt

from Interferogram.classes.base import BaseInterferometry


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
        self.baseif = BaseInterferometry()
        #create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        #remove temporary directory after the test
        shutil.rmtree(self.test_dir)

    def test_empty_arguments(self):
        """
        test the input arguments are existing and are all None
        """
        for var in []:
            self.assertIn(var, self.baseif.__dict__)
            self.assertEqual(self.baseif.__dict__[var], None)
