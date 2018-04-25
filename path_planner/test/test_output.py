# Note: these tests are quite preliminary....

import unittest
import sys
import os
from copy import deepcopy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../src/")  # hack...
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../../../rdv-cap-sim/src/")

import offline_path_planner
import rospace_lib

class SolverTest(unittest.TestCase):
    """
        Test output of the simulation using some predefined scenario
    """

    def test_sample_absolute(self):
        """
            Test output using scenario_sample_absolute.yaml

            Total DeltaV :
        """

        dvtot = offline_path_planner.main('scenario_camille')

        self.assertAlmostEqual(dvtot, 0.0176018439814, 5)


if __name__ == '__main__':
    unittest.main()

