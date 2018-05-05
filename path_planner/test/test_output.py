import unittest
import sys
import os
from copy import deepcopy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../src/")  # hack...
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../../../rdv-cap-sim/src/")

import offline_path_planner


class OutputTest(unittest.TestCase):
    """
        Test output of the simulation using some predefined scenario
    """

    def test_sample_absolute(self):
        """
            Test output using scenario_sample_absolute.yaml and using standard initial condition (std_ic).
            Note that this test is performed using 2-body propagator.

            Check if the total deltav and the final position match the following:
                Total DeltaV: 0.018327823164056397 km/s
                Final position: [-5820.23998662 -1859.35110161 -3559.27141534] km
        """

        dvtot, ch_final = offline_path_planner.main('scenario_sample_absolute')

        self.assertAlmostEqual(dvtot, 0.018327823164056397, 5)
        self.assertAlmostEqual(ch_final.R[0], -5820.23998662, 2)
        self.assertAlmostEqual(ch_final.R[1], -1859.35110161, 2)
        self.assertAlmostEqual(ch_final.R[2], -3559.27141534, 2)

    def test_sample_relative(self):
        """
            Test output using scenario_sample_relative.yaml and using standard initial condition (std_ic).
            Note that this test is performed using 2-body propagator.

            Check if the total deltav and the final position match the following:
                Total DeltaV: 0.023750081806959145 km/s
                Final position: [5622.97492523  1720.21133619  3910.33396728] km
        """

        dvtot, ch_final = offline_path_planner.main('scenario_sample_relative')

        self.assertAlmostEqual(dvtot, 0.023750081806959145, 5)
        self.assertAlmostEqual(ch_final.R[0], 5622.97492523, 2)
        self.assertAlmostEqual(ch_final.R[1], 1720.21133619, 2)
        self.assertAlmostEqual(ch_final.R[2], 3910.33396728, 2)


if __name__ == '__main__':
    unittest.main()

