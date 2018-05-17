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
        """Test output using scenario_sample_absolute.yaml and using standard initial condition (std_ic).

        Note that this test is performed using 2-body propagator.
        Check if the total deltav and the final position match the following:
            Total DeltaV: 0.018327823164056397 km/s
            Final absolute position: [-5820.23998662 -1859.35110161 -3559.27141534] km
            Final relative position: [-1.70355030e+03 -4.60253140e+03 -3.42802012e-01] km
        """

        dvtot, ch_final = offline_path_planner.main('scenario_sample_absolute')

        self.assertAlmostEqual(dvtot, 0.018327823164056397, 5)
        self.assertAlmostEqual(ch_final.abs_state.R[0], -5820.23998662, 2)
        self.assertAlmostEqual(ch_final.abs_state.R[1], -1859.35110161, 2)
        self.assertAlmostEqual(ch_final.abs_state.R[2], -3559.27141534, 2)
        self.assertAlmostEqual(ch_final.rel_state.R[0], -1.70355030e+03, 2)
        self.assertAlmostEqual(ch_final.rel_state.R[1], -4.60253140e+03, 2)
        self.assertAlmostEqual(ch_final.rel_state.R[2], -3.42802012e-01, 2)

    def test_sample_relative(self):
        """Test output using scenario_sample_relative.yaml and using standard initial condition (std_ic).

        Note that this test is performed using 2-body propagator.
        Check if the total deltav and the final position match the following:
            Total DeltaV: 0.0258773258976 km/s
            Final absolute position: [6279.54516554 2253.87085778 2314.42489077] km
            Final relative position: [1.02732401e-04 1.79997101e+01 7.23865412e-13] km
        """

        dvtot, ch_final = offline_path_planner.main('scenario_sample_relative')

        self.assertAlmostEqual(dvtot, 0.0258773258976, 5)
        self.assertAlmostEqual(ch_final.abs_state.R[0], 6279.54516554, 2)
        self.assertAlmostEqual(ch_final.abs_state.R[1], 2253.87085778, 2)
        self.assertAlmostEqual(ch_final.abs_state.R[2], 2314.42489077, 2)
        self.assertAlmostEqual(ch_final.rel_state.R[0], 1.02732401e-04, 2)
        self.assertAlmostEqual(ch_final.rel_state.R[1], 1.79997101e+01, 2)
        self.assertAlmostEqual(ch_final.rel_state.R[2], 7.23865412e-13, 2)


if __name__ == '__main__':
    unittest.main()

