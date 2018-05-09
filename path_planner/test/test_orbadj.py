import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../src/")  # hack...
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../../../rdv-cap-sim/src/")

from state import *
from checkpoint import *
from orbit_adjuster import *

class OrbAdjTest(unittest.TestCase):
    """
        Test orbit adjusters
    """

    def test_multi_lambert(self):
        """
            Test multi-lambert orbit adjuster.
            Given initial conditions 'test_rel', check if the multi-lambert perform a correct manoeuvre in case of
            2-body propagation.
        """

        target = Satellite()
        target.initialize_satellite('target', 'test_rel', '2-body')

        chaser = Chaser()
        chaser.initialize_satellite('chaser', 'test_rel', '2-body', target)

        checkpoint = RelativeCP()
        checkpoint.rel_state.R = np.array([0.0, 1.0, 0.0])
        checkpoint.rel_state.V = np.array([0.0, 0.0, 0.0])
        checkpoint.t_min = 7979
        checkpoint.t_max = 7980

        orb_adj = MultiLambert()
        orb_adj.evaluate_manoeuvre(chaser, checkpoint, target, [0.0, 0.0, 0.0], True)

        self.assertLess(abs(chaser.rel_state.R[0] - checkpoint.rel_state.R[0]), 1e-2)
        self.assertLess(abs(chaser.rel_state.R[1] - checkpoint.rel_state.R[1]), 1e-2)
        self.assertLess(abs(chaser.rel_state.R[2] - checkpoint.rel_state.R[2]), 1e-2)

    def test_clohessy_wiltshire(self):
        """
            Test clohessy-wiltshire orbit adjuster.
            Given initial conditions 'test_rel', check if the clohessy-wiltshire perform a correct manoeuvre in case of
            2-body propagation.
        """

        target = Satellite()
        target.initialize_satellite('target', 'test_rel', '2-body')

        chaser = Chaser()
        chaser.initialize_satellite('chaser', 'test_rel', '2-body', target)

        checkpoint = RelativeCP()
        checkpoint.rel_state.R = np.array([0.0, 1.0, 0.0])
        checkpoint.rel_state.V = np.array([0.0, 0.0, 0.0])
        checkpoint.t_min = 7979
        checkpoint.t_max = 7980

        orb_adj = ClohessyWiltshire()
        orb_adj.evaluate_manoeuvre(chaser, checkpoint, target)

        self.assertLess(abs(chaser.rel_state.R[0] - checkpoint.rel_state.R[0]), 1e-2)
        self.assertLess(abs(chaser.rel_state.R[1] - checkpoint.rel_state.R[1]), 1e-2)
        self.assertLess(abs(chaser.rel_state.R[2] - checkpoint.rel_state.R[2]), 1e-2)

    def test_drift(self):
        """
            Test drifting algorithm.
            Given initial conditions , check if the drifting algorithm perfom a correct manoeuvre in case of 2-body
            propagation.
        """

        target = Satellite()
        target.initialize_satellite('target', 'test_drift', '2-body')

        chaser = Chaser()
        chaser.initialize_satellite('chaser', 'test_drift', '2-body', target)

        checkpoint = RelativeCP()
        checkpoint.rel_state.R = np.array([-4.0, 8.0, 0.0])
        checkpoint.rel_state.V = np.array([0.0, 0.0, 0.0])
        checkpoint.error_ellipsoid = np.array([0.2, 0.5, 0.2])

        orb_adj = Drift()
        orb_adj.evaluate_manoeuvre(chaser, checkpoint, target, [])

        self.assertLess(abs(chaser.rel_state.R[0] - checkpoint.rel_state.R[0]), 1e-2)
        self.assertLess(abs(chaser.rel_state.R[1] - checkpoint.rel_state.R[1]), 1e-2)
        self.assertLess(abs(chaser.rel_state.R[2] - checkpoint.rel_state.R[2]), 1e-2)


if __name__ == '__main__':
    unittest.main()
