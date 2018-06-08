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

    def test_arg_of_perigee(self):
        """Test argument of perigee orbit adjuster.

        Given initial conditions 'test_abs', add a disturbance to the argument of perigee and try to correct it, to see
        if the manoeuvre perform correctly with 2-body propagator.

        """

        target = Satellite()
        target.initialize_satellite('target', 'test_abs', '2-body')

        chaser = Chaser()
        chaser.initialize_satellite('chaser', 'test_abs', '2-body', target)

        orb_adj = ArgumentOfPerigee()

        # Test 1 - Increase argument of perigee
        checkpoint1 = AbsoluteCP()
        checkpoint1.set_abs_state(chaser.get_mean_oe())
        checkpoint1.abs_state.w += 0.1

        orb_adj.evaluate_manoeuvre(chaser, checkpoint1, target)

        chaser_mean = chaser.get_mean_oe()

        self.assertAlmostEqual(checkpoint1.abs_state.a, chaser_mean.a, 4)
        self.assertAlmostEqual(checkpoint1.abs_state.e, chaser_mean.e, 4)
        self.assertAlmostEqual(checkpoint1.abs_state.i, chaser_mean.i, 4)
        self.assertAlmostEqual(checkpoint1.abs_state.O, chaser_mean.O, 4)

        # Test 2 - Decrease argument of perigee
        checkpoint2 = AbsoluteCP()
        checkpoint2.set_abs_state(chaser.get_mean_oe())
        checkpoint2.abs_state.w -= 0.1

        orb_adj.evaluate_manoeuvre(chaser, checkpoint2, target)

        chaser_mean = chaser.get_mean_oe()

        self.assertAlmostEqual(checkpoint2.abs_state.a, chaser_mean.a, 4)
        self.assertAlmostEqual(checkpoint2.abs_state.e, chaser_mean.e, 4)
        self.assertAlmostEqual(checkpoint2.abs_state.i, chaser_mean.i, 4)
        self.assertAlmostEqual(checkpoint2.abs_state.O, chaser_mean.O, 4)

    def test_plane_adjustment(self):
        """Test plane change orbit adjuster.

        Given initial conditions 'test_abs', check if the plane adjustment perform a correct manoeuvre in case of
        2-body propagation.

        """

        target = Satellite()
        target.initialize_satellite('target', 'test_abs', '2-body')

        chaser = Chaser()
        chaser.initialize_satellite('chaser', 'test_abs', '2-body', target)

        orb_adj = PlaneOrientation()

        # Test 1 - Increase both inclination and RAAN
        checkpoint1 = AbsoluteCP()
        checkpoint1.set_abs_state(chaser.get_mean_oe())
        checkpoint1.abs_state.i += 0.1
        checkpoint1.abs_state.O += 0.1

        orb_adj.evaluate_manoeuvre(chaser, checkpoint1, target)

        chaser_mean = chaser.get_mean_oe()

        self.assertAlmostEqual(checkpoint1.abs_state.a, chaser_mean.a, 4)
        self.assertAlmostEqual(checkpoint1.abs_state.e, chaser_mean.e, 4)
        self.assertAlmostEqual(checkpoint1.abs_state.i, chaser_mean.i, 4)
        self.assertAlmostEqual(checkpoint1.abs_state.O, chaser_mean.O, 4)

        # Test 2 - Decrease both inclination and RAAN
        checkpoint2 = AbsoluteCP()
        checkpoint2.set_abs_state(chaser.get_mean_oe())
        checkpoint2.abs_state.i -= 0.1
        checkpoint2.abs_state.O -= 0.1

        orb_adj.evaluate_manoeuvre(chaser, checkpoint2, target)

        chaser_mean = chaser.get_mean_oe()

        self.assertAlmostEqual(checkpoint2.abs_state.a, chaser_mean.a, 4)
        self.assertAlmostEqual(checkpoint2.abs_state.e, chaser_mean.e, 4)
        self.assertAlmostEqual(checkpoint2.abs_state.i, chaser_mean.i, 4)
        self.assertAlmostEqual(checkpoint2.abs_state.O, chaser_mean.O, 4)

        # Test 3 - Increase inclination and decrease RAAN
        checkpoint3 = AbsoluteCP()
        checkpoint3.set_abs_state(chaser.get_mean_oe())
        checkpoint3.abs_state.i += 0.1
        checkpoint3.abs_state.O -= 0.1

        orb_adj.evaluate_manoeuvre(chaser, checkpoint3, target)

        chaser_mean = chaser.get_mean_oe()

        self.assertAlmostEqual(checkpoint3.abs_state.a, chaser_mean.a, 4)
        self.assertAlmostEqual(checkpoint3.abs_state.e, chaser_mean.e, 4)
        self.assertAlmostEqual(checkpoint3.abs_state.i, chaser_mean.i, 4)
        self.assertAlmostEqual(checkpoint3.abs_state.O, chaser_mean.O, 4)

        # Test 4 - Decrease inclination and increase RAAN
        checkpoint4 = AbsoluteCP()
        checkpoint4.set_abs_state(chaser.get_mean_oe())
        checkpoint4.abs_state.i -= 0.1
        checkpoint4.abs_state.O += 0.1

        orb_adj.evaluate_manoeuvre(chaser, checkpoint4, target)

        chaser_mean = chaser.get_mean_oe()

        self.assertAlmostEqual(checkpoint4.abs_state.a, chaser_mean.a, 4)
        self.assertAlmostEqual(checkpoint4.abs_state.e, chaser_mean.e, 4)
        self.assertAlmostEqual(checkpoint4.abs_state.i, chaser_mean.i, 4)
        self.assertAlmostEqual(checkpoint4.abs_state.O, chaser_mean.O, 4)

    def test_multi_lambert(self):
        """Test multi-lambert orbit adjuster.

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
        orb_adj.evaluate_manoeuvre(chaser, checkpoint, target)

        self.assertLess(abs(chaser.rel_state.R[0] - checkpoint.rel_state.R[0]), 1e-2)
        self.assertLess(abs(chaser.rel_state.R[1] - checkpoint.rel_state.R[1]), 1e-2)
        self.assertLess(abs(chaser.rel_state.R[2] - checkpoint.rel_state.R[2]), 1e-2)

    def test_clohessy_wiltshire(self):
        """Test clohessy-wiltshire orbit adjuster.

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
        """Test drifting algorithm.

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
