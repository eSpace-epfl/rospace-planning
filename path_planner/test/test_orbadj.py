import unittest
import sys
import os
from copy import deepcopy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../src/")  # hack...
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../../../rdv-cap-sim/src/")

import rospace_lib
from state import *
from datetime import datetime
from checkpoint import *
from orbit_adjuster import *

class OrbAdjTest(unittest.TestCase):
    """
        Test orbit adjusters
    """

    def test_multi_lambert(self):

        date = datetime.utcnow()

        target = Satellite(date)
        target.mass = 0.82

        chaser = Chaser(date)
        chaser.mass = 30.0

        R_T_0 = np.array([1622.39, 5305.10, 3717.44])
        V_T_0 = np.array([-7.29977, 0.492357, 2.48318])

        target.abs_state.R = R_T_0
        target.abs_state.V = V_T_0

        chaser.rel_state.R = np.array([ -9.38182562e-01,  -9.99760428e-01,   2.92821323e-13])
        chaser.rel_state.V = np.array([  1.16197802e-07,   1.62816739e-03,   5.55477431e-16])
        chaser.abs_state.from_lvlh_frame(target.abs_state, chaser.rel_state)

        target.prop.initialize_propagator('target', target.get_osc_oe(), '2-body')
        chaser.prop.initialize_propagator('chaser', chaser.get_osc_oe(), '2-body')

        checkpoint = RelativeCP()
        checkpoint.rel_state.R = np.array([0.0, 1.0, 0.0])
        checkpoint.rel_state.V = np.array([0.0, 0.0, 0.0])
        checkpoint.t_min = 7979
        checkpoint.t_max = 7980

        orb_adj = MultiLambert()
        man = orb_adj.evaluate_manoeuvre(chaser, checkpoint, target, [0.0, 0.0, 0.0], True)

        dt = (man[1].execution_epoch - man[0].execution_epoch).total_seconds()

        self.assertLess(abs(chaser.rel_state.R[0]), 1e-2)
        self.assertLess(abs(chaser.rel_state.R[1] - 1.0), 1e-2)
        self.assertLess(abs(chaser.rel_state.R[2]), 1e-2)

    def test_clohessy_wiltshire(self):

        date = datetime.utcnow()

        target = Satellite(date)
        target.mass = 0.82

        chaser = Chaser(date)
        chaser.mass = 30.0

        R_T_0 = np.array([1622.39, 5305.10, 3717.44])
        V_T_0 = np.array([-7.29977, 0.492357, 2.48318])

        target.abs_state.R = R_T_0
        target.abs_state.V = V_T_0

        chaser.rel_state.R = np.array([ -9.38182562e-01,  -9.99760428e-01,   2.92821323e-13])
        chaser.rel_state.V = np.array([  1.16197802e-07,   1.62816739e-03,   5.55477431e-16])
        chaser.abs_state.from_lvlh_frame(target.abs_state, chaser.rel_state)

        target.prop.initialize_propagator('target', target.get_osc_oe(), '2-body')
        chaser.prop.initialize_propagator('chaser', chaser.get_osc_oe(), '2-body')

        checkpoint = RelativeCP()
        checkpoint.rel_state.R = np.array([0.0, 1.0, 0.0])
        checkpoint.rel_state.V = np.array([0.0, 0.0, 0.0])
        checkpoint.t_min = 7979
        checkpoint.t_max = 7980

        orb_adj = ClohessyWiltshire()
        man = orb_adj.evaluate_manoeuvre(chaser, checkpoint, target)

        dt = (man[1].execution_epoch - man[0].execution_epoch).total_seconds()

        self.assertLess(abs(chaser.rel_state.R[0]), 1e-2)
        self.assertLess(abs(chaser.rel_state.R[1] - 1.0), 1e-2)
        self.assertLess(abs(chaser.rel_state.R[2]), 1e-2)

if __name__ == '__main__':
    unittest.main()

