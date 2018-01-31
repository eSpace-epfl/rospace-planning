# Note: these tests are quite preliminary....

import unittest
import sys
import os
from copy import deepcopy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../src/")  # hack...
from solver import Solver
from space_tf import KepOrbElem
from scenario import Scenario
from state import State

class SolverTest(unittest.TestCase):

    def test_perigee_correction(self):
        """ Test perigee correction manoeuvre.
        """

        solver = Solver()

        num_tests = int(1e4)

        a_min = 6700
        a_max = 90000
        e_min = 0.0
        e_max = -20
        rad_min = 0.0
        rad_max = np.pi*1.999

        random_max = np.array([a_max, e_max, rad_max, rad_max, rad_max/2.0, rad_max])
        random_min = np.array([a_min, e_min, rad_min, rad_min, rad_min, rad_min])
        random_scaling = random_max - random_min

        # perform random tests
        for i in range(1, num_tests):
            random_vector = random_scaling * np.random.random_sample([6]) + random_min

            # generate random initial orbital element
            OE_i = KepOrbElem()
            OE_i.a = random_vector[0]
            OE_i.e = np.exp(random_vector[1])
            OE_i.O = random_vector[2]
            OE_i.w = random_vector[3]
            OE_i.i = random_vector[4]
            OE_i.v = random_vector[5]

            random_vector = random_scaling * np.random.random_sample([6]) + random_min

            # generate random final orbital element
            OE_f = KepOrbElem()
            OE_f.a = OE_i.a
            OE_f.e = OE_i.e
            OE_f.O = OE_i.O
            OE_f.w = random_vector[3]
            OE_f.i = OE_i.i
            OE_f.v = random_vector[5]

            chaser_i = State()
            chaser_i.kep = OE_i

            chaser_f = State()
            chaser_f.kep = OE_f

            solver.adjust_perigee(chaser_i, chaser_f, None)

            self.assertAlmostEqual(chaser_i.kep.i, chaser_f.kep.i, 3)
            self.assertAlmostEqual(chaser_i.kep.O, chaser_f.kep.O, 3)
            self.assertAlmostEqual(chaser_i.kep.a, chaser_f.kep.a, 2)
            self.assertAlmostEqual(chaser_i.kep.e, chaser_f.kep.e, 3)
            self.assertAlmostEqual(chaser_i.kep.w, chaser_f.kep.w, 3)

    def test_apoapsis_periapsis_correction(self):
        """ Test apoapsis/periapsis correction manoeuvres.
        """

        solver = Solver()

        num_tests = int(1e4)

        a_min = 6700
        a_max = 90000
        e_min = 0.0
        e_max = -20
        rad_min = 0.0
        rad_max = np.pi*1.999

        random_max = np.array([a_max, e_max, rad_max, rad_max, rad_max/2.0, rad_max])
        random_min = np.array([a_min, e_min, rad_min, rad_min, rad_min, rad_min])
        random_scaling = random_max - random_min

        # perform random tests
        for i in range(1, num_tests):
            random_vector = random_scaling * np.random.random_sample([6]) + random_min

            # generate random initial orbital element
            OE_i = KepOrbElem()
            OE_i.a = random_vector[0]
            OE_i.e = np.exp(random_vector[1])
            OE_i.O = random_vector[2]
            OE_i.w = random_vector[3]
            OE_i.i = random_vector[4]
            OE_i.v = random_vector[5]

            random_vector = random_scaling * np.random.random_sample([6]) + random_min

            # generate random final orbital element
            OE_f = KepOrbElem()
            OE_f.a = random_vector[0]
            OE_f.e = np.exp(random_vector[1])
            OE_f.O = OE_i.O
            OE_f.w = OE_i.w
            OE_f.i = OE_i.i
            OE_f.v = random_vector[5]

            chaser_i = State()
            chaser_i.kep = OE_i

            chaser_f = State()
            chaser_f.kep = OE_f

            solver.adjust_eccentricity_semimajoraxis(chaser_i, chaser_f, None)

            self.assertAlmostEqual(chaser_i.kep.i, chaser_f.kep.i, 3)
            self.assertAlmostEqual(chaser_i.kep.O, chaser_f.kep.O, 3)
            self.assertAlmostEqual(chaser_i.kep.a, chaser_f.kep.a, 2)
            self.assertAlmostEqual(chaser_i.kep.e, chaser_f.kep.e, 3)
            self.assertAlmostEqual(chaser_i.kep.w, chaser_f.kep.w, 3)

    def test_plane_correction(self):
        """ Test plane correction manoeuvre.
        """

        solver = Solver()

        num_tests = int(1e4)

        a_min = 6700
        a_max = 90000
        e_min = 0.0
        e_max = -20
        rad_min = 0.0
        rad_max = np.pi*1.999

        random_max = np.array([a_max, e_max, rad_max, rad_max, rad_max/2.0, rad_max])
        random_min = np.array([a_min, e_min, rad_min, rad_min, rad_min, rad_min])
        random_scaling = random_max - random_min

        # perform random tests
        for i in range(1, num_tests):
            random_vector = random_scaling * np.random.random_sample([6]) + random_min

            # generate random initial orbital element
            OE_i = KepOrbElem()
            OE_i.a = random_vector[0]
            OE_i.e = np.exp(random_vector[1])
            OE_i.O = random_vector[2]
            OE_i.w = random_vector[3]
            OE_i.i = random_vector[4]
            OE_i.v = random_vector[5]

            random_vector = random_scaling * np.random.random_sample([6]) + random_min

            # generate random final orbital element
            OE_f = KepOrbElem()
            OE_f.a = OE_i.a
            OE_f.e = OE_i.e
            OE_f.O = OE_i.O + np.random.random()*np.pi/18.0 - np.pi/36.0
            OE_f.w = OE_i.w
            OE_f.i = OE_i.i + np.random.random()*np.pi/18.0 - np.pi/36.0
            OE_f.v = random_vector[5]

            chaser_i = State()
            chaser_i.kep = OE_i
            chaser_i.update_target_from_keporb()

            chaser_f = State()
            chaser_f.kep = OE_f
            chaser_f.update_target_from_keporb()

            solver.plane_correction(chaser_i, chaser_f, None, False)

            self.assertAlmostEqual(chaser_i.kep.i, chaser_f.kep.i, 6)
            self.assertAlmostEqual(chaser_i.kep.O, chaser_f.kep.O, 6)
            self.assertAlmostEqual(chaser_i.kep.a, OE_f.a, 6)
            self.assertAlmostEqual(chaser_i.kep.e, OE_f.e, 6)
            # self.assertAlmostEqual(chaser_i.kep.w, OE_f.w, 6)

if __name__ == '__main__':
    unittest.main()

