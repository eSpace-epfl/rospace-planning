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
from state import Satellite
from checkpoint import AbsoluteCP

class SolverTest(unittest.TestCase):

    def test_perigee_correction(self):
        """
            Test perigee correction manoeuvre.
        """

        solver = Solver()

        num_tests = int(1e2)

        a_min = 6700
        a_max = 90000
        e_min = 0.0
        e_max = -10.0
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

            chaser_i = Satellite()
            chaser_i.abs_state = OE_i

            chaser_f = Satellite()
            chaser_f.abs_state = OE_f

            print OE_i.a
            print OE_i.e
            print OE_i.i
            print OE_i.O
            print OE_i.w
            print OE_i.v
            print OE_f.a
            print OE_f.e

            solver.adjust_perigee(chaser_i, chaser_f, None)

            self.assertAlmostEqual(chaser_i.abs_state.i, chaser_f.abs_state.i, 3)
            self.assertAlmostEqual(chaser_i.abs_state.O, chaser_f.abs_state.O, 3)
            self.assertAlmostEqual(chaser_i.abs_state.a, chaser_f.abs_state.a, 2)
            self.assertAlmostEqual(chaser_i.abs_state.e, chaser_f.abs_state.e, 3)
            self.assertAlmostEqual(chaser_i.abs_state.w, chaser_f.abs_state.w, 3)

    def test_apoapsis_periapsis_correction(self):
        """
            Test apoapsis/periapsis correction manoeuvres.
        """

        solver = Solver()

        num_tests = int(1e2)

        a_min = 6700
        a_max = 90000
        e_min = 0.0
        e_max = -10.0
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

            chaser_i = Satellite()
            chaser_i.abs_state = OE_i

            chaser_f = Satellite()
            chaser_f.abs_state = OE_f

            print OE_i.a
            print OE_i.e
            print OE_i.i
            print OE_i.O
            print OE_i.w
            print OE_i.v
            print OE_f.a
            print OE_f.e


            solver.adjust_eccentricity_semimajoraxis(chaser_i, chaser_f, None)

            self.assertAlmostEqual(chaser_i.abs_state.i, chaser_f.abs_state.i, 3)
            self.assertAlmostEqual(chaser_i.abs_state.O, chaser_f.abs_state.O, 3)
            self.assertAlmostEqual(chaser_i.abs_state.a, chaser_f.abs_state.a, 2)
            self.assertAlmostEqual(chaser_i.abs_state.e, chaser_f.abs_state.e, 3)
            self.assertAlmostEqual(chaser_i.abs_state.w, chaser_f.abs_state.w, 2)

    def test_plane_correction(self):
        """
            Test plane correction manoeuvre.
        """

        solver = Solver()

        num_tests = int(1e2)

        a_min = 6700
        a_max = 90000
        e_min = 0.0
        e_max = -10.0
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

            if OE_f.i > np.pi:
                OE_f.i = np.pi - 1e-2

            if OE_f.i < 0.0:
                OE_f.i = 1e-2

            if OE_f.O > 2.0 * np.pi:
                OE_f.O = 2.0 * np.pi - 1e-2

            if OE_f.O < 0.0:
                OE_f.O = 1e-2

            chaser = Satellite()
            chaser.set_abs_state(OE_i)

            print "Chaser state: "
            print " >> Kep: "
            print "      a :     " + str(OE_i.a)
            print "      e :     " + str(OE_i.e)
            print "      i :     " + str(OE_i.i)
            print "      O :     " + str(OE_i.O)
            print "      w :     " + str(OE_i.w)
            print "      v :     " + str(OE_i.v)

            print "Checkpoint state: "
            print " >> Kep: "
            print "      a :     " + str(OE_f.a)
            print "      e :     " + str(OE_f.e)
            print "      i :     " + str(OE_f.i)
            print "      O :     " + str(OE_f.O)
            print "      w :     " + str(OE_f.w)
            print "      v :     " + str(OE_f.v)


            checkpoint = AbsoluteCP()
            checkpoint.set_abs_state(OE_f)

            solver.plane_correction(chaser, checkpoint, None)

            self.assertAlmostEqual(chaser.abs_state.i, checkpoint.abs_state.i, 2)
            self.assertAlmostEqual(chaser.abs_state.O, checkpoint.abs_state.O, 2)
            self.assertAlmostEqual(chaser.abs_state.a, OE_f.a, 2)
            self.assertAlmostEqual(chaser.abs_state.e, OE_f.e, 2)
            # self.assertAlmostEqual(chaser.abs_state.w, OE_f.w, 1)

if __name__ == '__main__':
    unittest.main()

