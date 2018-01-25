# Note: these tests are quite preliminary....

import unittest
import sys
import os
from copy import deepcopy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../src/")  # hack...
from solver import *
from space_tf import *
from scenario import *

class SolverTest(unittest.TestCase):

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

            print "a initial: " + str(OE_i.a)
            print "e initial: " + str(OE_i.e)
            print "O initial: " + str(OE_i.O*180.0/np.pi)
            print "i initial: " + str(OE_i.i*180.0/np.pi)
            print "w initial: " + str(OE_i.w*180.0/np.pi)
            print "v initial: " + str(OE_i.v*180.0/np.pi)

            random_vector = random_scaling * np.random.random_sample([6]) + random_min

            # generate random final orbital element
            OE_f = KepOrbElem()
            OE_f.a = OE_i.a
            OE_f.e = OE_i.e
            OE_f.O = OE_i.O + np.random.random()*np.pi/18.0 - np.pi/36.0
            OE_f.w = OE_i.w
            OE_f.i = OE_i.i + np.random.random()*np.pi/18.0 - np.pi/36.0
            OE_f.v = random_vector[5]

            print "O final: " + str(OE_f.O*180.0/np.pi)
            print "i final: " + str(OE_f.i*180.0/np.pi)

            chaser_i = Position()
            chaser_i.kep = OE_i
            chaser_i.update_target_from_keporb()

            chaser_f = Position()
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

