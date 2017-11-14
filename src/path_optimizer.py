from math import sqrt,pi,atan,tan,sin,cos
from space_tf import *
from PyKEP import *
import numpy as np
import rospy


class PathOptimizer:

    def __init__(self):
        # Initialize the optimizer
        self.kep_target = KepOrbElem()
        self.kep_chaser = KepOrbElem()

        self.cart_target = Cartesian()
        self.cart_chaser = Cartesian()

        self.deltaV = np.array([0, 0, 0])
        self.impulse_start_time = rospy.Time.now()

        self.sleep_flag = False
        print "Initialize"

    def find_optimal_path(self, msg):
        # Find optimal path
        print "Searching for optimal path"

    def callback(self, target_oe, chaser_oe):
        # Update orbital elements value
        self.kep_target.from_message(target_oe.position)
        self.cart_target.from_keporb(self.kep_target)

        self.kep_chaser.from_message(chaser_oe.position)
        self.cart_chaser.from_keporb(self.kep_chaser)

        sol = self.simple_pykep_solution(self.cart_target.R, self.cart_target.V, self.cart_chaser.R, self.cart_chaser.V, 400, 20)

        self.deltaV = sol.get_v1()[0]
        self.sleep_flag = True


    def simple_pykep_solution(self, r_t1, v_t1, r_c1, v_c1, t_rdv, t_start):
        mu = Constants.mu_earth

        r_c_start, v_c_start = propagate_lagrangian(r_c1, v_c1, t_start, mu)

        r_t2, v_t2 = propagate_lagrangian(r_t1, v_t1, t_rdv + t_start, mu)

        l = lambert_problem(r_c_start, r_t2, t_rdv)
        return l
