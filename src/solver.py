"""
All the solver used, depending on the distance from the target.

References:
    [1]:
    [2]: Zhaohui Dang, Solutions of Tschauner-Hempel Equations, Journal of Guidance, Control and Dynamics, 2017
"""

import numpy as np
import scipy.io as sio

from numpy import cos, sin, pi
from space_tf.Constants import Constants as const


class Solver:

    def __init__(self):
        self.tolerance = 0.0
        self.name = 'Std Lambert Problem solver'

        self.cw_sol = None




    def clohessy_wiltshire_solver(self, a_c, r_rel_c_0, v_rel_c_0, max_time,
                                  r_rel_t_f=np.array([0.0, 0.0, 0.0]), v_rel_t_f=np.array([0.0, 0.0, 0.0]),
                                  id=0, ko_zone=0):

        # TODO: Check with HP relative velocity, if we can move to the next hold point easily by "using" the relative velocity already acquired.
        mu = const.mu_earth
        n = np.sqrt(mu/a_c**3.0)

        phi_rr = lambda t: np.array([
            [4.0 - 3.0*cos(n*t), 0.0, 0.0],
            [6.0*(sin(n*t) - n*t), 1.0, 0.0],
            [0.0, 0.0, cos(n*t)]
        ])

        phi_rv = lambda t: np.array([
            [1.0/n * sin(n*t), 2.0/n * (1 - cos(n*t)), 0.0],
            [2.0/n * (cos(n*t) - 1.0), 1.0/n * (4.0*sin(n*t) - 3.0*n*t), 0.0],
            [0.0, 0.0, 1.0/n * sin(n*t)]
        ])

        phi_vr = lambda t: np.array([
            [3.0*n*sin(n*t), 0.0, 0.0],
            [6.0*n * (cos(n*t) - 1), 0.0, 0.0],
            [0.0, 0.0, -n * sin(n*t)]
        ])

        phi_vv = lambda t: np.array([
            [cos(n*t), 2.0*sin(n*t), 0.0],
            [-2.0*sin(n*t), 4.0*cos(n*t) - 3.0, 0.0],
            [0.0, 0.0, cos(n*t)]
        ])

        best_deltaV = 1e12
        delta_T = 0

        # 1mm/sec accuracy. TODO: Check the accuracy of the thrusters!
        min_deltaV = 1e-6

        deltaT = range(0, max_time, 1)
        for t_ in deltaT:
            rv_t = phi_rv(t_)
            det_rv = np.linalg.det(rv_t)

            if det_rv != 0:
                deltaV_1 = np.dot(np.linalg.inv(rv_t), r_rel_t_f - np.dot(phi_rr(t_), r_rel_c_0)) - v_rel_c_0
                deltaV_2 = v_rel_t_f - np.dot(phi_vr(t_), r_rel_c_0) - np.dot(phi_vv(t_), v_rel_c_0 + deltaV_1)

                deltaV_tot = np.linalg.norm(deltaV_1) + np.linalg.norm(deltaV_2)

                # TODO: If thrust is really really low, it could mean that we may reach the target only by waiting
                if best_deltaV > deltaV_tot and any(abs(deltaV_1[i]) >= min_deltaV for i in range(0, 3))\
                        and any(abs(deltaV_2[i]) >= min_deltaV for i in range(0, 3)):
                    # Check if the keep out zone is invaded and if we are not approaching it
                    if id != 1:
                        for t_test in xrange(0, t_ + 1):
                            r_test = np.dot(phi_rr(t_test), r_rel_c_0) + np.dot(phi_rv(t_test), v_rel_c_0 + deltaV_1)
                            if all(abs(r_test[i]) >= ko_zone for i in range(0, 3)):
                                best_deltaV = deltaV_tot
                                best_deltaV_1 = deltaV_1
                                best_deltaV_2 = deltaV_2
                                delta_T = t_
                    #
                    # best_deltaV = deltaV_tot
                    # best_deltaV_1 = deltaV_1
                    # best_deltaV_2 = deltaV_2
                    # delta_T = t_

        T = np.arange(0, delta_T+1, 1)

        r = np.dot(phi_rr(T), r_rel_c_0) + np.dot(phi_rv(T),  v_rel_c_0 + best_deltaV_1)

        print "Saving manoeuvre " + str(id)

        sio.savemat('/home/dfrey/polybox/manoeuvre/maneouvre_'+str(id)+'.mat', mdict={'rel_pos': r})

        self.cw_sol = {'deltaV': best_deltaV, 'deltaV_1': best_deltaV_1, 'deltaV_2': best_deltaV_2, 'deltaT': delta_T}


    def tschauner_hempel_solver(self):
        """
        T-H-equations solver implemented according to [2].
        :return:
        """
        pass
