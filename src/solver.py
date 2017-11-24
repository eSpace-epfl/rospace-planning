"""
All the solver used, depending on the distance from the target.

References:
    [1]:
    [2]: Zhaohui Dang, Solutions of Tschauner-Hempel Equations, Journal of Guidance, Control and Dynamics, 2017
"""

import numpy as np
import scipy.io as sio
import pykep as pk

from numpy import cos, sin, pi
from space_tf.Constants import Constants as const
from space_tf import Cartesian, CartesianLVLH
from optimal_transfer_time import OptimalTime


class Solver:

    def __init__(self):
        self.tolerance = 0.0
        self.name = 'Std Lambert Problem solver'

        self.sol = None

    def multi_lambert_solver(self, chaser, chaser_next, target):

        print "\n -------------Solving Multi-Lambert Problem nr. " + str(chaser.id) + "--------------- \n"

        mu = const.mu_earth

        t_opt = OptimalTime()

        # TODO: Think about minimum time, at least the amount of time it takes to make a full orbit?
        t_min = 600
        t_max = chaser.execution_time

        # Absolute position of chaser at t = t0
        p_C_TEM_t0 = chaser.cartesian.R
        v_C_TEM_t0 = chaser.cartesian.V

        # Absolute position of target at t = t0
        p_T_TEM_t0 = target.cartesian.R
        v_T_TEM_t0 = target.cartesian.V

        best_deltaV = 1e12
        best_dt = 0
        best_sol = None

        print "Evaluating optimal transfer time... "
        t_opt.find_optimal_trajectory_time_for_scenario(target, chaser_next, chaser, t_max)

        t_optimal = t_opt.t_opt
        print "t optimal: " + str(t_optimal)

        t_min = t_optimal - 20 if t_optimal - 20 > 0 else 1

        print "\n Chaser velocity at the beginning: " + str(chaser.cartesian.V)

        for dt in xrange(t_min, t_optimal + 20):
            # Propagate target position at t = t0 + dt
            p_T_TEM_t1, v_T_TEM_t1 = pk.propagate_lagrangian(p_T_TEM_t0, v_T_TEM_t0, dt, mu)
            target.cartesian.R = p_T_TEM_t1
            target.cartesian.V = v_T_TEM_t1

            # Now that the target is propagated, we can calculate absolute position of the chaser from its relative
            # This is the position he will have at time t = t0 + dt
            chaser_next.cartesian.from_lhlv_frame(target.cartesian, chaser_next.lvlh)

            p_C_TEM_t1 = chaser_next.cartesian.R

            sol = pk.lambert_problem(p_C_TEM_t0, p_C_TEM_t1, dt, mu, True, 10)

            # Check for the best solution for this dt
            for i in xrange(0, len(sol.get_v1())):
                deltaV_1 = np.array(sol.get_v1()[i]) - chaser.cartesian.V
                deltaV_2 = np.array(sol.get_v2()[i]) - chaser_next.cartesian.V

                deltaV_tot = np.linalg.norm(deltaV_1) + np.linalg.norm(deltaV_2)

                if deltaV_tot < best_deltaV:
                    best_deltaV = deltaV_tot
                    best_deltaV_1 = deltaV_1
                    best_deltaV_2 = deltaV_2
                    best_dt = dt
                    best_sol = sol

        # Update ideal final target position at t1 = t0 + best_dt
        p_T_TEM_t1, v_T_TEM_t1 = pk.propagate_lagrangian(p_T_TEM_t0, v_T_TEM_t0, best_dt, mu)
        target.cartesian.R = p_T_TEM_t1
        target.cartesian.V = v_T_TEM_t1

        # Update next chaser position
        chaser_next.cartesian.from_lhlv_frame(target.cartesian, chaser_next.lvlh)

        print "Chaser final relative position: " + str(chaser_next.lvlh.R)

        print "Saving manoeuvre..."

        target_temp = Cartesian()
        chaser_temp = Cartesian()

        chaser_temp_lvlh = CartesianLVLH()

        r_abs = [p_C_TEM_t0]
        r_rel = [chaser.lvlh.R]
        chaser_temp.R = p_C_TEM_t0
        chaser_temp.V = v_C_TEM_t0 + best_deltaV_1
        target_temp.R = p_T_TEM_t0
        target_temp.V = v_T_TEM_t0
        for j in xrange(1, best_dt+1):
            r1, v1 = pk.propagate_lagrangian(chaser_temp.R, chaser_temp.V, 1, mu)
            r_abs.append(r1)

            chaser_temp.R = np.array(r1)
            chaser_temp.V = np.array(v1)

            r1_T, v1_T = pk.propagate_lagrangian(target_temp.R, target_temp.V, 1, mu)

            target_temp.R = np.array(r1_T)
            target_temp.V = np.array(v1_T)

            chaser_temp_lvlh.from_cartesian_pair(chaser_temp, target_temp)

            r_rel.append(chaser_temp_lvlh.R)

        print "Chaser propagated final position: " + str(chaser_temp_lvlh.R)

        sio.savemat('/home/dfrey/polybox/manoeuvre/ml_maneouvre_' + str(chaser.id) + '.mat',
                    mdict={'abs_pos': r_abs, 'rel_pos': r_rel})

        self.sol = {'deltaV': best_deltaV, 'deltaV_1': best_deltaV_1, 'deltaV_2': best_deltaV_2, 'deltaT': best_dt}



    def clohessy_wiltshire_solver(self, a_c, r_rel_c_0, v_rel_c_0, max_time,
                                  r_rel_t_f=np.array([0.0, 0.0, 0.0]), v_rel_t_f=np.array([0.0, 0.0, 0.0]),
                                  id=0, ko_zone=0):

        print "\n -------------Solving CW-equations--------------- \n"

        # TODO: Try to implement a version for continuous thrusting, maybe putting v_0_A dependant on time
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

        self.sol = {'deltaV': best_deltaV, 'deltaV_1': best_deltaV_1, 'deltaV_2': best_deltaV_2, 'deltaT': delta_T}

    def tschauner_hempel_solver(self):
        """
        T-H-equations solver implemented according to [2].
        :return:
        """
        pass
