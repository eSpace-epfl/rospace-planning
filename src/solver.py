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
from space_tf import Cartesian, CartesianLVLH, mu_earth
from optimal_transfer_time import OptimalTime


class Solver:

    def __init__(self):
        self.sol = None

    def solve(self, chaser, chaser_next, target):
        # TODO: Implement generic solver that decide the right thing to do depending on the distance

        distance = np.linalg.norm(chaser.cartesian.R - target.cartesian.R)

        # TODO: Version 2.0
        # if distance > 10:
        #     self.multi_lambert_solver(chaser, chaser_next, target)
        # else:
        #     self.clohessy_wiltshire_solver(chaser, chaser_next, target)


        # "Working" version only with lambert solver
        self.multi_lambert_solver(chaser, chaser_next, target)

    def multi_lambert_solver(self, chaser, chaser_next, target):

        print "\n -------------Solving Multi-Lambert Problem nr. " + str(chaser.id) + "--------------- \n"

        # TODO: Think about minimum time, at least the amount of time it takes to make a full orbit?

        # Absolute position of chaser at t = t0
        p_C_TEM_t0 = chaser.cartesian.R
        v_C_TEM_t0 = chaser.cartesian.V

        # Absolute position of target at t = t0
        p_T_TEM_t0 = target.cartesian.R
        v_T_TEM_t0 = target.cartesian.V

        best_deltaV = 1e12
        best_dt = 0
        best_sol = None

        # if np.linalg.norm(v_C_TEM_t0 - v_T_TEM_t0) != 0:
        #     t_min = int(np.linalg.norm(p_C_TEM_t0 - p_T_TEM_t0) / np.linalg.norm(v_C_TEM_t0 - v_T_TEM_t0))
        # else:
        #     t_min = 1

        t_min = int(chaser.execution_time/2 - 100)
        t_max = int(chaser.execution_time/2 + 100)
        if t_min < 0:
            t_min = 1

        for dt in xrange(t_min, t_max):
            # Propagate target position at t = t0 + dt
            p_T_TEM_t1, v_T_TEM_t1 = pk.propagate_lagrangian(p_T_TEM_t0, v_T_TEM_t0, dt, mu_earth)
            target.cartesian.R = np.array(p_T_TEM_t1)
            target.cartesian.V = np.array(v_T_TEM_t1)

            # Now that the target is propagated, we can calculate absolute position of the chaser from its relative
            # This is the position he will have at time t = t0 + dt
            chaser_next.cartesian.from_lvlh_frame(target.cartesian, chaser_next.lvlh)

            p_C_TEM_t1 = chaser_next.cartesian.R

            # Calculate the maximum number of revolutions given orbit period
            # T = 2*pi*sqrt(chaser.kep.a)

            # TODO: Instead of solving everytime the lambert problem, check if it can be optimal before
            sol = pk.lambert_problem(p_C_TEM_t0, p_C_TEM_t1, dt, mu_earth, True, 10)

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
        p_T_TEM_t1, v_T_TEM_t1 = pk.propagate_lagrangian(p_T_TEM_t0, v_T_TEM_t0, best_dt, mu_earth)
        target.cartesian.R = np.array(p_T_TEM_t1)
        target.cartesian.V = np.array(v_T_TEM_t1)

        # Update next chaser position
        chaser_next.cartesian.from_lvlh_frame(target.cartesian, chaser_next.lvlh)

        print "Chaser final relative position: " + str(chaser_next.lvlh.R)

        # print "Saving manoeuvre..."
        #
        # target_temp = Cartesian()
        # chaser_temp = Cartesian()
        #
        # chaser_temp_lvlh = CartesianLVLH()
        #
        # r_abs = [p_C_TEM_t0]
        # r_rel = [chaser.lvlh.R]
        # chaser_temp.R = p_C_TEM_t0
        # chaser_temp.V = v_C_TEM_t0 + best_deltaV_1
        # target_temp.R = p_T_TEM_t0
        # target_temp.V = v_T_TEM_t0
        # for j in xrange(1, best_dt+1):
        #     r1, v1 = pk.propagate_lagrangian(chaser_temp.R, chaser_temp.V, 1, mu_earth)
        #     r_abs.append(r1)
        #
        #     chaser_temp.R = np.array(r1)
        #     chaser_temp.V = np.array(v1)
        #
        #     r1_T, v1_T = pk.propagate_lagrangian(target_temp.R, target_temp.V, 1, mu_earth)
        #
        #     target_temp.R = np.array(r1_T)
        #     target_temp.V = np.array(v1_T)
        #
        #     chaser_temp_lvlh.from_cartesian_pair(chaser_temp, target_temp)
        #
        #     r_rel.append(chaser_temp_lvlh.R)
        #
        # sio.savemat('/home/dfrey/polybox/manoeuvre/ml_maneouvre_' + str(chaser.id) + '.mat',
        #             mdict={'abs_pos': r_abs, 'rel_pos': r_rel, 'deltaV_1': best_deltaV_1, 'deltaV_2': best_deltaV_2})

        self.sol = {'deltaV': best_deltaV, 'deltaV_1': best_deltaV_1, 'deltaV_2': best_deltaV_2, 'deltaT': best_dt}

    def clohessy_wiltshire_solver(self, chaser, chaser_next, target):

        print "\n -------------Solving CW-equations--------------- \n"
        print " Useful only for really close operations, "

        chaser.kep.from_cartesian(chaser.cartesian)
        chaser_next.kep.from_cartesian(chaser_next.cartesian)
        target.kep.from_cartesian(target.cartesian)

        a = target.kep.a
        max_time = chaser.execution_time
        id = chaser.id

        r_rel_c_0 = chaser.lhlv.R
        v_rel_c_0 = chaser.lhlv.V

        r_rel_c_n = chaser_next.lhlv.R
        v_rel_c_n = chaser_next.lhlv.V


        # TODO: Try to implement a version for continuous thrusting, maybe putting v_0_A dependant on time
        # TODO: Check with HP relative velocity, if we can move to the next hold point easily by "using" the relative velocity already acquired.

        n = np.sqrt(mu_earth/a**3.0)

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
                deltaV_1 = np.dot(np.linalg.inv(rv_t), r_rel_c_n - np.dot(phi_rr(t_), r_rel_c_0)) - v_rel_c_0
                deltaV_2 = v_rel_c_n - np.dot(phi_vr(t_), r_rel_c_0) - np.dot(phi_vv(t_), v_rel_c_0 + deltaV_1)

                deltaV_tot = np.linalg.norm(deltaV_1) + np.linalg.norm(deltaV_2)

                # TODO: If thrust is really really low, it could mean that we may reach the target only by waiting
                if best_deltaV > deltaV_tot and any(abs(deltaV_1[i]) >= min_deltaV for i in range(0, 3))\
                        and any(abs(deltaV_2[i]) >= min_deltaV for i in range(0, 3)):
                    # Check if the keep out zone is invaded and if we are not approaching it
                    # if id != 1:
                    #     for t_test in xrange(0, t_ + 1):
                    #         r_test = np.dot(phi_rr(t_test), r_rel_c_0) + np.dot(phi_rv(t_test), v_rel_c_0 + deltaV_1)
                    #         if all(abs(r_test[i]) >= ko_zone for i in range(0, 3)):
                    #             best_deltaV = deltaV_tot
                    #             best_deltaV_1 = deltaV_1
                    #             best_deltaV_2 = deltaV_2
                    #             delta_T = t_
                    best_deltaV = deltaV_tot
                    best_deltaV_1 = deltaV_1
                    best_deltaV_2 = deltaV_2
                    delta_T = t_

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

    def solve_between_rel_orbits(self, chaser, chaser_next, target):
        pass



    def save_result(self, chaser, chaser_next, target):
        # TODO: Function to store and save results and manoeuvres in .mat format
        print "Saving manoeuvre..."

        # target_temp = Cartesian()
        # chaser_temp = Cartesian()
        #
        # chaser_temp_lvlh = CartesianLVLH()
        #
        # r_abs = [p_C_TEM_t0]
        # r_rel = [chaser.lvlh.R]
        # chaser_temp.R = p_C_TEM_t0
        # chaser_temp.V = v_C_TEM_t0 + best_deltaV_1
        # target_temp.R = p_T_TEM_t0
        # target_temp.V = v_T_TEM_t0
        #
        # for j in xrange(1, best_dt+1):
        #     r1, v1 = pk.propagate_lagrangian(chaser_temp.R, chaser_temp.V, 1, mu_earth)
        #     r_abs.append(r1)
        #
        #     chaser_temp.R = np.array(r1)
        #     chaser_temp.V = np.array(v1)
        #
        #     r1_T, v1_T = pk.propagate_lagrangian(target_temp.R, target_temp.V, 1, mu_earth)
        #
        #     target_temp.R = np.array(r1_T)
        #     target_temp.V = np.array(v1_T)
        #
        #     chaser_temp_lvlh.from_cartesian_pair(chaser_temp, target_temp)
        #
        #     r_rel.append(chaser_temp_lvlh.R)
        #
        # sio.savemat('/home/dfrey/polybox/manoeuvre/ml_maneouvre_' + str(chaser.id) + '.mat',
        #             mdict={'abs_pos': r_abs, 'rel_pos': r_rel, 'deltaV_1': best_deltaV_1, 'deltaV_2': best_deltaV_2})        target_temp = Cartesian()
        #
