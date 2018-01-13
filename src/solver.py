"""
    Module that contains a definition of the solver, that calculates in real time the next manouver to be done.
    To be reviewed.

    References:
        [1] Orbital Mechanics for Engineering Students, 3rd Edition, Howard D. Curtis, ISBN 978-0-08-097747-8
        [2] Fundamentals of Astrodynamics and Applications, 2nd Edition, David A. Vallado, ISBN 1-881883-12-4
"""

import numpy as np
import scipy.io as sio
import pykep as pk
import os

from space_tf import Cartesian, CartesianLVLH, mu_earth, KepOrbElem, R_earth
from scenario import Position

import matplotlib.pyplot as plt

class Command:

    def __init__(self):
        self.deltaV_C = [0, 0, 0]
        self.ideal_transfer_orbit = []
        self.duration = 0

        self.true_anomaly = 0
        self.mean_anomaly = 0
        self.epoch = 0

        self.theta_diff = None

        self.description = None

class Solver:

    def __init__(self):
        self.command_line = []
        self.solver_clock = 0

    def solve_scenario(self, scenario, chaser, target):
        print "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        print "Solving the scenario: " + scenario.name + "\n"
        print "Scenario overview: "
        print scenario.overview
        print "Mission start as soon as the simulation start! "

        chaser_old = Position()
        chaser_old.from_other_position(chaser)
        target_old = Position()
        target_old.from_other_position(target)

        # Extract scenario positions
        positions = scenario.checkpoints

        # Define total deltav and total time
        tot_time = 0
        tot_deltaV = 0

        print '\n >>> Initial state:'
        self.print_state(chaser, target)

        # Start solving scenario by popping positions from position list
        i = 0
        while i < len(positions):
            print "\n\n>>>> EVALUATING MANOEUVRE " + str(i)

            # Extract the first position, at t = t_start
            chaser_next = positions[i].position
            time_dependancy = positions[i].time_dependancy
            error = positions[i].error_ellipsoid

            self.solve(chaser, chaser_next, target, time_dependancy, error)

            i += 1

        tot_dV, tot_dt = self.print_result()

        print "\n\n-----------------> Manoeuvre elaborated <--------------------\n"
        print "---> Manoeuvre duration:    " + str(tot_dt) + " seconds"
        print "---> Total deltaV:          " + str(tot_dV) + " km/s"

        scenario.export_solved_scenario(self.command_line)

        self._save_result(chaser_old, target_old)

    def solve(self, chaser, chaser_next, target, time_dependancy, error):

        tol = 1e-7

        if time_dependancy:
            # Check if the chaser can drift inside a certain ellipsoid around the checkpoint under some t_max
            if np.linalg.norm(error) != 0:
                # Drift allowed for this point
                t_est = self.drift_to(chaser, chaser_next, target, error)
                t_limit = 600000.0

                # If the target cannot drift easily to the wanted position, and if we are still somewhere where CW equations
                # cannot be used.
                # if t_est is None and np.linalg.norm(chaser.lvlh.R) > 10.0:
                #     # TODO: Drift BEFORE changing orbit, s.t the wait time is much lower
                #
                #     # Chaser cannot drift to the wanted position, has to be adjusted to another orbit to be able to drift
                #     # Evaluate wanted radius difference to move to a coelliptic orbit with that difference
                #     r_diff = chaser_next.lvlh.R[0]
                #
                #     # Create new checkpoint
                #     chaser_new_next = Position()
                #     chaser_new_next.from_other_position(chaser_next)
                #     chaser_new_next.kep.a = target.kep.a + r_diff
                #     chaser_new_next.kep.e = target.kep.a * target.kep.e / chaser_new_next.kep.a
                #
                #     # Adjust orbit though a standard manoeuvre
                #     self.adjust_eccentricity_semimajoraxis(chaser, chaser_new_next, target)
                #     self.print_state(chaser, target)
                #
                #     # Evaluate the new drift time
                #     t_est = self.drift_to(chaser, chaser_new_next, target, error)
                #
                # el

                if t_est is None: # and np.linalg.norm(chaser.lvlh.R) <= 20.0:
                    # Distance from the target is below 10.0 km => use CW-solver
                    # self.clohessy_wiltshire_solver(chaser, chaser_next, target)

                    self.multi_lambert(chaser, chaser_next, target)

                # Check if the drift time is below a certain limit
                if t_est is not None and t_est <= t_limit:
                    # Drift, propagate chaser and target for t_est
                    self._propagator(chaser, target, t_est)
                    self.print_state(chaser, target)

                    # Add drift command to the command line
                    c = Command()
                    c.deltaV_C = np.array([0.0, 0.0, 0.0])
                    c.true_anomaly = chaser.kep.v
                    c.ideal_transfer_orbit = []
                    c.duration = t_est
                    c.description = 'Drift for ' + str(t_est) + ' seconds'

                elif t_est is not None and t_est > t_limit:
                    # TODO: Do a resynchronisation manoeuvre
                    pass
            else:
                # Drift not allowed
                self.multi_lambert(chaser, chaser_next, target)

        else:
            # Correct for the plane angles,
            # Inclination
            di = chaser_next.kep.i - chaser.kep.i
            dO = chaser_next.kep.O - chaser.kep.O
            if abs(di) > tol or abs(dO) > tol:
                # Relative inclination between chaser and chaser next has to be adjusted
                self.plane_correction(chaser, chaser_next, target)
                self.print_state(chaser, target)

            # Perigee
            dw = chaser_next.kep.w - chaser.kep.w
            if abs(dw) > 1e-3:
                # Relative argument of perigee between chaser and chaser next has to be adjusted.
                self.adjust_perigee(chaser, chaser_next, target)
                self.print_state(chaser, target)

            # Eccentricity and Semi-Major Axis
            da = chaser_next.kep.a - chaser.kep.a
            de = chaser_next.kep.e - chaser.kep.e
            if abs(da) > 0.2 or abs(de) > 1e-3:
                self.adjust_eccentricity_semimajoraxis(chaser, chaser_next, target)
                self.print_state(chaser, target)

    def adjust_eccentricity_semimajoraxis(self, chaser, chaser_next, target):
        """
            Adjust eccentricity and semi-major axis at the same time with an Hohmann-Transfer like manouevre:
            1) Burn at perigee to match the needed intermediate orbit
            2) Burn at apogee to arrive at the final, wanted orbit

            [1] Chapter 6

        Args:
            chaser (Position)
            chaser_next (Position)
            target (Position)
        """

        # TODO: Think about circular orbit true anomaly... And where the burn is executed...

        # When this solver is called, only the first two relative orbital elements have to be aligned with
        # the requested one.
        a_i = chaser.kep.a
        e_i = chaser.kep.e
        a_f = chaser_next.kep.a
        e_f = chaser_next.kep.e

        # Check if the orbits intersecate
        t = (a_f*(1-e_f**2) - a_i*(1-e_i**2))/(a_i*e_f*(1-e_i)**2 - a_f*e_i*(1-e_f**2))

        if abs(t) <= 1:
            # Orbit intersecate, burn at the intersection
            theta_i = np.arccos(t)
            theta_i_tmp = 2.0 * np.pi - theta_i

            if theta_i < chaser.kep.v:
                dv1 = 2*np.pi + theta_i - chaser.kep.v
            else:
                dv1 = theta_i - chaser.kep.v

            if theta_i_tmp < chaser.kep.v:
                dv2 = 2*np.pi + theta_i_tmp - chaser.kep.v
            else:
                dv2 = theta_i_tmp - chaser.kep.v

            if dv1 > dv2:
                theta_i = theta_i_tmp

            V_PERI_i = np.sqrt(mu_earth / (a_i * (1.0 - e_i**2))) * np.array([-np.sin(theta_i), e_i + np.cos(theta_i), 0.0])
            V_TEM_i = np.linalg.inv(chaser.kep.get_pof()).dot(V_PERI_i)

            V_PERI_f = np.sqrt(mu_earth / (a_f * (1.0 - e_f**2))) * np.array([-np.sin(theta_i), e_f + np.cos(theta_i), 0.0])
            V_TEM_f = np.linalg.inv(chaser_next.kep.get_pof()).dot(V_PERI_f)

            deltaV_C = V_TEM_f - V_TEM_i

            # Create command
            c = Command()
            c.deltaV_C = deltaV_C
            c.true_anomaly = theta_i
            c.ideal_transfer_orbit = []
            c.duration = self.travel_time(chaser, chaser.kep.v, theta_i)
            c.description = 'Apogee/Perigee raise (intersecating orbits)'
            self.command_line.append(c)

            # Save single manoeuvre...
            self._save_result(chaser, target, len(self.command_line), True)

            # Propagate chaser and target to evaluate all the future commands properly
            self._propagator(chaser, target, c.duration)
            chaser.cartesian.V += deltaV_C

        else:
            if a_f > a_i:
                r_a_f = a_f * (1.0 + e_f)
                r_p_i = a_i * (1.0 - e_i)

                # Calculate intermediate orbital elements
                a_int = (r_a_f + r_p_i) / 2.0
                e_int = 1.0 - r_p_i / a_int

                # First burn at perigee, then apogee
                theta_1 = 0.0
                theta_2 = np.pi
            else:
                r_a_i = a_i * (1.0 + e_i)
                r_p_f = a_f * (1.0 - e_f)

                # Calculate intermediate orbital elements
                a_int = (r_a_i + r_p_f) / 2.0
                e_int = 1.0 - r_p_f / a_int

                # First burn at apogee, then perigee
                theta_1 = np.pi
                theta_2 = 0.0


            # Calculate Delta-V's in perifocal frame of reference
            V_PERI_i_1 = np.sqrt(mu_earth / (a_i * (1.0 - e_i**2))) * np.array([-np.sin(theta_1), e_i + np.cos(theta_1), 0.0])
            V_TEM_i_1 = np.linalg.inv(chaser.kep.get_pof()).dot(V_PERI_i_1)
            V_PERI_f_1 = np.sqrt(mu_earth / (a_int * (1.0 - e_int**2))) * np.array([-np.sin(theta_1), e_int + np.cos(theta_1), 0.0])
            V_TEM_f_1 = np.linalg.inv(chaser.kep.get_pof()).dot(V_PERI_f_1)

            deltaV_C_1 = V_TEM_f_1 - V_TEM_i_1

            V_PERI_i_2 = np.sqrt(mu_earth / (a_int * (1.0 - e_int ** 2))) * np.array([-np.sin(theta_2), e_int + np.cos(theta_2), 0.0])
            V_TEM_i_2 = np.linalg.inv(chaser.kep.get_pof()).dot(V_PERI_i_2)
            V_PERI_f_2 = np.sqrt(mu_earth / (a_f * (1.0 - e_f ** 2))) * np.array([-np.sin(theta_2), e_f + np.cos(theta_2), 0.0])
            V_TEM_f_2 = np.linalg.inv(chaser.kep.get_pof()).dot(V_PERI_f_2)

            deltaV_C_2 = V_TEM_f_2 - V_TEM_i_2

            # Check if a specific relative distance w.r.t the target is wanted.
            # In that case compute the waiting time on the intermediate orbit, check if it's possible to achieve the wanted
            # position. If not, adjust the burn to achieve phasing.
            # dtheta = target.kep.v - chaser_next.kep.v
            # synodal_period = self.calc_synodic_period(chaser_next, target)
            # Check after how much time such a configuration will happen again
            # Wait till the configuration is reached
            # Do the second (if wanted) burn

            # Create command
            c1 = Command()
            c1.deltaV_C = deltaV_C_1
            c1.true_anomaly = theta_1
            c1.ideal_transfer_orbit = []
            c1.duration = self.travel_time(chaser, chaser.kep.v, theta_1)
            c1.description = 'Apogee/Perigee raise'
            self.command_line.append(c1)

            # Save single manoeuvre...
            self._save_result(chaser, target, len(self.command_line), True)

            # Propagate chaser and target to evaluate all the future commands properly
            self._propagator(chaser, target, c1.duration)
            chaser.cartesian.V += deltaV_C_1

            c2 = Command()
            c2.deltaV_C = deltaV_C_2
            c2.true_anomaly = theta_2
            c2.ideal_transfer_orbit = []
            c2.duration = np.pi * np.sqrt(a_int**3 / mu_earth)
            c2.description = 'Apogee/Perigee raise'
            self.command_line.append(c2)

            # Save single manoeuvre...
            self._save_result(chaser, target, len(self.command_line), True)

            # Propagate chaser and target to evaluate all the future commands properly
            self._propagator(chaser, target, c2.duration)
            chaser.cartesian.V += deltaV_C_2

    def adjust_perigee(self, chaser, chaser_next, target):
        """
            Given the chaser relative orbital elements w.r.t target,
            ad its next wanted status correct the perigee argument.

        Args:
              chaser (Position):
              chaser_next (Position):
        """

        # Easy fix for the case of circular orbit:
        if chaser.kep.e < 1e-8:
            # When the orbit is almost circular approximate the argument of perigee to be at 0
            # Approximation to be checked... Bigger orbits may lead to bigger errors
            chaser.kep.v = (chaser.kep.v + chaser.kep.w) % (2.0 * np.pi)
            chaser.kep.w = 0
        else:
            # Extract constants
            a = chaser.kep.a
            e = chaser.kep.e

            # Evaluate perigee difference to correct
            dw = (chaser_next.kep.w - chaser.kep.w) % (2.0 * np.pi)

            # Two possible positions where the burn can occur
            theta_i = dw/2.0
            theta_i_tmp = theta_i + np.pi
            theta_f = 2.0*np.pi - theta_i
            theta_f_tmp = theta_f - np.pi

            # Check which one is the closest
            if theta_i < chaser.kep.v:
                dv1 = 2*np.pi + theta_i - chaser.kep.v
            else:
                dv1 = theta_i - chaser.kep.v

            if theta_i_tmp < chaser.kep.v:
                dv2 = 2*np.pi + theta_i_tmp - chaser.kep.v
            else:
                dv2 = theta_i_tmp - chaser.kep.v

            if dv1 > dv2:
                theta_i = theta_i_tmp
                theta_f = theta_f_tmp

            V_PERI_i = np.sqrt(mu_earth / (a * (1.0 - e**2))) * np.array([-np.sin(theta_i), e + np.cos(theta_i), 0.0])
            V_TEM_i = np.linalg.inv(chaser.kep.get_pof()).dot(V_PERI_i)

            V_PERI_f = np.sqrt(mu_earth / (a * (1.0 - e**2))) * np.array([-np.sin(theta_f), e + np.cos(theta_f), 0.0])
            V_TEM_f = np.linalg.inv(chaser_next.kep.get_pof()).dot(V_PERI_f)

            deltaV_C = V_TEM_f - V_TEM_i

            # Create command
            # TODO: Think about ideal transfer orbit to check for differences + duration
            c = Command()
            c.deltaV_C = deltaV_C
            c.true_anomaly = theta_i
            c.ideal_transfer_orbit = []
            c.duration = self.travel_time(chaser, chaser.kep.v, theta_i)
            c.description = 'Argument of Perigee correction'
            self.command_line.append(c)

            # Save single manoeuvre...
            self._save_result(chaser, target, len(self.command_line), True)

            # Propagate chaser and target to evaluate all the future commands properly
            self._propagator(chaser, target, c.duration)
            chaser.cartesian.V += deltaV_C            # Propagate as an impulsive thrust has been given

    def _error_estimator(self, chaser, chaser_next, target):
        """
            Given the position of chaser, the next position we want to reach and the target position
            estimate the amount of dV more that will be needed by integrating the disturbances over
            the path planned.
        """
        pass

    def _propagator(self, chaser, target, dt, dv=np.array([0, 0, 0])):
        """
            Propagate chaser and target to t* = now + dt.

        Args:
            chaser (Position):
            target (Position):
            dt (float):
        """

        # TODO: Change to a propagator that includes at least J2 disturbance

        r_C, v_C = pk.propagate_lagrangian(chaser.cartesian.R, chaser.cartesian.V + dv, dt, mu_earth)
        r_T, v_T = pk.propagate_lagrangian(target.cartesian.R, target.cartesian.V, dt, mu_earth)

        # Update positions of chaser and target objects
        target.update_target_from_cartesian(r_T, v_T)
        chaser.update_from_cartesian(r_C, v_C, target)

    def _target_propagator(self, target, dt):

        # TODO: Change to a propagator that includes at least J2 disturbance
        # Or maybe leave this propagator s.t we have discrepancies to correct

        r_T, v_T = pk.propagate_lagrangian(target.cartesian.R, target.cartesian.V, dt, mu_earth)
        target.update_target_from_cartesian(r_T, v_T)

    def _save_result(self, chaser, target, id=0, single_manoeuvre=False):
        if os.path.isdir('/home/dfrey/polybox/manoeuvre'):
            if single_manoeuvre:
                print "Saving single manoeuvre..."
                L = 1
            else:
                print "Saving complete manoeuvre..."
                L = len(self.command_line)

            # Simulating the whole manoeuvre and store the result
            chaser_tmp = Position()
            target_tmp = Position()

            chaser_tmp.from_other_position(chaser)
            target_tmp.from_other_position(target)

            # Creating list of radius of target and chaser
            R_target = [target_tmp.cartesian.R]
            R_chaser = [chaser_tmp.cartesian.R]
            R_chaser_lvlh = [chaser_tmp.lvlh.R]
            R_chaser_lvc = [np.array([chaser_tmp.lvc.dR, chaser_tmp.lvc.dV, chaser_tmp.lvc.dH])]

            for i in xrange(0, L):
                if single_manoeuvre:
                    cmd = self.command_line[-1]
                else:
                    cmd = self.command_line[i]

                duration = np.arange(0, cmd.duration, 0.1)

                for j in xrange(0, len(duration)):
                    self._propagator(chaser_tmp, target_tmp, 0.1)
                    R_chaser.append(chaser_tmp.cartesian.R)
                    R_target.append(target_tmp.cartesian.R)
                    R_chaser_lvlh.append(chaser_tmp.lvlh.R)
                    R_chaser_lvc.append(np.array([chaser_tmp.lvc.dR, chaser_tmp.lvc.dV, chaser_tmp.lvc.dH]))

                # Apply dV
                chaser_tmp.cartesian.V += cmd.deltaV_C

                self.print_state(chaser_tmp, target_tmp)

            # Saving in .mat file
            if single_manoeuvre:
                sio.savemat('/home/dfrey/polybox/manoeuvre/manoeuvre_' + str(id) + '.mat',
                        mdict={'abs_pos_c': R_chaser, 'rel_pos_c': R_chaser_lvlh, 'abs_pos_t': R_target,
                                'lvc_c': R_chaser_lvc})
            else:
                sio.savemat('/home/dfrey/polybox/manoeuvre/complete_manoeuvre.mat',
                        mdict={'abs_pos_c': R_chaser, 'rel_pos_c': R_chaser_lvlh, 'abs_pos_t': R_target,
                               'lvc_c': R_chaser_lvc})

            print "Manoeuvre saved."

    def clohessy_wiltshire_solver(self, chaser, chaser_next, target):
        """
            Solve Hill's Equation to get the amount of DeltaV needed to go from chaser position to chaser_next.
            [2] p. 382 (Algorithm 47)

        Args:
            chaser (position)
            chaser_next (position)
            target (position)
        """

        print ">>>> Solving CW-equations\n"

        a = target.kep.a
        max_time = int(2*np.pi * np.sqrt(a**3 / mu_earth))

        r_rel_c_0 = chaser.lvlh.R
        v_rel_c_0 = chaser.lvlh.V

        r_rel_c_n = chaser_next.lvlh.R
        v_rel_c_n = [0, 0, 0]

        n = np.sqrt(mu_earth/np.linalg.norm(target.cartesian.R)**3.0)

        phi_rr = lambda t: np.array([
            [4.0 - 3.0 * np.cos(n*t), 0.0, 0.0],
            [6.0*(np.sin(n*t) - n*t), 1.0, 0.0],
            [0.0, 0.0, np.cos(n*t)]
        ])

        phi_rv = lambda t: np.array([
            [1.0/n * np.sin(n*t), 2.0/n * (1 - np.cos(n*t)), 0.0],
            [2.0/n * (np.cos(n*t) - 1.0), 1.0 / n * (4.0 * np.sin(n*t) - 3.0*n*t), 0.0],
            [0.0, 0.0, 1.0 / n * np.sin(n*t)]
        ])

        phi_vr = lambda t: np.array([
            [3.0 * n * np.sin(n*t), 0.0, 0.0],
            [6.0 * n * (np.cos(n*t) - 1), 0.0, 0.0],
            [0.0, 0.0, -n * np.sin(n*t)]
        ])

        phi_vv = lambda t: np.array([
            [np.cos(n*t), 2.0 * np.sin(n*t), 0.0],
            [-2.0 * np.sin(n*t), 4.0 * np.cos(n*t) - 3.0, 0.0],
            [0.0, 0.0, np.cos(n*t)]
        ])

        best_deltaV = 1e12
        delta_T = 0

        # 1mm/sec accuracy. TODO: Check the accuracy of the thrusters!
        min_deltaV = 1e-6

        for t_ in xrange(1, max_time):

            rv_t = phi_rv(t_)
            det_rv = np.linalg.det(rv_t)

            if det_rv != 0:
                deltaV_1 = np.linalg.inv(rv_t).dot(r_rel_c_n - np.dot(phi_rr(t_), r_rel_c_0)) - v_rel_c_0
                deltaV_2 = np.dot(phi_vr(t_), r_rel_c_0) + np.dot(phi_vv(t_), v_rel_c_0 + deltaV_1) - v_rel_c_n

                deltaV_tot = np.linalg.norm(deltaV_1) + np.linalg.norm(deltaV_2)

                if best_deltaV > deltaV_tot:
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
        v = np.dot(phi_vr(T), r_rel_c_0) + np.dot(phi_vv(T), v_rel_c_0 + best_deltaV_1)

        plt.plot(r[:][0], r[:][1], 'r--')

        target_old = Position()
        target_old.from_other_position(target)
        chaser_old = Position()
        chaser_old.from_other_position(chaser)

        self._target_propagator(target, delta_T)

        B = target.cartesian.get_lof()
        R_target = target.cartesian.R + np.linalg.inv(B).dot(chaser_next.lvlh.R)

        sol = pk.lambert_problem(chaser.cartesian.R, R_target, delta_T, mu_earth, True, 10)

        target.from_other_position(target_old)

        self._propagator(chaser, target, 1e-3, np.array(sol.get_v1()[0]) - chaser.cartesian.V)
        for l in xrange(0, delta_T):
            plt.plot(chaser.lvlh.R[0], chaser.lvlh.R[1], 'g.')
            self._propagator(chaser, target, 1)

        chaser.from_other_position(chaser_old)
        target.from_other_position(target_old)

        # Change frame of reference of deltaV. From LVLH to Earth-Inertial
        B = target.cartesian.get_lof()
        deltaV_C_1 = np.linalg.inv(B).dot(best_deltaV_1)

        # Create command
        c1 = Command()
        c1.deltaV_C = deltaV_C_1
        c1.true_anomaly = chaser.kep.v
        c1.ideal_transfer_orbit = []
        c1.duration = 0
        c1.description = 'CW approach'

        # Propagate chaser and target to evaluate all the future commands properly
        self._propagator(chaser, target, 1e-3, deltaV_C_1)            # Propagate almost as an impulsive thrust has been given

        for l in xrange(0, delta_T):
            plt.plot(chaser.lvlh.R[0], chaser.lvlh.R[1], 'b.')
            self._propagator(chaser, target, 1)

        plt.show()

        self.print_state(chaser, target)

        self.command_line.append(c1)

        R = target.cartesian.get_lof()
        deltaV_C_2 = np.linalg.inv(R).dot(best_deltaV_2)

        # Create command
        c2 = Command()
        c2.deltaV_C = deltaV_C_2
        c2.true_anomaly = chaser.kep.v
        c2.ideal_transfer_orbit = []
        c2.duration = delta_T
        c2.description = 'CW approach'

        # Propagate chaser and target to evaluate all the future commands properly
        self._propagator(chaser, target, 1e-3, deltaV_C_2)            # Propagate almost as an impulsive thrust has been given

        self.print_state(chaser, target)

        self.command_line.append(c2)

    def multi_lambert(self, chaser, chaser_next, target):

        # Absolute position of chaser at t = t0
        R_C_i = chaser.cartesian.R
        V_C_i = chaser.cartesian.V

        # Absolute position of the target at t = t0
        R_T_i = target.cartesian.R
        V_T_i = target.cartesian.V

        best_deltaV = 1e12
        best_dt = 0

        # PROBLEM: To solve the lambert problem I need to state a certain dt. And the target will
        # be propagated by that dt.
        for dt in xrange(10, 10000):
            # Propagate absolute position we want to reach in the optimal way at t1 = t_start + dt
            # Propagate target position at t = t0 + dt
            target_old = Position()
            target_old.from_other_position(target)
            self._target_propagator(target_old, dt)

            # Transformation matrix from TEME to LVLH at time t1
            B_LVLH_TEME_f = target_old.cartesian.get_lof()

            # Evaluate final wanted absolute position of the chaser
            R_C_f = np.array(target_old.cartesian.R) + np.linalg.inv(B_LVLH_TEME_f).dot(chaser_next.lvlh.R)
            O_T_f = np.cross(target_old.cartesian.R, target_old.cartesian.V) / np.linalg.norm(target_old.cartesian.R)**2
            V_C_f = np.array(target_old.cartesian.V) + np.array([0.0, 0.0, 0.0]) + np.cross(O_T_f, np.linalg.inv(B_LVLH_TEME_f).dot(chaser_next.lvlh.R))

            T_CN = 2*np.pi*np.sqrt(chaser_next.kep.a**3 / mu_earth)
            N_max = int(np.floor(dt/T_CN)) + 1

            # Solve lambert in dt starting from the chaser position at t0 going to t1
            sol = pk.lambert_problem(R_C_i, R_C_f, dt, mu_earth, True, N_max)

            # Check for the best solution for this dt
            for i in xrange(0, len(sol.get_v1())):
                deltaV_1 = np.array(sol.get_v1()[i]) - V_C_i
                deltaV_2 = V_C_f - np.array(sol.get_v2()[i])
                deltaV_tot = np.linalg.norm(deltaV_1) + np.linalg.norm(deltaV_2)

                if deltaV_tot < best_deltaV:
                    best_deltaV = deltaV_tot
                    best_deltaV_1 = deltaV_1
                    best_deltaV_2 = deltaV_2
                    best_dt = dt


        # ############################## TEST WITH CW
        # n = np.sqrt(mu_earth/np.linalg.norm(target.cartesian.R)**3.0)
        # n_test = np.linalg.norm(np.cross(target.cartesian.R, target.cartesian.V))/np.linalg.norm(target.cartesian.R)**2
        #
        # O_T_i = np.cross(target.cartesian.R, target.cartesian.V) / np.linalg.norm(target.cartesian.R)**2
        # B_LVLH_TEME_i = target.cartesian.get_lof()
        #
        #
        # phi_rr = lambda t: np.array([
        #     [4.0 - 3.0 * np.cos(n*t), 0.0, 0.0],
        #     [6.0*(np.sin(n*t) - n*t), 1.0, 0.0],
        #     [0.0, 0.0, np.cos(n*t)]
        # ])
        #
        # phi_rv = lambda t: np.array([
        #     [1.0/n * np.sin(n*t), 2.0/n * (1 - np.cos(n*t)), 0.0],
        #     [2.0/n * (np.cos(n*t) - 1.0), 1.0 / n * (4.0 * np.sin(n*t) - 3.0*n*t), 0.0],
        #     [0.0, 0.0, 1.0 / n * np.sin(n*t)]
        # ])
        #
        # phi_vr = lambda t: np.array([
        #     [3.0 * n * np.sin(n*t), 0.0, 0.0],
        #     [6.0 * n * (np.cos(n*t) - 1), 0.0, 0.0],
        #     [0.0, 0.0, -n * np.sin(n*t)]
        # ])
        #
        # phi_vv = lambda t: np.array([
        #     [np.cos(n*t), 2.0 * np.sin(n*t), 0.0],
        #     [-2.0 * np.sin(n*t), 4.0 * np.cos(n*t) - 3.0, 0.0],
        #     [0.0, 0.0, np.cos(n*t)]
        # ])
        #
        # dr_0_TEME = R_C_i - R_T_i
        # dv_0_TEME = V_C_i - V_T_i - np.cross(O_T_i, dr_0_TEME)
        #
        # dr_0_LVLH = B_LVLH_TEME_i.dot(dr_0_TEME)
        # dv_0_LVLH = B_LVLH_TEME_i.dot(dv_0_TEME)
        #
        # dv_0_LVLH_transfer = dv_0_LVLH + B_LVLH_TEME_i.dot(best_deltaV_1)
        #
        # dr_C_f_testCW = phi_rr(best_dt).dot(dr_0_LVLH) + phi_rv(best_dt).dot(dv_0_LVLH_transfer)
        #
        # target_old = Position()
        # target_old.from_other_position(target)
        # self._target_propagator(target_old, best_dt)
        #
        # R_C_f_testCW = target_old.cartesian.R + np.linalg.inv(target_old.cartesian.get_lof()).dot(dr_C_f_testCW)
        # ##################################### ----> IT WORKS!

        c1 = Command()
        c1.deltaV_C = best_deltaV_1
        c1.true_anomaly = chaser.kep.v
        c1.duration = 0
        c1.description = 'Multi-Lambert solution'
        self.command_line.append(c1)

        # Save single manoeuvre...
        self._save_result(chaser, target, len(self.command_line), True)

        self._propagator(chaser, target, best_dt, best_deltaV_1)
        self.print_state(chaser, target)

        c2 = Command()
        c2.deltaV_C = best_deltaV_2
        c2.true_anomaly = chaser_next.kep.v
        c2.duration = best_dt
        c2.description = 'Multi-Lambert solution'
        self.command_line.append(c2)

        # Save single manoeuvre...
        self._save_result(chaser, target, len(self.command_line), True)

        chaser.cartesian.V += best_deltaV_2
        self.print_state(chaser, target)

    def plane_correction(self, chaser, chaser_next, target):
        """
            Correct plane inclination and raan with one single manoeuvre
            [1] Chapter 6
            [2] Chapter 6
        """

        # In a plane correction the following parameters are changed:
        # - RAAN
        # - Inclination
        # While those remain the same:
        # - Eccentricity
        # - Semi-Major Axis

        # TODO: account for dO/dt and di/dt

        # Constant values
        a = chaser.kep.a
        e = chaser.kep.e

        # Changing values
        O_i = chaser.kep.O
        O_f = chaser_next.kep.O
        dO = O_f - O_i
        i_i = chaser.kep.i
        i_f = chaser_next.kep.i
        di = i_f - i_i

        # Spherical trigonometry
        alpha = np.arccos(np.sin(i_i) * np.sin(i_f) * np.cos(dO) + np.cos(i_i) * np.cos(i_f))
        A_Li = np.arcsin(np.sin(i_f) * np.sin(dO) / np.sin(alpha))
        A_Lf = np.arcsin(np.sin(i_i) * np.sin(dO) / np.sin(alpha))

        B_Lf = np.arcsin(np.sqrt(np.cos(i_f)**2 * np.sin(i_i)**2 * np.sin(dO)**2 / (np.sin(alpha)**2 -
                        np.sin(i_i)**2 * np.sin(i_f)**2 * np.sin(dO)**2)))

        if i_f > np.pi/2.0:
            phi = O_f - B_Lf
        else:
            phi = O_f + B_Lf

        psi = np.arcsin(np.sin(i_i) * np.sin(i_f) * np.sin(dO) / np.sin(alpha))

        # Two possible positions where the burn can occur
        theta_i = (A_Li - chaser.kep.w) % (2.0 * np.pi)
        theta_i_tmp = (theta_i + np.pi) % (2.0 * np.pi)

        # Choose which of the two position is the closest
        # They consume different dV, the decision has to be taken then depending on if you want to spent a bit more
        # and burn in a specific point, or if you can born anywhere regardless on how much it will cost.
        # Now it's just takin the closest point to do the burn, to decrease the total time of the mission.
        if theta_i < chaser.kep.v:
            dv1 = 2*np.pi + theta_i - chaser.kep.v
        else:
            dv1 = theta_i - chaser.kep.v

        if theta_i_tmp < chaser.kep.v:
            dv2 = 2*np.pi + theta_i_tmp - chaser.kep.v
        else:
            dv2 = theta_i_tmp - chaser.kep.v

        if dv1 > dv2:
            theta_i = theta_i_tmp

        # Define vector c in Earth-Inertial frame of reference
        cx = np.cos(psi) * np.cos(phi)
        cy = np.cos(psi) * np.sin(phi)
        cz = np.sin(psi)

        # Define rotation of alpha radiants around vector c following right-hand rule
        k1 = 1.0 - np.cos(alpha)
        k2 = np.cos(alpha)
        k3 = np.sin(alpha)
        p = np.array([k1 * cx**2 + k2, k1 * cx * cy + k3 * cz, k1 * cx * cz - k3 * cy])
        q = np.array([k1 * cx * cy - k3 * cz, k1 * cy**2 + k2, k1 * cy * cz + k3 * cx])
        w = np.array([k1 * cx * cz + k3 * cy, k1 * cy * cz - k3 * cx, k1 * cz**2 + k2])
        R_c = np.identity(3)
        R_c[0:3, 0] = p
        R_c[0:3, 1] = q
        R_c[0:3, 2] = w

        # Evaluate velocity vector in Earth-Inertial reference frame at theta_i
        V_PERI_i = np.sqrt(mu_earth / (a * (1.0 - e**2))) * np.array([-np.sin(theta_i), e + np.cos(theta_i), 0.0])
        V_TEM_i = np.linalg.inv(chaser.kep.get_pof()).dot(V_PERI_i)

        # Rotate vector around c by alpha radiants
        V_TEM_f = R_c.dot(V_TEM_i)

        # Evaluate deltaV
        deltaV_C = V_TEM_f - V_TEM_i

        # # Compare against lambert solver
        # R_PERI_i =  a * (1.0 - e**2)/(1 + e * np.cos(theta_i)) * np.array([np.cos(theta_i), np.sin(theta_i), 0.0])
        # R_TEM_i = np.linalg.inv(chaser.kep.get_pof()).dot(R_PERI_i)
        #
        # R_PERI_i_n = a * (1.0 - e**2)/(1 + e * np.cos(theta_i + np.pi/2.0)) * np.array([np.cos(theta_i + np.pi/2.0), np.sin(theta_i + np.pi/2.0), 0.0])
        # R_TEM_i_n = np.linalg.inv(chaser.kep.get_pof()).dot(R_PERI_i_n)
        # R_TEM_f = R_c.dot(R_TEM_i_n)
        #
        # dt = self.travel_time(chaser, theta_i, theta_i + np.pi/2.0)
        # sol = pk.lambert_problem(R_TEM_i, R_TEM_f, dt, mu_earth, True)
        #

        # Create command
        c = Command()
        c.deltaV_C = deltaV_C
        c.true_anomaly = theta_i
        c.ideal_transfer_orbit = []
        c.duration = self.travel_time(chaser, chaser.kep.v, theta_i)
        c.description = 'Inclination and RAAN correction'
        self.command_line.append(c)

        # Save single manoeuvre...
        self._save_result(chaser, target, len(self.command_line), True)

        # Propagate chaser and target to evaluate all the future commands properly
        self._propagator(chaser, target, c.duration)
        self._propagator(chaser, target, 1e-3, deltaV_C)            # Propagate almost as an impulsive thrust has been given

    def travel_time(self, chaser, theta0, theta1):
        """
            Evaluate the travel time from a starting true anomaly theta0 to an end anomaly theta1.
            Ref: Exercise of Nicollier's Lecture.

        Args:
            chaser (Position): Position structure of the chaser
            theta0 (rad): Starting true anomaly
            theta1 (rad): Ending true anomaly

        Return:
            travel time (seconds)
        """

        a = chaser.kep.a
        e = chaser.kep.e

        T = 2.0 * np.pi * np.sqrt(a**3 / mu_earth)

        theta0 = theta0 % (2.0 * np.pi)
        theta1 = theta1 % (2.0 * np.pi)

        t0 = np.sqrt(a**3/mu_earth) * (2.0 * np.arctan((np.sqrt((1.0 - e)/(1.0 + e)) * np.tan(theta0 / 2.0))) -
                                       (e * np.sqrt(1.0 - e**2) * np.sin(theta0))/(1.0 + e * np.cos(theta0)))
        t1 = np.sqrt(a**3/mu_earth) * (2.0 * np.arctan((np.sqrt((1.0 - e)/(1.0 + e)) * np.tan(theta1 / 2.0))) -
                                       (e * np.sqrt(1.0 - e**2) * np.sin(theta1))/(1.0 + e * np.cos(theta1)))

        dt = t1 - t0

        if dt < 0:
            dt += T

        return dt

    def calc_synodic_period(self, chaser, target):
        """
            Calculate the synodic period
        """

        T_chaser = 2.0 * np.pi * np.sqrt(chaser.kep.a**3 / mu_earth)
        T_target = 2.0 * np.pi * np.sqrt(target.kep.a**3 / mu_earth)

        if T_chaser < T_target:
            T_syn = 1.0 / (1.0 / T_chaser - 1.0 / T_target)
        else:
            T_syn = 1.0 / (1.0 / T_target - 1.0 / T_chaser)

        return T_syn

    def optimal_dv(self, chaser, chaser_next, target):
        """
            Estimate anomaly difference that target and chaser should have to initiate a time dependant manoeuvre.
            Not suited for highly elliptical orbits
        """

        e_T = target.kep.e
        a_T = target.kep.a

        a_i = chaser.kep.a
        a_f = chaser_next.kep.a
        e_i = chaser.kep.e
        e_f = chaser_next.kep.e

        chaser_int = Position()

        # Assume both orbits are not intersecated
        if a_f > a_i:
            r_a_f = a_f * (1.0 + e_f)
            r_p_i = a_i * (1.0 - e_i)

            # Calculate intermediate orbital elements
            a_int = (r_a_f + r_p_i) / 2.0
            e_int = 1.0 - r_p_i / a_int

        else:
            r_a_i = a_i * (1.0 + e_i)
            r_p_f = a_f * (1.0 - e_f)

            # Calculate intermediate orbital elements
            a_int = (r_a_i + r_p_f) / 2.0
            e_int = 1.0 - r_p_f / a_int

        chaser_int.from_other_position(chaser)
        chaser_int.kep.a = a_int
        chaser_int.kep.e = e_int

        alpha_int = np.sqrt(1.0 - e_int**2) / (1.0 + e_int)
        alpha_T = np.sqrt(1.0 - e_T**2) / (1.0 + e_T)

        beta_int = e_int * np.sqrt(1.0 - e_int**2)
        beta_T = e_T * np.sqrt(1.0 - e_T**2)

        delta = 0.01

        t_int = self.travel_time(chaser_int, 0, np.pi - delta)

        J_T = np.sqrt(a_T**3 / mu_earth) * (2.0 * np.arctan(alpha_T * np.tan(v)))

    def drift_to(self, chaser, chaser_next, target, error):
        """
            Try to drift to the next checkpoint, inside a certain error ellipsoid. Return the time needed.
        """
        # Assuming that if we enter that point we are on a coelliptic orbit, we can directly say:
        if abs((np.linalg.norm(target.cartesian.R) + chaser_next.lvlh.R[0]) -
                       np.linalg.norm(chaser.cartesian.R)) >= error[1]:
            return None

        chaser_old = Position()
        chaser_old.from_other_position(chaser)
        target_old = Position()
        target_old.from_other_position(target)

        n_c = np.sqrt(mu_earth / chaser.kep.a ** 3)
        n_t = np.sqrt(mu_earth / target.kep.a ** 3)

        # If n_rel is below zero, we are moving slower than target. Otherwise faster.
        n_rel = n_c - n_t

        # Required dv at the end of the manoeuvre, estimation based on the relative position
        dv_req = np.sign(chaser_next.lvlh.R[1]) * np.linalg.norm(chaser_next.lvlh.R) / R_earth

        # Check if a drift to the wanted position is possible, if yes check if it can be done under a certain time,
        # if not try to resync
        actual_dv = (chaser.kep.v + chaser.kep.w) % (2.0*np.pi) - (target.kep.v + target.kep.w) % (2.0*np.pi)

        # Define a function F for the angle calculation
        F = lambda dv_req, dv, n: int((dv - dv_req) / n > 0.0) * np.sign(n)

        t_est = (2.0 * np.pi * F(dv_req, actual_dv, n_rel) + dv_req - actual_dv) / n_rel
        t_est_old = 0
        ellipsoid_flag = False
        tol = 1e-3         # Millisecond tolerance
        while abs(t_est - t_est_old) > tol:
            self._propagator(chaser_old, target_old, t_est)
            dr_next = chaser_next.lvlh.R[1] - chaser_old.lvlh.R[1]

            t_est_old = t_est
            t_est *= 1.0 + dr_next/np.linalg.norm(chaser_old.cartesian.R)

            # Assuming to stay on the same plane
            if abs(chaser_next.lvlh.R[0] - chaser_old.lvlh.R[0]) <= error[0] and \
                abs(chaser_next.lvlh.R[1] - chaser_old.lvlh.R[1]) <= error[1]:
                # We have reached the error ellipsoid, can break
                ellipsoid_flag = True

            chaser_old.from_other_position(chaser)
            target_old.from_other_position(target)


        if ellipsoid_flag:
            # With the estimated time, we are in the error-ellipsoid
            return t_est
        else:
            return None

    def print_state(self, chaser, target):

        print "Chaser state: "
        print " >> Kep: "
        print "      a :     " + str(chaser.kep.a)
        print "      e :     " + str(chaser.kep.e)
        print "      i :     " + str(chaser.kep.i)
        print "      O :     " + str(chaser.kep.O)
        print "      w :     " + str(chaser.kep.w)
        print "      v :     " + str(chaser.kep.v)
        print " >> "


        print "Target state: "
        print " >> Kep: "
        print "      a :     " + str(target.kep.a)
        print "      e :     " + str(target.kep.e)
        print "      i :     " + str(target.kep.i)
        print "      O :     " + str(target.kep.O)
        print "      w :     " + str(target.kep.w)
        print "      v :     " + str(target.kep.v)

    def print_result(self):
        tot_dv = 0
        tot_dt = 0

        for it, command in enumerate(self.command_line):
            print '\n' + command.description + ', command nr. ' + str(it) + ':'
            print '--> DeltaV:            ' + str(command.deltaV_C)
            print '--> Normalized DeltaV: ' + str(np.linalg.norm(command.deltaV_C))
            print '--> Idle after burn:   ' + str(command.duration)
            print '--> Burn position:     ' + str(command.true_anomaly)
            tot_dv += np.linalg.norm(command.deltaV_C)
            tot_dt += command.duration

        return tot_dv, tot_dt