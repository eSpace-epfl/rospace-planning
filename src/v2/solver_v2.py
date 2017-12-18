import numpy as np
import scipy.io as sio
import epoch_clock
import pykep as pk
import os
import datetime

from space_tf import Cartesian, CartesianLVLH, mu_earth
from scenario_v2 import Position

class Command:

    def __init__(self):
        self.deltaV_C = [0, 0, 0]
        self.ideal_transfer_orbit = []
        self.duration = 0

        self.true_anomaly = 0
        self.mean_anomaly = 0
        self.epoch = 0


class Solver:

    def __init__(self):
        self.command_line = []
        self.solver_clock = 0

        self.test_error = {}

    def solve_scenario(self, scenario, chaser, target):
        print "Solving the scenario: " + scenario.name + "\n"
        print "Scenario overview: "
        print scenario.overview

        # Extract scenario start time to propagate chaser and target position
        epoch = epoch_clock.Epoch()
        t_start = scenario.mission_start
        t_now = epoch.now()

        dt = t_start - t_now
        dt = dt.seconds
        self.solver_clock = t_start

        chaser_old = Position()
        chaser_old.from_other_position(chaser)
        target_old = Position()
        target_old.from_other_position(target)

        print "Chaser position at 12:20:00:    " + str(chaser.cartesian.R)
        print "Target position at 12:20:00:    " + str(target.cartesian.R)

        # Propagate chaser and target position to t_start
        self._propagator(chaser, target, dt)

        print "Manoeuvre start at (chaser):     " + str(chaser.cartesian.R)
        print "               ... (target):     " + str(target.cartesian.R)
        print "Relative position at start:      " + str(chaser.lvlh.R)

        # Update scenario positions with the target position at t_start
        scenario.update_yaml_scenario(target)

        # Extract scenario positions
        positions = scenario.positions

        # Define total deltav and total time
        tot_time = 0
        tot_deltaV = 0

        # Start solving scenario by popping positions from position list
        i = 0
        while i < len(positions):
            print "\n------> Evaluating manoeuvre " + str(i)

            # Extract the first position, at t = t_start
            chaser_next = positions[i].position

            # self.solve(chaser, chaser_next, target)
            self.solve_multi_lambert(chaser, chaser_next, target)

            print "Delta-V needed for the first burn:     " + str(self.command_line[-2].deltaV_C)
            print "Delta-V needed for the second burn:    " + str(self.command_line[-1].deltaV_C)
            print "Total Delta-V for this manoeuvre:      " + str(np.linalg.norm(self.command_line[-2].deltaV_C)
                                                             + np.linalg.norm(self.command_line[-1].deltaV_C))
            print "Time needed to perform this manoeuvre: " + str(self.command_line[-2].duration)
            print "Scheduled at:                          " + str(self.command_line[-2].epoch)

            print "Arrival relative position:             " + str(chaser.lvlh.R)

            tot_time += self.command_line[-2].duration
            tot_deltaV += np.linalg.norm(self.command_line[-2].deltaV_C) + np.linalg.norm(self.command_line[-1].deltaV_C)

            # Update positions
            scenario.update_yaml_scenario(target)
            i += 1

        print "\n\n-----------------> Manoeuvre elaborated <--------------------"
        print "--> Start time:            " + str(scenario.mission_start)
        print "--> Manoeuvre duration:    " + str(tot_time) + " seconds"
        print "--> Total deltaV:          " + str(tot_deltaV) + " km/s"

        scenario.export_solved_scenario(self.command_line)

        self._save_result(chaser_old, target_old, dt)

    def solve(self, chaser, chaser_next, target):
        # TODO: Version 2.0
        # Correct for the plane angles, TODO: Set some thresholds in which we can change them
        # Inclination
        dIx_C_CN = chaser.rel_kep.dIx - chaser_next.rel_kep.dIx
        if dIx_C_CN != 0:
            # Relative inclination between chaser and chaser next has to be adjusted
            self.adjust_inclination(chaser, chaser_next)

        # RAAN, TODO: check the case i = pi/2 and if i for chaser next is available
        dIy_C_CN = chaser.rel_kep.dIy / np.sin(chaser.kep.i) - \
                   chaser_next.rel_kep.dIy / np.sin(target.kep.i - chaser_next.rel_kep.dIx)
        if dIy_C_CN != 0:
            # Relative RAAN bewteen chaser and chaser next has to be adjusted
            self.adjust_raan(chaser, chaser_next)

        # Perigee
        dEx_C_CN = chaser.rel_kep.dEx - chaser_next.rel_kep.dEx
        dEy_C_CN = chaser.rel_kep.dEy - chaser_next.rel_kep.dEy
        dw = dEx_C_CN**2 + dEy_C_CN*2
        if dw == 0:
            # Relative argument of perigee between chaser and chaser next has to be adjusted.
            # TODO: put tolerance instead of 0
            self.adjust_perigee(chaser, chaser_next)

        # Once the angles are corrected, the orbit can be modified according to the next position wanted using the
        # Lambert solver.
        self.adjust_eccentricity_semimajoraxis(chaser, chaser_next, target)

    def solve_multi_lambert(self, chaser, chaser_next, target):
        print "------------- Solving Multi-Lambert Problem --------------- \n"

        # Absolute position of chaser at t = t0
        p_C_TEM_t0 = chaser.cartesian.R
        v_C_TEM_t0 = chaser.cartesian.V

        # Absolute position of target at t = t0
        p_T_TEM_t0 = target.cartesian.R
        v_T_TEM_t0 = target.cartesian.V

        best_deltaV = 1e12
        best_dt = 0

        # PROBLEM: To solve the lambert problem I need to state a certain dt. And the target will
        # be propagated by that dt.

        print "Starting relative position:       " + str(chaser.lvlh.R)

        for dt in xrange(50, 500):
            # Propagate absolute position we want to reach in the optimal way at t1 = t_start + dt
            # Propagate target position at t = t0 + dt
            # chaser_next_old = Position()
            # chaser_next_old.from_other_position(chaser_next)
            target_old = Position()
            target_old.from_other_position(target)

            # Propagate target and next position to t1, chaser stays at t_start
            # self._propagator(chaser_next, target, dt)

            # Propagate target to t1
            self._target_propagator(target, dt)

            # Update positions given the new target position (LVLH is locked)
            chaser_next.update_from_lvlh(target, chaser_next.lvlh.R)

            p_CN_TEM_t1 = chaser_next.cartesian.R
            v_CN_TEM_t1 = chaser_next.cartesian.V

            # Test min time
            # c = np.linalg.norm(p_CN_TEM_t1 - p_C_TEM_t0)
            # dv = chaser_next.kep.v - chaser.kep.v
            # p_min = np.linalg.norm(p_CN_TEM_t1) * np.linalg.norm(p_C_TEM_t0) / c * (1 - np.cos(dv))
            # s = 2*(np.linalg.norm(p_CN_TEM_t1) + np.linalg.norm(p_C_TEM_t0) + c)
            # e_min = np.sqrt(1.0 - 2.0*p_min/s)
            # a_min = p_min / (1.0 - e_min**2)
            # beta_e = 2*np.arcsin(np.sqrt((s - c)/(2*a_min)))
            # t_min = np.sqrt(a_min**3/mu_earth) * (np.pi - beta_e + np.sin(beta_e))

            T_CN = 2*np.pi*np.sqrt(chaser_next.kep.a**3 / mu_earth)
            N_max = int(np.floor(dt/T_CN)) + 1

            # Solve lambert in dt starting from the chaser position at t0 going to t1
            sol = pk.lambert_problem(p_C_TEM_t0, p_CN_TEM_t1, dt, mu_earth, True, N_max)

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

            # Depropagate, otherwise it keeps propagating over the previous
            # chaser_next.from_other_position(chaser_next_old)
            target.from_other_position(target_old)

        # Update target and chaser after the first burn
        self._propagator(chaser, target, best_dt, best_deltaV_1)

        # Update chaser after the second burn
        chaser.cartesian.V += best_deltaV_2

        c1 = Command()
        c1.deltaV_C = best_deltaV_1
        c1.true_anomaly = chaser.kep.v
        c1.duration = best_dt
        c1.epoch = self.solver_clock
        print self.solver_clock

        self.solver_clock += datetime.timedelta(0, best_dt)
        print self.solver_clock

        c2 = Command()
        c2.deltaV_C = best_deltaV_2
        c2.true_anomaly = chaser_next.kep.v
        c2.duration = 0
        c2.epoch = self.solver_clock

        # Add to the command line the burn needed
        self.command_line.append(c1)
        self.command_line.append(c2)

    def adjust_eccentricity_semimajoraxis(self, chaser, chaser_next, target):
        # TODO: Add manoeuvre duration & print of informations (deltaV, time)
        # TODO: Try to add the multilambert solver
        # When this solver is called, only the first two relative orbital elements have to be aligned with
        # the requested one.

        # Calculate the wanted next semimajor axis
        a_CN = chaser.kep.a * (chaser.rel_kep.dA + 1)/(chaser_next.rel_kep.dA + 1)
        a_T = target.kep.a
        a_C = chaser.kep.a

        # Calculate the wanted mean anomaly difference, the manoeuvre configuration (the true anomalies that
        # target and chaser has to have) and the eventual waiting time to reach that configuration.
        # Theoretically at this point, dL = M_T - M_C
        e_T = target.kep.e
        e_CN = np.sqrt(chaser_next.rel_kep.dEx**2 + chaser_next.rel_kep.dEy**2) - e_T
        e_C = chaser.kep.e

        n_T = np.sqrt(mu_earth / a_T**3)
        n_CN = np.sqrt(mu_earth / a_CN**3)
        n_C = np.sqrt(mu_earth / a_C**3)

        a_CN_int = (a_C/(1+e_C) + a_CN/(1-e_CN))/2
        e_CN_int = a_CN_int / (a_C/(1+e_C)) - 1
        n_CN_int = np.sqrt(mu_earth / a_CN_int**3)

        # Burning position is set to be periapsis to reduce losses, maneouvre start anomaly set to be 0
        E_CN_int_t1 = self._eccentric_anomaly_from_true_anomaly(e_CN_int, np.pi)
        M_CN_int_t1 = self._mean_anomaly_from_eccentric_anomaly(e_CN_int, E_CN_int_t1)

        # Calculate the transfer time, starting the burn at the periapsis
        dt = M_CN_int_t1 / n_CN_int

        # TODO: Review from here
        # Calculate mean anomaly of the target at t = t1 - dt = t0 = 0
        # Knowing that dL = 1
        M_T_t1 = chaser_next.rel_kep.dL + M_CN_int_t1
        tau_T = dt - M_T_t1 / n_T
        M_T_t0 = -n_T * tau_T
        E_T_t0 = self._eccentric_anomaly_from_mean_anomaly(e_T, M_T_t0)
        theta_T_t0 = self._true_anomaly_from_eccentric_anomaly(e_T, E_T_t0)

        # Now we know that the configuration we need to have in order to start the manoeuvre should be
        # theta = 0 for the chaser
        # theta = theta_T_t0 for the target

        # Take the actual configuration and propagate target and chaser to get the idle time
        # TODO: Insert the case where the orbit are synchronous not working well
        theta_T_actual = target.kep.v
        theta_C_actual = chaser.kep.v
        T_C = 2*np.pi/n_C
        E_T_actual = self._eccentric_anomaly_from_true_anomaly(e_T, theta_T_actual)
        M_T_actual = self._mean_anomaly_from_eccentric_anomaly(e_T, E_T_actual)
        E_C_actual = self._eccentric_anomaly_from_true_anomaly(e_C, theta_C_actual)
        M_C_actual = self._mean_anomaly_from_eccentric_anomaly(e_C, E_C_actual)
        dt_idle = (n_C * T_C - M_C_actual) / n_C
        flag = True
        M_T_next = M_T_actual + n_T * dt_idle
        # while flag:
        #     M_T_next = M_T_next + n_T * T_C
        #     E_T_next = self._eccentric_anomaly_from_mean_anomaly(e_T, M_T_next)
        #     theta_T_next = self._true_anomaly_from_eccentric_anomaly(e_T, E_T_next)
        #     dt_idle += T_C
        #     if abs(theta_T_next % (2*np.pi) - theta_T_t0) < 1e-5:
        #         flag = False

        # At this point, the manoeuvre start anomaly can be calculated
        M_C_start = n_C * dt_idle + M_C_actual
        E_C_start = self._eccentric_anomaly_from_mean_anomaly(e_C, M_C_start)
        theta_start_1 = self._mean_anomaly_from_eccentric_anomaly(e_C, E_C_start)

        # TODO: Maybe go for vectorial calculation to be more generic
        # And also the manoeuvre start burn intensity
        deltaV_C_1 = np.sqrt(mu_earth * (2*(1+e_C)/a_C) - 1/a_CN) - np.sqrt(mu_earth * (1+2*e_C)/a_C)

        # Evaluate burn vector in chaser frame of reference
        deltaV_C_1 = deltaV_C_1 * np.array([1, 0, 0])

        # The semimajor-axis now matches the relative orbital elements, therefore we need to burn at the apoapsis
        # to match eccentricity, dL and da
        # TODO: What happen when cos(w) = 0?
        e_F = e_T - chaser_next.rel_kep.dEx / np.cos(chaser.kep.w)

        deltaV_C_2 = np.sqrt(2*(1 - e_CN)/a_CN - 1/a_CN) - np.sqrt(2*(1 - e_CN)/a_CN - 1/a_CN_int)

        # Evaluate burn vector in chaser frame of reference
        deltaV_C_2 = deltaV_C_2 * np.array([1, 0, 0])

        # Create command
        # TODO: Think about ideal transfer orbit to check for differences
        c1 = Command()
        c1.deltaV_C = deltaV_C_1
        c1.true_anomaly = theta_start_1
        c1.ideal_transfer_orbit = []

        # TODO: Debug the true anomaly thing
        c2 = Command()
        c2.deltaV_C = deltaV_C_2
        c2.true_anomaly = np.pi
        c2.ideal_transfer_orbit = []

        # Add to the command line the burn needed
        self.command_line.append(c1)
        self.command_line.append(c2)

    def adjust_inclination(self, chaser, chaser_next):
        """
            Given the chaser relative orbital elements w.r.t target,
            ad its next wanted status correct the inclination.

        Args:
              chaser (Position):
              chaser_next (Position):
        """

        # TODO: Add manoeuvre duration & print of informations (deltaV, time)
        # TODO: Try to integrate changes in inclination and RAAN with lambert solver

        # Position at which the burn should occur
        theta = [2*np.pi - chaser.kep.w, (3*np.pi - chaser.kep.w) % (2*np.pi)]

        # Evaluate the inclination difference to correct
        di = chaser.rel_kep.dIx - chaser_next.rel_kep.dIx

        # Calculate burn intensity
        # TODO: The velocity has to be propagated to the point where the burn occurs
        deltav = np.linalg.norm(chaser.cartesian.V) * np.sqrt(2*(1 - np.cos(di)))

        # Evaluate burn direction in chaser frame of reference
        deltav_C = deltav * np.array([np.cos(np.pi/2 + di/2), 0, np.sin(np.pi/2 + di/2)])

        # Create command
        # TODO: Think about ideal transfer orbit to check for differences
        c = Command()
        c.deltaV_C = deltav_C
        c.true_anomaly = theta
        c.ideal_transfer_orbit = []

        # Add to the command line the burn needed
        self.command_line.append(c)

    def adjust_raan(self, chaser, chaser_next):
        """
            Given the chaser relative orbital elements w.r.t target,
            ad its next wanted status correct the RAAN.

        Args:
              chaser (Position):
              chaser_next (Position):
        """

        # TODO: Add manoeuvre duration & print of informations (deltaV, time)

        # Evaluate RAAN difference to correct
        # TODO: Think about what happen when i = pi/2...
        i_C_next = chaser.rel_kep.dIx - chaser_next.rel_kep.dIy + chaser.kep.i
        draan = chaser.rel_kep.dIy / np.sin(chaser.kep.i) - chaser_next.rel_kep.dIy / np.sin(i_C_next)

        # Rotational matrix between the two planes
        R = np.identity(3)
        R[0, 0:3] = np.array([np.cos(draan), -np.sin(draan), 0])
        R[1, 0:3] = np.array([np.sin(draan), np.cos(draan), 0])
        R[2, 0:3] = np.array([0, 0, 1])

        # Position at which the burn should occur
        # TODO: Review this calculation...
        h = np.cross(chaser.cartesian.R, chaser.cartesian.V)
        h_next = R.dot(h)
        n = np.cross(h, h_next)
        n = n / np.linalg.norm(n)
        theta = [np.arccos(n[0]), np.arccos(n[0]) + np.pi]

        # Calculate burn intensity
        # TODO: The velocity has to be propagated to the point where the burn occurs
        deltav = np.linalg.norm(chaser.cartesian.V) * np.sqrt(2 * (1 - np.cos(draan)))

        # Evaluate burn direction in chaser frame of reference
        deltav_C = deltav * np.array([np.cos(np.pi / 2 + draan / 2), 0, np.sin(np.pi / 2 + draan / 2)])

        # Create command
        # TODO: Think about ideal transfer orbit to check for differences
        c = Command()
        c.deltaV_C = deltav_C
        c.true_anomaly = theta
        c.ideal_transfer_orbit = []

        # Add to the command line the burn needed
        self.command_line.append(c)

    def adjust_perigee(self, chaser, chaser_next):
        """
            Given the chaser relative orbital elements w.r.t target,
            ad its next wanted status correct the perigee argument.

        Args:
              chaser (Position):
              chaser_next (Position):
        """

        # TODO: Add manoeuvre duration & print of informations (deltaV, time)

        # Evaluate perigee difference to correct
        # TODO: Think about what happen when sin(0.5*(w_n - w)) = 0...
        ddEx = chaser.rel_kep.dEx - chaser_next.rel_kep.dEx
        ddEy = chaser.rel_kep.dEy - chaser_next.rel_kep.dEy
        dw = 2 * (np.arctan(-ddEx/ddEy) - chaser.kep.w)

        # Position at which the burn should occur
        # TODO: Review this calculation, think in which of the two the deltav will be less
        theta = [dw/2.0, np.pi + dw/2.0]

        # Calculate burn intensity
        # TODO: The velocity has to be propagated to the point where the burn occurs as well as the radius
        alpha = np.arccos((1 + 2*chaser.kep.e*np.cos(dw/2.0) + np.cos(dw) * chaser.kep.e**2) *
                          np.linalg.norm(chaser.cartesian.R)/((1 - chaser.kep.e**2)*(2*chaser.kep.a - np.linalg.norm(chaser.cartesian.R))))
        deltav = np.linalg.norm(chaser.cartesian.v) * np.sqrt(2 * (1 - np.cos(alpha)))

        # Evaluate burn direction in chaser frame of reference
        deltav_C = deltav * np.array([np.cos(np.pi / 2 + alpha / 2), 0, np.sin(np.pi / 2 + alpha / 2)])

        # Create command
        # TODO: Think about ideal transfer orbit to check for differences
        c = Command()
        c.deltaV_C = deltav_C
        c.true_anomaly = theta
        c.ideal_transfer_orbit = []

        # Add to the command line the burn needed
        self.command_line.append(c)

    def _eccentric_anomaly_from_true_anomaly(self, e, theta):
        E = 2 * np.arctan(np.sqrt((1-e)/(1+e)) * np.tan(theta/2))
        return E

    def _mean_anomaly_from_eccentric_anomaly(self, e, E):
        M = E - e * np.sin(E)
        return M

    def _eccentric_anomaly_from_mean_anomaly(self, e, M):
        tol = 1e-8
        E = np.pi/4
        err = 1e12
        while err > tol:
            E_old = E
            E = E - (self._mean_anomaly_from_eccentric_anomaly(e, E) - M)/(1 - e * np.cos(E))
            err = abs(E - E_old)
        return E

    def _true_anomaly_from_eccentric_anomaly(self, e, E):
        theta = 2 * np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(E/2))
        return theta

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

    def _save_result(self, chaser, target, dt):
        if os.path.isdir('/home/dfrey/polybox/manoeuvre'):
            print "Saving manoeuvre..."

            # Simulating the whole manoeuvre and store the result
            chaser_tmp = Position()
            target_tmp = Position()

            chaser_tmp.from_other_position(chaser)
            target_tmp.from_other_position(target)

            # Creating list of radius of target and chaser
            R_target = [target_tmp.cartesian.R]
            R_chaser = [chaser_tmp.cartesian.R]
            R_chaser_lvlh = [chaser_tmp.lvlh.R]
            R_chaser_lvc =  [np.array([chaser_tmp.lvc.dR, chaser_tmp.lvc.dV, chaser_tmp.lvc.dH])]


            for k in xrange(0, dt):
                self._propagator(chaser_tmp, target_tmp, 1)
                R_chaser.append(chaser_tmp.cartesian.R)
                R_target.append(target_tmp.cartesian.R)
                R_chaser_lvlh.append(chaser_tmp.lvlh.R)
                R_chaser_lvc.append(np.array([chaser_tmp.lvc.dR, chaser_tmp.lvc.dV, chaser_tmp.lvc.dH]))


            for i in xrange(0, len(self.command_line)):
                cmd = self.command_line[i]

                # Apply dV
                chaser_tmp.cartesian.V += cmd.deltaV_C

                for j in xrange(0, cmd.duration):
                    self._propagator(chaser_tmp, target_tmp, 1)
                    R_chaser.append(chaser_tmp.cartesian.R)
                    R_target.append(target_tmp.cartesian.R)
                    R_chaser_lvlh.append(chaser_tmp.lvlh.R)
                    R_chaser_lvc.append(np.array([chaser_tmp.lvc.dR, chaser_tmp.lvc.dV, chaser_tmp.lvc.dH]))

                print "Relative Position after command " + str(i) + ":    " + str(chaser_tmp.lvlh.R)

            # Saving in .mat file
            sio.savemat('/home/dfrey/polybox/manoeuvre/full_manoeuvre.mat',
                        mdict={'abs_pos_c': R_chaser, 'rel_pos_c': R_chaser_lvlh, 'abs_pos_t': R_target,
                               'lvc_c': R_chaser_lvc})

            print "Manoeuvre saved."

    def clohessy_wiltshire_solver(self, chaser, chaser_next, target):

        print "\n -------------Solving CW-equations--------------- \n"
        print " Useful only for really close operations, "

        chaser.kep.from_cartesian(chaser.cartesian)
        chaser_next.kep.from_cartesian(chaser_next.cartesian)
        target.kep.from_cartesian(target.cartesian)

        a = target.kep.a
        max_time = 150000
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
        v = np.dot(phi_vr(T), r_rel_c_0) + np.dot(phi_vv(T), v_rel_c_0 + best_deltaV_1)

        self.test_error['CW-sol'] = {'deltaV_1': best_deltaV_1, 'deltaV_2': best_deltaV_2, 'dt': delta_T, 'r': r, 'v': v}