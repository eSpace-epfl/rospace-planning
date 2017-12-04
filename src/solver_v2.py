import numpy as np
import scipy.io as sio
import epoch_clock
import pykep as pk

from space_tf import Cartesian, CartesianLVLH, mu_earth


class Command:

    def __init__(self):
        self.deltaV_C = [0, 0, 0]
        self.true_anomaly = 0
        self.ideal_transfer_orbit = []
        self.duration = 0

class Solver:

    def __init__(self):
        self.command_line = []

    def solve_scenario(self, scenario, chaser, target):
        print "Solving the scenario: " + scenario.name + "\n"
        print "Scenario overview: "
        print scenario.overview

        # Extract scenario positions
        positions = scenario.positions

        # Extract scenario start time to propagate chaser and target position
        epoch = epoch_clock.Epoch()
        dt = scenario.mission_start - epoch.now()
        dt = dt.seconds
        self._propagator(chaser, target, dt)

        # Define total deltav and total time
        tot_time = 0
        tot_deltaV = 0

        # Start solving scenario by popping positions from position list
        i = 1
        while len(positions) > 0:
            print "\n ------> Evaluating manoeuvre " + str(i)
            chaser_next = positions.pop(0)

            self.solve(chaser, chaser_next, target)

            i += 1

            tot_time += self.command_line[-1].duration
            tot_deltaV += np.linalg.norm(self.command_line[-1].deltaV_C)

        print "-----------------> Manoeuvre elaborated <--------------------"
        print "--> Start time:            " + str(scenario.mission_start)
        print "--> Manoeuvre duration:    " + str(tot_time) + " seconds"
        print "--> Total deltaV:          " + str(tot_deltaV) + " km/s"

    def solve(self, chaser, chaser_next, target):
        # TODO: Version 2.0
        # Correct for the plane angles, TODO: Set some thresholds in which we can change them
        # Inclination
        dIx_C_CN = chaser.rel_kep.dIx - chaser_next.rel_kep.dIx
        if dIx_C_CN != 0:
            # Relative inclination between chaser and chaser next has to be adjusted
            self.adjust_inclination(chaser, chaser_next)

        # RAAN, TODO: check the case i = pi/2 and if i for chaser next is available
        dIy_C_CN = chaser.rel_kep.dIy / np.sin(chaser.kep.i) - chaser_next.rel_kep.dIy / np.sin(chaser_next.kep.i)
        if dIy_C_CN != 0:
            # Relative RAAN bewteen chaser and chaser next has to be adjusted
            self.adjust_raan(chaser, chaser_next)

        # Perigee
        dEx_C_CN = chaser.rel_kep.dEx - chaser_next.rel_kep.dEx
        dEy_C_CN = chaser.rel_kep.dEy - chaser_next.rel_kep.dEy
        dw = dEx_C_CN**2 + dEy_C_CN*2
        if dw == 0:
            # Relative argument of perigee between chaser and chaser next has to be adjusted. TODO: put tolerance instead of 0
            self.adjust_perigee(chaser, chaser_next)

        # Once the angles are corrected, the orbit can be modified according to the next position wanted using the
        # Lambert solver.
        self.adjust_eccentricity_semimajoraxis(chaser, chaser_next, target)

    def adjust_eccentricity_semimajoraxis(self, chaser, chaser_next, target):
        # TODO: Add manoeuvre duration & print of informations (deltaV, time)
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
        e_CN = e_T - np.sqrt(chaser_next.rel_kep.dEx**2 + chaser_next.rel_kep.dEy**2)
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
        # TODO: Insert the case where the orbit are synchronous
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
        while flag:
            M_T_next = M_T_next + n_T * T_C
            E_T_next = self._eccentric_anomaly_from_mean_anomaly(e_T, M_T_next)
            theta_T_next = self._true_anomaly_from_eccentric_anomaly(e_T, E_T_next)
            dt_idle += T_C
            if abs(theta_T_next % 2*np.pi - theta_T_t0) < 1e-5:
                flag = False

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
            E = E - self._mean_anomaly_from_eccentric_anomaly(e, E)/(1 - e * np.cos(E))
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

    def _propagator(self, chaser, target, dt):
        """
            Propagate chaser and target to t* = now + dt.

        Args:
            chaser (Position):
            target (Position):
            dt (float):
        """

        r_C, v_C = pk.propagate_lagrangian(chaser.cartesian.R, chaser.cartesian.V, dt, mu_earth)
        r_T, v_T = pk.propagate_lagrangian(target.cartesian.R, target.cartesian.V, dt, mu_earth)

        # Update positions
        target.update_target_from_cartesian(r_T, v_T)
        chaser.update_from_cartesian(r_C, v_C, target)

    def _save_result(self, chaser, chaser_next, target):
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
