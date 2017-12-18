import numpy as np
import scipy.io as sio
import epoch_clock
import pykep as pk
import os
import datetime

from space_tf import Cartesian, CartesianLVLH, mu_earth, KepOrbElem
from scenario import Position

class Command:

    def __init__(self):
        self.deltaV_C = [0, 0, 0]
        self.ideal_transfer_orbit = []
        self.duration = 0

        self.true_anomaly = 0
        self.mean_anomaly = 0
        self.epoch = 0

        self.theta_diff = None


class Solver:

    def __init__(self):
        self.command_line = []
        self.solver_clock = 0

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

            self.solve(chaser, chaser_next, target)

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
        dIy_C_CN = chaser.rel_kep.dIy / np.sin(chaser.kep.i) - \
                   chaser_next.rel_kep.dIy / np.sin(target.kep.i - chaser_next.rel_kep.dIx)
        if dIx_C_CN != 0 or dIy_C_CN != 0:
            # Relative inclination between chaser and chaser next has to be adjusted
            self.plane_correction(chaser, chaser_next, target)

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

    def adjust_eccentricity_semimajoraxis(self, chaser, chaser_next, target):
        # When this solver is called, only the first two relative orbital elements have to be aligned with
        # the requested one.
        a_i = chaser.kep.a
        e_i = chaser.kep.e
        a_f = chaser_next.kep.a
        e_f = chaser_next.kep.e

        r_a_f = a_f * (1.0 + e_f)
        r_p_i = a_i * (1.0 - e_i)

        # Calculate intermediate orbital elements
        a_int = (r_a_f + r_p_i) / 2.0
        e_int = 1.0 - r_p_i / a_int

        # Calculate deltaV's
        deltaV_C_1 = np.sqrt(mu_earth * (2.0 / r_p_i - 1.0 / a_int)) - np.sqrt(mu_earth * (2.0 / r_p_i - 1.0 / a_i))
        deltaV_C_2 = np.sqrt(mu_earth * (2.0 / r_a_f - 1.0 / a_f)) - np.sqrt(mu_earth * (2.0 / r_a_f - 1.0 / a_int))

        # Delta-V in chaser reference frame
        deltaV_C_1 = deltaV_C_1 * np.array([1, 0, 0])
        deltaV_C_2 = deltaV_C_2 * np.array([1, 0, 0])


        # Check if a specific relative distance w.r.t the target is wanted.
        # In that case compute the waiting time on the intermediate orbit, check if it's possible to achieve the wanted
        # position. If not, adjust the burn to achieve phasing.
        dtheta = target.kep.v - chaser_next.kep.v
        synodal_period = self.calc_synodal_period(chaser_next, target)
        # Check after how much time such a configuration will happen again
        # Wait till the configuration is reached
        # Do the second (if wanted) burn

        # Create command
        c1 = Command()
        c1.deltaV_C = deltaV_C_1
        c1.true_anomaly = 0
        c1.ideal_transfer_orbit = []

        c2 = Command()
        c2.deltaV_C = deltaV_C_2
        c2.true_anomaly = np.pi
        c2.ideal_transfer_orbit = []

        # Add to the command line the burn needed
        self.command_line.append(c1)
        self.command_line.append(c2)

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

    def plane_correction(self, chaser, chaser_next, target):
        """
            Correct plane inclination and raan with one single manoeuvre
            [] Space Mechanics, Chapter 9
        """

        # In a plane correction the following parameters are changed:
        # - RAAN
        # - Inclination
        # While those remain the same:
        # - Eccentricity
        # - Semi-Major Axis
        #
        # Normally, a plane change is better suited with a circularized orbit.
        # Therefore, in this function it is assumed that the orbit is circular, and the nodes where the burn has to be
        # executed is calculated according to that.

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

        theta_i = A_Li - chaser.kep.w
        theta_i_tmp = theta_i + np.pi

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

        # Evaluate velocity vector in Earth-Inertial reference frame at theta_i
        V_PERI_i = np.sqrt(mu_earth / (a * (1.0 - e**2))) * np.array([-np.sin(theta_i_tmp), e + np.cos(theta_i_tmp), 0.0])
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
        # # Try to propagate
        # self._propagator(chaser, target, self.travel_time(chaser, chaser.kep.v, theta_i))
        # self._propagator(chaser, target, dt, deltaV_C)

        # Create command
        c1 = Command()
        c1.deltaV_C = deltaV_C
        c1.true_anomaly = theta_i
        c1.ideal_transfer_orbit = []

    def travel_time(self, chaser, theta0, theta1):

        a = chaser.kep.a
        e = chaser.kep.e

        theta0 = theta0 % (2.0 * np.pi)
        theta1 = theta1 % (2.0 * np.pi)

        t0 = np.sqrt(a**3/mu_earth) * (2.0 * np.arctan((np.sqrt((1.0 - e)/(1.0 + e)) * np.tan(theta0 / 2.0))) -
                                       (e * np.sqrt(1.0 - e**2) * np.sin(theta0))/(1.0 + e * np.cos(theta0)))
        t1 = np.sqrt(a**3/mu_earth) * (2.0 * np.arctan((np.sqrt((1.0 - e)/(1.0 + e)) * np.tan(theta1 / 2.0))) -
                                       (e * np.sqrt(1.0 - e**2) * np.sin(theta1))/(1.0 + e * np.cos(theta1)))

        return t1 - t0

    def calc_synodal_period(self, chaser, chaser_next):
        """
            Calculate the synodal period
        """
        pass

