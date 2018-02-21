# @copyright Copyright (c) 2017, Davide Frey (frey.davide.ae@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

"""Class holding the definition of the Solver, which outputs a manoeuvre plan given a scenario."""

import numpy as np
import scipy.io as sio
import pykep as pk
import os

from space_tf import Cartesian, mu_earth, R_earth
from manoeuvre import Manoeuvre, RelativeMan
from state import Satellite, Chaser
from checkpoint import RelativeCP, AbsoluteCP


class Solver(object):
    """
        Base solver class.

    Attributes:
        manoeuvre_plan (list): List of the manoeuvre that has to be executed to perform the scenario
    """

    def __init__(self):
        self.manoeuvre_plan = []

    def solve_scenario(self, scenario):
        """
            Function that solve a given scenario.

        Args:
            scenario (Scenario): Planned scenario.
        """

        print "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        print "                      Solving the scenario: " + scenario.name
        print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
        print "Scenario overview: "
        print scenario.overview

        # Extract scenario checkpoints
        checkpoints = scenario.checkpoints

        # Extract target and chaser states
        chaser = scenario.chaser
        target = scenario.target

        # Extract keep out zone radius
        koz_r = scenario.koz_r

        # Start solving scenario by popping positions from position list
        for checkpoint in checkpoints:
            self.print_state(chaser, target)
            self.to_next_checkpoint(chaser, checkpoint, target, koz_r)

        tot_dV, tot_dt = self.print_result()

        print "Final achieved relative position:     " + str(chaser.rel_state.R)

        print "\n\n-----------------> Manoeuvre elaborated <--------------------\n"
        print "---> Manoeuvre duration:    " + str(tot_dt) + " seconds"
        print "---> Total deltaV:          " + str(tot_dV) + " km/s"

    def to_next_checkpoint(self, chaser, checkpoint, target, koz_r):
        """
            Solve the problem of going from the actual position (chaser) to the next position (checkpoint), and output
            the manoeuvres that have to be performed.

        Args:
            chaser (Chaser): Chaser state
            checkpoint (RelativeCP or AbsoluteCP): Target checkpoint, either absolute or relative defined.
            target (Satellite): Target state
        """

        t_limit = 604800

        if hasattr(checkpoint, 'rel_state'):
            # Relative navigation
            # TODO: Review this part

            # Calculate cartesian coordinates
            chaser_cart = Cartesian()
            target_cart = Cartesian()

            target_cart.from_keporb(target.abs_state)
            chaser_cart.from_lvlh_frame(target_cart, chaser.rel_state)

            # Drift allowed for this point
            t_est = self.drift_to(chaser, checkpoint, target)

            # Time needed from actual position to perigee
            t_to_perigee = self.travel_time(chaser, chaser.abs_state.v, 2.0 * np.pi)

            # If the target cannot drift easily to the wanted position, move to a better coelliptic orbit
            if t_est is None and np.linalg.norm(chaser.rel_state.R) > 20.0:
                # Assume chaser always below target
                n_drift = np.sqrt(mu_earth / chaser.abs_state.a**3) - np.sqrt(mu_earth / target.abs_state.a**3)
                t_drift = (target.abs_state.v - chaser.abs_state.v) % (2.0 * np.pi) / n_drift

                dv_act = target.abs_state.v - chaser.abs_state.v
                dv_at_perigee = dv_act - n_drift * t_to_perigee

                # Chaser cannot drift to the wanted position, has to be adjusted to another orbit to be able to drift
                # Evaluate wanted radius difference to move to a coelliptic orbit with that difference
                r_diff = checkpoint.rel_state.R[0]

                # Create new checkpoint
                checkpoint_new_abs = AbsoluteCP()
                checkpoint_new_rel = RelativeCP()

                checkpoint_new_abs.set_abs_state(chaser.abs_state)
                checkpoint_new_abs.abs_state.a = target.abs_state.a + r_diff
                checkpoint_new_abs.abs_state.e = target.abs_state.a * target.abs_state.e / checkpoint_new_abs.abs_state.a

                checkpoint_new_rel.rel_state.R = checkpoint.rel_state.R
                checkpoint_new_rel.rel_state.V = checkpoint.rel_state.V

                checkpoint_new_rel.error_ellipsoid = checkpoint.error_ellipsoid

                k = (dv_at_perigee / n_drift - np.pi * np.sqrt(
                    checkpoint_new_abs.abs_state.a ** 3 / mu_earth) - t_limit) / np.sqrt(chaser.abs_state.a ** 3 / mu_earth)

                if k > 0.0:
                    # Wait np.ceil(k) revolutions to ensure we will be below t_limit after the semimajoraxis correction
                    self._propagator(chaser, target, np.ceil(k) * np.sqrt(chaser.abs_state.a ** 3 / mu_earth))

                # Adjust orbit though a standard manoeuvre
                self.adjust_eccentricity_semimajoraxis(chaser, checkpoint_new_abs, target)
                self.print_state(chaser, target)

                # Evaluate the new drift time
                t_est = self.drift_to(chaser, checkpoint_new_rel, target)

            if t_est is None and np.linalg.norm(chaser.rel_state.R) <= 20.0:
                # Distance from the target is below 20.0 km => use CW-solver
                # self.clohessy_wiltshire_solver(chaser, checkpoint, target)
                self.multi_lambert(chaser, checkpoint, target, koz_r)

            # Check if the drift time is below a certain limit
            # FOR NOW: drift in any case, just does not care about a time limit
            if t_est is not None:
                # Drift, propagate chaser and target for t_est
                # Add drift command to the command line
                c = RelativeMan()
                c.dV = np.array([0.0, 0.0, 0.0])
                c.set_abs_state(chaser.abs_state)
                c.set_rel_state(chaser.rel_state)
                c.duration = t_est
                c.description = 'Drift for ' + str(t_est) + ' seconds'
                self.manoeuvre_plan.append(c)

                self._propagator(chaser, target, t_est)

            elif t_est is not None and t_est > t_limit:
                # Resync manoeuvre
                n_mean_T = np.sqrt(mu_earth / target.kep.a**3)
                n_mean_C = np.sqrt(mu_earth / chaser.kep.a**3)
                n_rel_mean = n_mean_C - n_mean_T

                dv_act = target.kep.v - chaser.kep.v
                dv_at_perigee = dv_act - n_rel_mean * t_to_perigee

                a_min = (mu_earth / (dv_act / t_limit + n_mean_T)**2)**(1.0/3.0)
                e_min = 1.0 - chaser.kep.a / a_min * (1.0 - chaser.kep.e)
                pass

        else:
            # Absolute navigation
            # Define tolerances, if we get deviations greater than ~1 km then correct
            tol_i = 1.0 / chaser.abs_state.a
            tol_O = 1.0 / chaser.abs_state.a
            tol_w = 1.0 / chaser.abs_state.a
            tol_a = 0.2
            tol_e = 1.0 / chaser.abs_state.a

            # Correct for the plane angles,
            # Inclination and RAAN
            di = checkpoint.abs_state.i - chaser.abs_state.i
            dO = checkpoint.abs_state.O - chaser.abs_state.O
            if abs(di) > tol_i or abs(dO) > tol_O:
                self.plane_correction(chaser, checkpoint, target)
                self.print_state(chaser, target)

            # Argument of Perigee
            dw = checkpoint.abs_state.w - chaser.abs_state.w
            if abs(dw) > tol_w:
                self.adjust_perigee(chaser, checkpoint, target)
                self.print_state(chaser, target)

            # Eccentricity and Semi-Major Axis
            da = checkpoint.abs_state.a - chaser.abs_state.a
            de = checkpoint.abs_state.e - chaser.abs_state.e
            if abs(da) > tol_a or abs(de) > tol_e:
                self.adjust_eccentricity_semimajoraxis(chaser, checkpoint, target)
                self.print_state(chaser, target)

    def adjust_eccentricity_semimajoraxis(self, chaser, checkpoint, target):
        """
            Adjust eccentricity and semi-major axis at the same time with an Hohmann-Transfer like manouevre:
            1) Burn at perigee to match the needed intermediate orbit
            2) Burn at apogee to arrive at the final, wanted orbit

        References:
            Howard Curtis, Orbital Mechanics for Engineering Students, Chapter 6
            David A. Vallado, Fundamentals of Astrodynamics and Applications, Second Edition, Chapter 6

        Args:
            chaser (Chaser): Chaser state.
            checkpoint (AbsoluteCP or RelativeCP): Next checkpoint.
            target (Satellite): Target state.
        """

        # Extract initial and final semi-major axis and eccentricities
        a_i = chaser.abs_state.a
        e_i = chaser.abs_state.e
        a_f = checkpoint.abs_state.a
        e_f = checkpoint.abs_state.e

        r_p_i = a_i * (1.0 - e_i)
        r_p_f = a_f * (1.0 - e_f)

        r_a_i = a_i * (1.0 + e_i)
        r_a_f = a_f * (1.0 + e_f)

        if a_f > a_i:
            # Calculate intermediate orbital elements
            a_int = (r_a_f + r_p_i) / 2.0
            e_int = 1.0 - r_p_i / a_int

            # First burn at perigee, then apogee
            theta_1 = 0.0
            theta_2 = np.pi
        else:
            # Calculate intermediate orbital elements
            a_int = (r_a_i + r_p_f) / 2.0
            e_int = 1.0 - r_p_f / a_int

            # First burn at apogee, then perigee
            theta_1 = np.pi
            theta_2 = 0.0

        # Calculate delta-V's in perifocal frame of reference
        # First burn
        V_PERI_i_1 = np.sqrt(mu_earth / (a_i * (1.0 - e_i**2))) * np.array([-np.sin(theta_1), e_i + np.cos(theta_1), 0.0])
        V_PERI_f_1 = np.sqrt(mu_earth / (a_int * (1.0 - e_int**2))) * np.array([-np.sin(theta_1), e_int + np.cos(theta_1), 0.0])

        deltaV_C_1 = np.linalg.inv(chaser.abs_state.get_pof()).dot(V_PERI_f_1 - V_PERI_i_1)

        # Second burn
        V_PERI_i_2 = np.sqrt(mu_earth / (a_int * (1.0 - e_int ** 2))) * np.array([-np.sin(theta_2), e_int + np.cos(theta_2), 0.0])
        V_PERI_f_2 = np.sqrt(mu_earth / (a_f * (1.0 - e_f ** 2))) * np.array([-np.sin(theta_2), e_f + np.cos(theta_2), 0.0])

        deltaV_C_2 = np.linalg.inv(chaser.abs_state.get_pof()).dot(V_PERI_f_2 - V_PERI_i_2)

        # Create commands
        c1 = Manoeuvre()
        c1.dV = deltaV_C_1
        c1.set_abs_state(chaser.abs_state)
        c1.abs_state.v = theta_1
        c1.duration = self.travel_time(chaser, chaser.abs_state.v, theta_1)
        c1.description = 'Apogee/Perigee raise'
        self.manoeuvre_plan.append(c1)

        # Propagate chaser and target
        self._propagator(chaser, target, c1.duration)
        self._propagator(chaser, target, 1e-3, deltaV_C_1)

        c2 = Manoeuvre()
        c2.dV = deltaV_C_2
        c2.set_abs_state(chaser.abs_state)
        c2.abs_state.v = theta_2
        c2.duration = np.pi * np.sqrt(a_int**3 / mu_earth)
        c2.description = 'Apogee/Perigee raise'
        self.manoeuvre_plan.append(c2)

        # Propagate chaser and target
        self._propagator(chaser, target, c2.duration)
        self._propagator(chaser, target, 1e-3, deltaV_C_2)

    def adjust_perigee(self, chaser, checkpoint, target):
        """
            Given the chaser relative orbital elements with respect to the target adjust the perigee argument.

        References:
            Howard Curtis, Orbital Mechanics for Engineering Students, Chapter 6
            David A. Vallado, Fundamentals of Astrodynamics and Applications, Second Edition, Chapter 6

        Args:
            chaser (Chaser): Chaser state.
            checkpoint (AbsoluteCP or RelativeCP): Next checkpoint.
            target (Satellite): Target state.
        """

        # Easy fix for the case of circular orbit (TO BE REVIEWED - Bigger orbits may lead to bigger errors)
        if chaser.abs_state.e < 1e-12:
            # When the orbit is almost circular approximate the argument of perigee to be at 0
            chaser.abs_state.v = (chaser.abs_state.v + chaser.abs_state.w) % (2.0 * np.pi)
            chaser.abs_state.w = 0
        else:
            # Extract constants
            a = chaser.abs_state.a
            e = chaser.abs_state.e

            # Evaluate perigee difference to correct
            dw = (checkpoint.abs_state.w - chaser.abs_state.w) % (2.0 * np.pi)

            # Positions where burn can occur
            theta_i_1 = dw / 2.0
            theta_i_2 = theta_i_1 + np.pi
            theta_f_1 = 2.0 * np.pi - theta_i_1
            theta_f_2 = theta_f_1 - np.pi

            # Check which one is the closest
            if theta_i_1 < chaser.abs_state.v:
                dv1 = 2.0 * np.pi + theta_i_1 - chaser.abs_state.v
            else:
                dv1 = theta_i_1 - chaser.abs_state.v

            if theta_i_2 < chaser.abs_state.v:
                dv2 = 2.0 * np.pi + theta_i_2 - chaser.abs_state.v
            else:
                dv2 = theta_i_2 - chaser.abs_state.v

            if dv1 > dv2:
                theta_i = theta_i_2
                theta_f = theta_f_2
            else:
                theta_i = theta_i_1
                theta_f = theta_f_1

            # Initial velocity
            V_PERI_i = np.sqrt(mu_earth / (a * (1.0 - e**2))) * np.array([-np.sin(theta_i), e + np.cos(theta_i), 0.0])
            V_TEM_i = np.linalg.inv(chaser.abs_state.get_pof()).dot(V_PERI_i)

            # Final velocity
            V_PERI_f = np.sqrt(mu_earth / (a * (1.0 - e**2))) * np.array([-np.sin(theta_f), e + np.cos(theta_f), 0.0])
            V_TEM_f = np.linalg.inv(checkpoint.abs_state.get_pof()).dot(V_PERI_f)

            # Delta-V
            deltaV_C = V_TEM_f - V_TEM_i

            # Create command
            c = Manoeuvre()
            c.dV = deltaV_C
            c.set_abs_state(chaser.abs_state)
            c.abs_state.v = theta_i
            c.duration = self.travel_time(chaser, chaser.abs_state.v, theta_i)
            c.description = 'Argument of Perigee correction'
            self.manoeuvre_plan.append(c)

            # Propagate chaser and target
            self._propagator(chaser, target, c.duration)
            self._propagator(chaser, target, 1e-3, deltaV_C)

    def plane_correction(self, chaser, checkpoint, target):
        """
            Correct plane inclination and RAAN with a single manoeuvre at the node between the two orbital planes.

        References:
            Howard Curtis, Orbital Mechanics for Engineering Students, Chapter 6
            David A. Vallado, Fundamentals of Astrodynamics and Applications, Second Edition, Chapter 6

        Args:
            chaser (Chaser): Chaser state.
            checkpoint (AbsoluteCP or RelativeCP): Next checkpoint.
            target (Satellite): Target state.
        """

        # Extract semi-major axis and eccentricity
        a = chaser.abs_state.a
        e = chaser.abs_state.e

        # Changing values
        O_i = chaser.abs_state.O
        O_f = checkpoint.abs_state.O
        dO = O_f - O_i
        i_i = chaser.abs_state.i
        i_f = checkpoint.abs_state.i
        di = i_f - i_i

        # Spherical trigonometry
        alpha = np.arccos(np.sin(i_i) * np.sin(i_f) * np.cos(dO) + np.cos(i_i) * np.cos(i_f))
        A_Li = np.arcsin(np.sin(i_f) * np.sin(dO) / np.sin(alpha))

        B_Lf = np.arcsin(np.sqrt(np.cos(i_f)**2 * np.sin(i_i)**2 * np.sin(dO)**2 /
                                 (np.sin(alpha)**2 - np.sin(i_i)**2 * np.sin(i_f)**2 * np.sin(dO)**2)))

        if (i_f > np.pi / 2.0 > i_i) or (i_i > np.pi / 2.0 > i_f):
            B_Lf *= -np.sign(dO)
        elif (i_f > i_i > np.pi / 2.0) or (i_i > i_f > np.pi / 2.0):
            B_Lf *= -np.sign(dO) * np.sign(di)
        else:
            B_Lf *= np.sign(dO) * np.sign(di)

        phi = O_f + B_Lf

        psi = np.sign(dO) * abs(np.arcsin(np.sin(i_i) * np.sin(i_f) * np.sin(dO) / np.sin(alpha)))

        if i_i > i_f:
            psi *= -1.0

        A_Li = -abs(A_Li) * np.sign(psi)

        # Two possible positions where the burn can occur
        theta_1 = (2.0 * np.pi - A_Li - chaser.abs_state.w) % (2.0 * np.pi)
        theta_2 = (theta_1 + np.pi) % (2.0 * np.pi)

        # Choose which of the two position is the closest
        # They consume different dV, the decision has to be taken then depending on if you want to spent a bit more
        # and burn in a specific point, or if you can born anywhere regardless on how much it will cost.
        # Now it's just taking the closest point to do the burn, to decrease the total time of the mission.
        if theta_1 < chaser.abs_state.v:
            dv1 = 2*np.pi + theta_1 - chaser.abs_state.v
        else:
            dv1 = theta_1 - chaser.abs_state.v

        if theta_2 < chaser.abs_state.v:
            dv2 = 2*np.pi + theta_2 - chaser.abs_state.v
        else:
            dv2 = theta_2 - chaser.abs_state.v

        if dv1 > dv2:
            theta_i = theta_2
        else:
            theta_i = theta_1

        # Define vector c in Earth-Inertial frame of reference
        cx = np.cos(psi) * np.cos(phi)
        cy = np.cos(psi) * np.sin(phi)
        cz = np.sin(psi)

        if i_i > i_f:
            cx *= -1.0
            cy *= -1.0
            cz *= -1.0

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
        V_TEM_i = np.linalg.inv(chaser.abs_state.get_pof()).dot(V_PERI_i)

        # Rotate vector around c by alpha radiants
        V_TEM_f = R_c.dot(V_TEM_i)

        # Evaluate deltaV
        deltaV_C = V_TEM_f - V_TEM_i

        # Create command
        c = Manoeuvre()
        c.dV = deltaV_C
        c.set_abs_state(chaser.abs_state)
        c.abs_state.v = theta_i
        c.duration = self.travel_time(chaser, chaser.abs_state.v, theta_i)
        c.description = 'Inclination and RAAN correction'
        self.manoeuvre_plan.append(c)

        # Propagate chaser and target
        self._propagator(chaser, target, c.duration)
        self._propagator(chaser, target, 1e-3, deltaV_C)

    def drift_to(self, chaser, checkpoint, target):
        """
            Algorithm that tries to drift to the next checkpoint, staying within a certain error ellipsoid.

        Args:
            chaser (Chaser): Chaser state.
            checkpoint (AbsoluteCP or RelativeCP): Next checkpoint.
            target (Satellite): Target state.
        """
        # Calculate the cartesian coordinates of target and chaser
        chaser_cart = Cartesian()
        target_cart = Cartesian()

        chaser_cart.from_keporb(chaser.abs_state)
        target_cart.from_keporb(target.abs_state)

        r_C = chaser.abs_state.a * (1.0 - chaser.abs_state.e ** 2) / (1.0 + chaser.abs_state.e * np.cos(chaser.abs_state.v))
        r_T = target.abs_state.a * (1.0 - target.abs_state.e ** 2) / (1.0 + target.abs_state.e * np.cos(chaser.abs_state.v))

        # Assuming that if we enter that point we are on a coelliptic orbit, we can directly say:
        if abs(checkpoint.rel_state.R[0] - r_C + r_T) >= checkpoint.error_ellipsoid[0] or abs(chaser.abs_state.a - target.abs_state.a) < 2e-1:
            return None

        chaser_old = Chaser()
        chaser_old.set_from_other_satellite(chaser)
        target_old = Satellite()
        target_old.set_from_other_satellite(target)

        n_c = np.sqrt(mu_earth / chaser.abs_state.a ** 3)
        n_t = np.sqrt(mu_earth / target.abs_state.a ** 3)

        # If n_rel is below zero, we are moving slower than target. Otherwise faster.
        n_rel = n_c - n_t

        # Required dv at the end of the manoeuvre, estimation based on the relative position
        dv_req = np.sign(checkpoint.rel_state.R[1]) * np.linalg.norm(checkpoint.rel_state.R) / R_earth

        # Check if a drift to the wanted position is possible, if yes check if it can be done under a certain time,
        # if not try to resync
        actual_dv = (chaser.abs_state.v + chaser.abs_state.w) % (2.0*np.pi) - (target.abs_state.v + target.abs_state.w) % (2.0*np.pi)

        # Define a function F for the angle calculation
        F = lambda dv_req, dv, n: int((dv - dv_req) / n > 0.0) * np.sign(n)

        t_est = (2.0 * np.pi * F(dv_req, actual_dv, n_rel) + dv_req - actual_dv) / n_rel
        t_est_old = 0.0
        t_old = 0.0
        ellipsoid_flag = False
        tol = 1e-3         # Millisecond tolerance
        dt = 100.0
        dr_next_old = 0.0
        while abs(t_est - t_old) > tol:
            self._propagator(chaser_old, target_old, t_est)
            dr_next = chaser_old.rel_state.R[1] - checkpoint.rel_state.R[1]

            t_old = t_est

            if dr_next <= 0.0 and dr_next_old <= 0.0:
                t_est_old = t_est
                t_est += dt
            elif dr_next >= 0.0 and dr_next_old >= 0.0:
                t_est_old = t_est
                t_est -= dt
            elif (dr_next <= 0.0 and dr_next_old >= 0.0) or (dr_next >= 0.0 and dr_next_old <= 0.0):
                t_est = (t_est_old + t_est) / 2.0
                t_est_old = t_old
                dt /= 10.0


            dr_next_old = dr_next

            # Assuming to stay on the same plane
            if abs(checkpoint.rel_state.R[0] - chaser_old.rel_state.R[0]) <= checkpoint.error_ellipsoid[0] and \
                abs(checkpoint.rel_state.R[1] - chaser_old.rel_state.R[1]) <= checkpoint.error_ellipsoid[1]:
                # We have reached the error ellipsoid, can break
                ellipsoid_flag = True

            chaser_old.set_from_other_satellite(chaser)
            target_old.set_from_other_satellite(target)

        if ellipsoid_flag:
            # With the estimated time, we are in the error-ellipsoid
            return t_est
        else:
            return None

    def multi_lambert(self, chaser, checkpoint, target, koz_r):
        """
            Solve the Multi-Lambert Problem.

        Args:
            chaser (Chaser): Chaser state.
            checkpoint (AbsoluteCP or RelativeCP): Next checkpoint (theoretically used only for RelativeCP).
            target (Satellite): Target state.
        """

        # Calculate the cartesian coordinates of target and chaser
        chaser_cart = Cartesian()
        target_cart = Cartesian()

        chaser_cart.from_keporb(chaser.abs_state)
        target_cart.from_keporb(target.abs_state)

        # Absolute position of chaser at t = t0
        R_C_i = chaser_cart.R
        V_C_i = chaser_cart.V

        # Absolute position of the target at t = t0
        R_T_i = target_cart.R
        V_T_i = target_cart.V

        best_deltaV = 1e12
        best_dt = 0

        # Minimum deltaV deliverable -> 5 mm/s
        min_deltaV = 5e-6

        # Check all the possible transfers time from tmin to tmax (seconds)
        tmin = 10
        tmax = 30000
        for dt in xrange(tmin, tmax):
            # Propagate target position at t1 = t0 + dt
            target_old = Cartesian()
            target_old.R = R_T_i
            target_old.V = V_T_i

            r_T, v_T = pk.propagate_lagrangian(target_old.R, target_old.V, dt, mu_earth)

            target_old.R = np.array(r_T)
            target_old.V = np.array(v_T)

            # Transformation matrix from TEME to LVLH at time t1
            B_LVLH_TEME_f = target_old.get_lof()

            # Evaluate final wanted absolute position of the chaser
            R_C_f = np.array(target_old.R) + np.linalg.inv(B_LVLH_TEME_f).dot(checkpoint.rel_state.R)
            O_T_f = np.cross(target_old.R, target_old.V) / np.linalg.norm(target_old.R)**2
            V_C_f = np.array(target_old.V) + np.array([0.0, 0.0, 0.0]) + np.cross(O_T_f, np.linalg.inv(B_LVLH_TEME_f).dot(checkpoint.rel_state.R))

            # Solve lambert in dt starting from the chaser position at t0 going to t1
            sol = pk.lambert_problem(R_C_i, R_C_f, dt, mu_earth, True, 10)

            # Check for the best solution for this dt
            for i in xrange(0, len(sol.get_v1())):
                deltaV_1 = np.array(sol.get_v1()[i]) - V_C_i
                deltaV_2 = V_C_f - np.array(sol.get_v2()[i])
                deltaV_tot = np.linalg.norm(deltaV_1) + np.linalg.norm(deltaV_2)

                # Check if the deltaV is above the minimum deliverable by thrusters
                if np.linalg.norm(deltaV_1) > min_deltaV and np.linalg.norm(deltaV_2) > min_deltaV:
                    # Check if the new deltaV is less than previous
                    if deltaV_tot < best_deltaV:
                        # Check if the trajectory is safe
                        if self.is_trajectory_safe(chaser_cart, target_cart, dt, koz_r):
                            best_deltaV = deltaV_tot
                            best_deltaV_1 = deltaV_1
                            best_deltaV_2 = deltaV_2
                            best_dt = dt

        c1 = RelativeMan()
        c1.dV = best_deltaV_1
        c1.set_abs_state(chaser.abs_state)
        c1.set_rel_state(chaser.rel_state)
        c1.duration = 0
        c1.description = 'Multi-Lambert solution'
        self.manoeuvre_plan.append(c1)

        self._propagator(chaser, target, 1e-3, best_deltaV_1)
        self._propagator(chaser, target, best_dt)

        chaser_cart.from_keporb(chaser.abs_state)

        c2 = RelativeMan()
        c2.dV = best_deltaV_2
        c2.set_abs_state(chaser.abs_state)
        c2.set_rel_state(chaser.rel_state)
        c2.duration = best_dt
        c2.description = 'Multi-Lambert solution'
        self.manoeuvre_plan.append(c2)

        self._propagator(chaser, target, 1e-3, best_deltaV_2)

    def clohessy_wiltshire_solver(self, chaser, checkpoint, target):
        """
            Solve Hill's Equation to get the amount of DeltaV needed to go to the next checkpoint.

        References:
            David A. Vallado, Fundamentals of Astrodynamics and Applications, Second Edition, Algorithm 47 (p. 382)

        Args:
            chaser (Chaser): Chaser state.
            checkpoint (RelativeCP): Next checkpoint.
            target (Satellite): Target state.
        """

        # TODO: Apply correction according to the new definition of objects
        # TODO: Consider thruster accuracy

        print ">>>> Solving CW-equations\n"

        a = target.abs_state.a
        max_time = int(2*np.pi * np.sqrt(a**3 / mu_earth))

        r_rel_c_0 = chaser.rel_state.R
        v_rel_c_0 = chaser.rel_state.V

        r_rel_c_n = checkpoint.rel_state.R
        v_rel_c_n = [0, 0, 0]

        n = np.sqrt(mu_earth/a**3.0)

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

        for t_ in xrange(1, max_time):
            rv_t = phi_rv(t_)
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

        target_cart = Cartesian()
        target_cart.from_keporb(target.abs_state)

        # Change frame of reference of deltaV. From LVLH to Earth-Inertial
        B = target_cart.get_lof()
        deltaV_C_1 = np.linalg.inv(B).dot(best_deltaV_1)

        # Create command
        c1 = RelativeMan()
        c1.dV = deltaV_C_1
        c1.set_abs_state(chaser.abs_state)
        c1.set_rel_state(chaser.rel_state)
        c1.duration = 0
        c1.description = 'CW approach'

        # Propagate chaser and target
        self._propagator(chaser, target, 1e-5, deltaV_C_1)

        self._propagator(chaser, target, delta_T)

        self.print_state(chaser, target)

        self.manoeuvre_plan.append(c1)

        target_cart.from_keporb(target.abs_state)

        R = target_cart.get_lof()
        deltaV_C_2 = np.linalg.inv(R).dot(best_deltaV_2)

        # Create command
        c2 = RelativeMan()
        c2.dV = deltaV_C_2
        c2.set_abs_state(chaser.abs_state)
        c2.set_rel_state(chaser.rel_state)
        c2.duration = delta_T
        c2.description = 'CW approach'

        # Propagate chaser and target to evaluate all the future commands properly
        self._propagator(chaser, target, 1e-5, deltaV_C_2)

        self.print_state(chaser, target)

        self.manoeuvre_plan.append(c2)

    def _propagator(self, chaser, target, dt, dv=np.array([0, 0, 0])):
        """
            Propagate chaser and target to t* = now + dt.

        Args:
            chaser (Chaser): Chaser state.
            target (Satellite): Target state.
            dt (float64): Propagated time in seconds.
            dv (np.array): Amount of delta-v to be applied at the beginning of the propagation, in km/s in TEME
                reference frame.
        """

        # TODO: Change to a propagator that includes at least J2 disturbance

        # Calculate cartesian components
        chaser_cart = Cartesian()
        chaser_cart.from_keporb(chaser.abs_state)
        r_C = chaser_cart.R
        v_C = chaser_cart.V
        r_C, v_C = pk.propagate_lagrangian(r_C, v_C + dv, dt, mu_earth)
        chaser_cart.R = np.array(r_C)
        chaser_cart.V = np.array(v_C)
        chaser.abs_state.from_cartesian(chaser_cart)

        if target is not None:
            target_cart = Cartesian()
            target_cart.from_keporb(target.abs_state)
            r_T = target_cart.R
            v_T = target_cart.V
            r_T, v_T = pk.propagate_lagrangian(r_T, v_T, dt, mu_earth)
            target_cart.R = np.array(r_T)
            target_cart.V = np.array(v_T)
            target.abs_state.from_cartesian(target_cart)

            chaser.rel_state.from_cartesian_pair(chaser_cart, target_cart)

    def travel_time(self, chaser, theta0, theta1):
        """
            Evaluate the travel time from a starting true anomaly theta0 to an end anomaly theta1.

        Reference:
            Exercise of Nicollier's Lecture.
            David A. Vallado, Fundamentals of Astrodynamics and Applications, Second Edition, Algorithm 11 (p. 133)

        Args:
            chaser (Position): Position structure of the chaser
            theta0 (rad): Starting true anomaly
            theta1 (rad): Ending true anomaly

        Return:
            Travel time (seconds)
        """

        a = chaser.abs_state.a
        e = chaser.abs_state.e

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

    def print_state(self, chaser, target):

        print "Chaser state: "
        print " >> Kep: "
        print "      a :     " + str(chaser.abs_state.a)
        print "      e :     " + str(chaser.abs_state.e)
        print "      i :     " + str(chaser.abs_state.i)
        print "      O :     " + str(chaser.abs_state.O)
        print "      w :     " + str(chaser.abs_state.w)
        print "      v :     " + str(chaser.abs_state.v)

        print "Target state: "
        print " >> Kep: "
        print "      a :     " + str(target.abs_state.a)
        print "      e :     " + str(target.abs_state.e)
        print "      i :     " + str(target.abs_state.i)
        print "      O :     " + str(target.abs_state.O)
        print "      w :     " + str(target.abs_state.w)
        print "      v :     " + str(target.abs_state.v)

    def print_result(self):
        tot_dv = 0
        tot_dt = 0

        for it, command in enumerate(self.manoeuvre_plan):
            print '\n' + command.description + ', command nr. ' + str(it) + ':'
            print '--> DeltaV:            ' + str(command.dV)
            print '--> Normalized DeltaV: ' + str(np.linalg.norm(command.dV))
            print '--> Idle after burn:   ' + str(command.duration)
            print '--> Burn position:     ' + str(command.abs_state.v)
            tot_dv += np.linalg.norm(command.dV)
            tot_dt += command.duration

        return tot_dv, tot_dt

    def _save_result(self, chaser, target, id=0, single_manoeuvre=False):
        if os.path.isdir('/home/dfrey/polybox/manoeuvre'):
            if single_manoeuvre:
                print "Saving single manoeuvre " + str(id) + "..."
                L = 1
            else:
                print "Saving complete manoeuvre..."
                L = len(self.command_line)

            # Simulating the whole manoeuvre and store the result
            chaser_tmp = Chaser()
            target_tmp = Satellite()

            chaser_tmp.from_other_state(chaser)
            target_tmp.from_other_state(target)

            # Creating list of radius of target and chaser
            R_target = [target_tmp.cartesian.R]
            R_chaser = [chaser_tmp.cartesian.R]
            R_chaser_lvlh = [chaser_tmp.lvlh.R]
            # R_chaser_lvc = [np.array([chaser_tmp.lvc.dR, chaser_tmp.lvc.dV, chaser_tmp.lvc.dH])]

            for i in xrange(0, L):
                if single_manoeuvre:
                    cmd = self.command_line[-1]
                else:
                    cmd = self.command_line[i]

                for j in xrange(0, int(np.floor(cmd.duration))):
                    self._propagator(chaser_tmp, target_tmp, 1.0)
                    R_chaser.append(chaser_tmp.cartesian.R)
                    R_target.append(target_tmp.cartesian.R)
                    R_chaser_lvlh.append(chaser_tmp.lvlh.R)
                    # R_chaser_lvc.append(np.array([chaser_tmp.lvc.dR, chaser_tmp.lvc.dV, chaser_tmp.lvc.dH]))

                self._propagator(chaser_tmp, target_tmp, cmd.duration - np.floor(cmd.duration))

                # Apply dV
                self._propagator(chaser_tmp, target_tmp, 1e-3, cmd.deltaV_C)
                # self.print_state(chaser_tmp, target_tmp)

            # Saving in .mat file
            if single_manoeuvre:
                sio.savemat('/home/dfrey/polybox/manoeuvre/manoeuvre_' + str(id) + '.mat',
                        mdict={'abs_pos_c': R_chaser, 'rel_pos_c': R_chaser_lvlh, 'abs_pos_t': R_target})
            else:
                sio.savemat('/home/dfrey/polybox/manoeuvre/complete_manoeuvre.mat',
                        mdict={'abs_pos_c': R_chaser, 'rel_pos_c': R_chaser_lvlh, 'abs_pos_t': R_target})

            print "Manoeuvre saved."

    def is_trajectory_safe(self, chaser_cart, target_cart, dt, koz_r):

        d_SF = 0.1
        t_SF = 36000.0

        T = np.arange(0.0, dt + t_SF, 1.0)

        r = chaser_cart.R
        v = chaser_cart.V

        r_t = target_cart.R
        v_t = target_cart.V

        for t in T:
            r, v = pk.propagate_lagrangian(r, v, 1.0, mu_earth)
            r_t, v_t = pk.propagate_lagrangian(r_t, v_t, 1.0, mu_earth)

            r = np.array(r)
            v = np.array(v)

            r_t = np.array(r_t)
            v_t = np.array(v_t)

            if (r[0] - r_t[0])**2 + (r[1] - r_t[1])**2 + (r[1] - r_t[1])**2 <= (koz_r + d_SF)**2:
                return False

        return True
