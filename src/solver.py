# @copyright Copyright (c) 2017, Davide Frey (frey.davide.ae@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

"""Class holding the definition of the Solver, which outputs a manoeuvre plan given a scenario."""

import numpy as np
import pykep as pk

from space_tf import Cartesian, mu_earth, R_earth, KepOrbElem, J_2
from manoeuvre import Manoeuvre, RelativeMan
from state import Satellite, Chaser
from checkpoint import RelativeCP, AbsoluteCP
from scenario import Scenario
from datetime import timedelta


class Solver(object):
    """
        Base solver class.

    Attributes:
        manoeuvre_plan (list): List of the manoeuvre that has to be executed to perform the scenario
    """

    def __init__(self):
        self.manoeuvre_plan = []

        self.scenario = Scenario()

        self.chaser = Chaser()
        self.target = Satellite()

        self.epoch = self.scenario.date

    def set_ic(self, scenario):
        self.scenario = scenario
        self.chaser = scenario.chaser_ic
        self.target = scenario.target_ic

    def solve_scenario(self):
        """
            Function that solve a given scenario.

        Args:
            scenario (Scenario): Planned scenario exported from yaml configuration file.
        """

        print "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        print "                      SOLVING SCENARIO: " + self.scenario.name
        print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
        print "Scenario overview: "
        print self.scenario.overview

        # Extract scenario checkpoints
        checkpoints = self.scenario.checkpoints

        # Extract the approach ellipsoid
        approach_ellipsoid = self.scenario.approach_ellipsoid

        print "--------------------Chaser initial state-------------------"
        print " >> Osc Elements:"
        self.print_state(self.chaser.abs_state)
        print "\n >> Mean Elements:"
        chaser_mean = KepOrbElem()
        chaser_mean.from_osc_elems(self.chaser.abs_state, self.scenario.settings)
        self.print_state(chaser_mean)
        print "\n--------------------Target initial state-------------------"
        print " >> Osc Elements:"
        self.print_state(self.target.abs_state)
        print "\n >> Mean Elements:"
        target_mean = KepOrbElem()
        target_mean.from_osc_elems(self.target.abs_state, self.scenario.settings)
        self.print_state(target_mean)
        print "------------------------------------------------------------\n"


        # Start solving scenario by popping positions from position list
        for checkpoint in checkpoints:
            print "\n\n======================================================================="
            print "[GOING TO CHECKPOINT NR. " + str(checkpoint.id) + "]"
            print "======================================================================="
            print "[CHECKPOINT]:"
            if hasattr(checkpoint, 'abs_state'):
                print " >> Kep:"
                self.print_state(checkpoint.abs_state)
            else:
                print " >> LVLH:"
                print "      R:     " + str(checkpoint.rel_state.R)
                print "      V:     " + str(checkpoint.rel_state.V)
            print "======================================================================="
            if hasattr(checkpoint, 'rel_state'):
                # Relative navigation
                self.relative_solver(checkpoint, approach_ellipsoid)
            else:
                # Absolute navigation
                self.absolute_solver(checkpoint)
            print "[REACHED STATE]:"
            chaser_mean = KepOrbElem()
            chaser_mean.from_osc_elems(self.chaser.abs_state, self.scenario.settings)
            print " >> Kep:"
            self.print_state(chaser_mean)
            print "\n >> LVLH:"
            print "      R:     " + str(self.chaser.rel_state.R)
            print "      V:     " + str(self.chaser.rel_state.V)
            print "=======================================================================\n"

        tot_dV, tot_dt = self.print_result()

        print "Final achieved relative position:     " + str(self.chaser.rel_state.R)

        print "\n\n-----------------> Manoeuvre elaborated <--------------------\n"
        print "---> Manoeuvre duration:    " + str(tot_dt) + " seconds"
        print "---> Total deltaV:          " + str(tot_dV) + " km/s"

    def adjust_eccentricity_semimajoraxis(self, checkpoint):
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
        # Evaluating mean orbital elements
        chaser_mean = KepOrbElem()
        chaser_mean.from_osc_elems(self.chaser.abs_state, self.scenario.settings)

        # Extract initial and final semi-major axis and eccentricities
        a_i = chaser_mean.a
        e_i = chaser_mean.e
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

        deltaV_C_1 = np.linalg.inv(chaser_mean.get_pof()).dot(V_PERI_f_1 - V_PERI_i_1)

        # Second burn
        V_PERI_i_2 = np.sqrt(mu_earth / (a_int * (1.0 - e_int ** 2))) * np.array([-np.sin(theta_2), e_int + np.cos(theta_2), 0.0])
        V_PERI_f_2 = np.sqrt(mu_earth / (a_f * (1.0 - e_f ** 2))) * np.array([-np.sin(theta_2), e_f + np.cos(theta_2), 0.0])

        deltaV_C_2 = np.linalg.inv(chaser_mean.get_pof()).dot(V_PERI_f_2 - V_PERI_i_2)

        # Create commands
        c1 = Manoeuvre()
        c1.dV = deltaV_C_1
        c1.set_abs_state(chaser_mean)
        c1.abs_state.v = theta_1
        c1.duration = self.travel_time(chaser_mean, chaser_mean.v, theta_1)
        c1.description = 'Apogee/Perigee raise'
        self.manoeuvre_plan.append(c1)

        # Propagate chaser and target
        self._propagator(c1.duration, deltaV_C_1)

        chaser_mean.from_osc_elems(self.chaser.abs_state, self.scenario.settings)

        c2 = Manoeuvre()
        c2.dV = deltaV_C_2
        c2.set_abs_state(chaser_mean)
        c2.abs_state.v = theta_2
        c2.duration = np.pi * np.sqrt(chaser_mean.a**3 / mu_earth)
        c2.description = 'Apogee/Perigee raise'
        self.manoeuvre_plan.append(c2)

        # Propagate chaser and target
        self._propagator(c2.duration, deltaV_C_2)

        chaser_mean.from_osc_elems(self.chaser.abs_state, self.scenario.settings)

    def adjust_perigee(self, checkpoint):
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
        # Mean orbital elements
        chaser_mean = KepOrbElem()
        chaser_mean.from_osc_elems(self.chaser.abs_state, self.scenario.settings)

        # Extract constants
        a = chaser_mean.a
        e = chaser_mean.e

        # Evaluate perigee difference to correct
        dw = (checkpoint.abs_state.w - chaser_mean.w) % (2.0 * np.pi)

        # Positions where burn can occur
        theta_i_1 = dw / 2.0
        theta_i_2 = theta_i_1 + np.pi
        theta_f_1 = 2.0 * np.pi - theta_i_1
        theta_f_2 = theta_f_1 - np.pi

        # Check which one is the closest
        if theta_i_1 < chaser_mean.v:
            dv1 = 2.0 * np.pi + theta_i_1 - chaser_mean.v
        else:
            dv1 = theta_i_1 - chaser_mean.v

        if theta_i_2 < chaser_mean.v:
            dv2 = 2.0 * np.pi + theta_i_2 - chaser_mean.v
        else:
            dv2 = theta_i_2 - chaser_mean.v

        if dv1 > dv2:
            theta_i = theta_i_2
            theta_f = theta_f_2
        else:
            theta_i = theta_i_1
            theta_f = theta_f_1

        # Initial velocity
        V_PERI_i = np.sqrt(mu_earth / (a * (1.0 - e**2))) * np.array([-np.sin(theta_i), e + np.cos(theta_i), 0.0])
        V_TEM_i = np.linalg.inv(chaser_mean.get_pof()).dot(V_PERI_i)

        # Final velocity
        V_PERI_f = np.sqrt(mu_earth / (a * (1.0 - e**2))) * np.array([-np.sin(theta_f), e + np.cos(theta_f), 0.0])
        V_TEM_f = np.linalg.inv(checkpoint.abs_state.get_pof()).dot(V_PERI_f)

        # Delta-V
        deltaV_C = V_TEM_f - V_TEM_i

        # Create command
        c = Manoeuvre()
        c.dV = deltaV_C
        c.set_abs_state(chaser_mean)
        c.abs_state.v = theta_i
        c.duration = self.travel_time(chaser_mean, chaser_mean.v, theta_i)
        c.description = 'Argument of Perigee correction'
        self.manoeuvre_plan.append(c)

        # Propagate chaser and target
        self._propagator(c.duration, deltaV_C)

    def plane_correction(self, checkpoint):
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
        # Mean orbital elements
        chaser_mean = KepOrbElem()
        chaser_mean.from_osc_elems(self.chaser.abs_state, self.scenario.settings)

        # Extract semi-major axis and eccentricity
        a = chaser_mean.a
        e = chaser_mean.e

        # Changing values
        O_i = chaser_mean.O
        O_f = checkpoint.abs_state.O
        dO = O_f - O_i
        i_i = chaser_mean.i
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
        theta_1 = (2.0 * np.pi - A_Li - chaser_mean.w) % (2.0 * np.pi)
        theta_2 = (theta_1 + np.pi) % (2.0 * np.pi)

        # Choose which of the two position is the closest
        # They consume different dV, the decision has to be taken then depending on if you want to spent a bit more
        # and burn in a specific point, or if you can born anywhere regardless on how much it will cost.
        # Now it's just taking the closest point to do the burn, to decrease the total time of the mission.
        if theta_1 < chaser_mean.v:
            dv1 = 2*np.pi + theta_1 - chaser_mean.v
        else:
            dv1 = theta_1 - chaser_mean.v

        if theta_2 < chaser_mean.v:
            dv2 = 2*np.pi + theta_2 - chaser_mean.v
        else:
            dv2 = theta_2 - chaser_mean.v

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
        V_TEM_i = np.linalg.inv(chaser_mean.get_pof()).dot(V_PERI_i)

        # Rotate vector around c by alpha radiants
        V_TEM_f = R_c.dot(V_TEM_i)

        # Evaluate deltaV
        deltaV_C = V_TEM_f - V_TEM_i

        # Create command
        c = Manoeuvre()
        c.dV = deltaV_C
        c.set_abs_state(chaser_mean)
        c.abs_state.v = theta_i
        c.duration = self.travel_time(chaser_mean, chaser_mean.v, theta_i)
        c.description = 'Inclination and RAAN correction'
        self.manoeuvre_plan.append(c)

        # Propagate chaser and target
        self._propagator(c.duration, deltaV_C)

    def drift_to(self, checkpoint):
        """
            Algorithm that tries to drift to the next checkpoint, staying within a certain error ellipsoid.

        Args:
            chaser (Chaser): Chaser state.
            checkpoint (AbsoluteCP or RelativeCP): Next checkpoint.
            target (Satellite): Target state.

        Return:
            t_est (float64): Drifting time to reach the next checkpoint (in seconds). If not reachable, return None.
        """
        # Calculate the mean altitude difference
        chaser_mean = KepOrbElem()
        chaser_mean.from_osc_elems(self.chaser.abs_state)

        target_mean = KepOrbElem()
        target_mean.from_osc_elems(self.target.abs_state)

        chaser_cart = Cartesian()
        target_cart = Cartesian()

        chaser_cart.from_keporb(chaser_mean)
        target_cart.from_keporb(target_mean)

        r_C = np.linalg.norm(chaser_cart.R)
        r_T = np.linalg.norm(target_cart.R)

        # Assuming we are on a coelliptic orbit, check if the distance allows us to drift or if we are not really
        # close to the target's orbit
        # if abs(checkpoint.rel_state.R[0] - r_C + r_T) >= checkpoint.error_ellipsoid[0] or \
        #                 abs(self.chaser.abs_state.a - self.target.abs_state.a) < 2e-1:
        #     return None

        chaser_old = Chaser()
        chaser_old.set_from_other_satellite(self.chaser)
        target_old = Satellite()
        target_old.set_from_other_satellite(self.target)

        n_c = np.sqrt(mu_earth / chaser_mean.a ** 3)
        n_t = np.sqrt(mu_earth / target_mean.a ** 3)

        # If n_rel is below zero, we are moving slower than target. Otherwise faster.
        n_rel = n_c - n_t

        # Required dv at the end of the manoeuvre, estimation based on the relative position
        dv_req = checkpoint.rel_state.R[1] / r_C

        # Check if a drift to the wanted position is possible, if yes check if it can be done under a certain time,
        # if not try to resync
        actual_dv = (chaser_mean.v + chaser_mean.w) % (2.0*np.pi) - (target_mean.v + target_mean.w) % (2.0*np.pi)

        # Define a function F for the angle calculation
        F = lambda dv_req, dv, n: int((dv - dv_req) / n > 0.0) * np.sign(n)

        t_est = (2.0 * np.pi * F(dv_req, actual_dv, n_rel) + dv_req - actual_dv) / n_rel
        t_est_old = 0.0
        t_old = 0.0
        ellipsoid_flag = False
        tol = 1e-3         # Millisecond tolerance
        dt = 10000.0
        dr_next_old = 0.0
        while abs(t_est - t_old) > tol:
            chaser_prop = self.scenario.prop_chaser.propagate(self.epoch + timedelta(seconds=t_est))
            target_prop = self.scenario.prop_target.propagate(self.epoch + timedelta(seconds=t_est))

            chaser_cart = chaser_prop[0]
            target_cart = target_prop[0]

            chaser_old.abs_state.from_cartesian(chaser_cart)
            target_old.abs_state.from_cartesian(target_cart)
            chaser_old.rel_state.from_cartesian_pair(chaser_cart, target_cart)

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

            chaser_old.set_from_other_satellite(self.chaser)
            target_old.set_from_other_satellite(self.target)

        if ellipsoid_flag:
            # With the estimated time, we are in the error-ellipsoid
            return t_est
        else:
            return None

    def drift_to_new(self, checkpoint):
        """
            Algorithm that tries to drift to the next checkpoint, staying within a certain error ellipsoid.

        Args:
            checkpoint (AbsoluteCP or RelativeCP): Next checkpoint.

        Return:
            t_est (float64): Drifting time to reach the next checkpoint (in seconds). If not reachable, return None.
        """
        # Creating mean orbital elements
        chaser_mean = KepOrbElem()
        target_mean = KepOrbElem()

        # Creating cartesian coordinates
        chaser_cart = Cartesian()
        target_cart = Cartesian()

        # Creating old chaser and target objects to store their temporary value
        chaser_old = Chaser()
        target_old = Satellite()

        # Define a function F for the angle calculation
        F = lambda dv_req, dv, n: int((dv - dv_req) / n > 0.0) * np.sign(n)

        # Correct altitude at every loop until drifting is possible
        while 1:
            # Assign mean values from osculating
            chaser_mean.from_osc_elems(self.chaser.abs_state)
            target_mean.from_osc_elems(self.target.abs_state)

            # Assign cartesian coordinates from mean-orbital (mean orbital radius needed)
            chaser_cart.from_keporb(chaser_mean)
            target_cart.from_keporb(target_mean)

            # Assign information to the new chaser and target objects
            chaser_old.set_from_other_satellite(self.chaser)
            target_old.set_from_other_satellite(self.target)

            # Evaluate relative mean angular velocity. If it's below zero chaser moves slower than target,
            # otherwise faster
            n_c = np.sqrt(mu_earth / chaser_mean.a ** 3)
            n_t = np.sqrt(mu_earth / target_mean.a ** 3)
            n_rel = n_c - n_t

            # Required true anomaly difference at the end of the manoeuvre, estimation assuming circular
            # orbit
            r_C = np.linalg.norm(chaser_cart.R)
            dv_req = checkpoint.rel_state.R[1] / r_C

            # Evaluate the actual true anomaly difference
            actual_dv = (chaser_mean.v + chaser_mean.w) % (2.0 * np.pi) - (target_mean.v + target_mean.w) % (
            2.0 * np.pi)

            # Millisecond tolerance to exit the loop
            tol = 1e-3

            t_est = 10**np.floor(np.log10((2.0 * np.pi * F(dv_req, actual_dv, n_rel) + dv_req - actual_dv) / n_rel))
            t_est_old = 0.0
            t_old = 0.0
            ellipsoid_flag = False
            dt = t_est
            dr_next_old = 0.0
            while abs(t_est - t_old) > tol:
                chaser_prop = self.scenario.prop_chaser.propagate(self.epoch + timedelta(seconds=t_est))
                target_prop = self.scenario.prop_target.propagate(self.epoch + timedelta(seconds=t_est))

                chaser_cart = chaser_prop[0]
                target_cart = target_prop[0]

                self.chaser.abs_state.from_cartesian(chaser_cart)
                self.target.abs_state.from_cartesian(target_cart)
                self.chaser.rel_state.from_cartesian_pair(chaser_cart, target_cart)

                # Correct plane in the middle of the drifting
                tol_i = 1.0 / self.chaser.abs_state.a
                tol_O = 1.0 / self.chaser.abs_state.a

                # At this point, inclination and raan should match the one of the target
                di = target_mean.i - chaser_mean.i
                dO = target_mean.O - chaser_mean.O
                if abs(di) > tol_i or abs(dO) > tol_O:
                    checkpoint_abs = AbsoluteCP()
                    checkpoint_abs.abs_state.i = target_mean.i
                    checkpoint_abs.abs_state.O = target_mean.O
                    self.plane_correction(checkpoint_abs)

                dr_next = self.chaser.rel_state.R[1] - checkpoint.rel_state.R[1]

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

                if abs(checkpoint.rel_state.R[1] - self.chaser.rel_state.R[1]) <= checkpoint.error_ellipsoid[1]:
                    # Almost in line with the checkpoint
                    if abs(checkpoint.rel_state.R[0] - self.chaser.rel_state.R[0]) <= checkpoint.error_ellipsoid[0]:
                        # Inside the tolerance, the point may be reached by drifting
                        ellipsoid_flag = True
                    else:
                        # Outside tolerance, point may not be reached!
                        break

                self.chaser.set_from_other_satellite(chaser_old)
                self.target.set_from_other_satellite(target_old)

            if ellipsoid_flag:
                # It is possible to drift in t_est
                return t_est
            else:
                # Drift is not possible, drop a warning and correct altitude!
                print "\n[WARNING]: Drifting to checkpoint nr. " + str(checkpoint.id) + " not possible!"
                print "           Correcting altitude automatically...\n"

                # Create new checkpoint
                checkpoint_new_abs = AbsoluteCP()
                checkpoint_new_abs.set_abs_state(chaser_mean)
                checkpoint_new_abs.abs_state.a = target_mean.a + checkpoint.rel_state.R[0]
                checkpoint_new_abs.abs_state.e = target_mean.a * target_mean.e / checkpoint_new_abs.abs_state.a

                self.adjust_eccentricity_semimajoraxis(checkpoint_new_abs)

    def drift_to_new2(self, checkpoint):
        """
            Algorithm that tries to drift to the next checkpoint, staying within a certain error ellipsoid.

        Args:
            checkpoint (AbsoluteCP or RelativeCP): Next checkpoint.

        Return:
            t_est (float64): Drifting time to reach the next checkpoint (in seconds). If not reachable, return None.
        """
        # Creating mean orbital elements
        chaser_mean = KepOrbElem()
        target_mean = KepOrbElem()

        # Creating cartesian coordinates
        chaser_cart = Cartesian()
        target_cart = Cartesian()

        # Creating old chaser and target objects to store their temporary value
        chaser_old = Chaser()
        target_old = Satellite()

        # Define a function F for the angle calculation
        F = lambda dv_req, dv, n: int((dv - dv_req) / n > 0.0) * np.sign(n)

        # Correct altitude at every loop until drifting is possible
        while 1:
            # Assign mean values from osculating
            chaser_mean.from_osc_elems(self.chaser.abs_state, self.scenario.settings)
            target_mean.from_osc_elems(self.target.abs_state, self.scenario.settings)

            # Assign cartesian coordinates from mean-orbital (mean orbital radius needed)
            chaser_cart.from_keporb(chaser_mean)

            # Assign information to the new chaser and target objects
            chaser_old.set_from_other_satellite(self.chaser)
            target_old.set_from_other_satellite(self.target)

            # Store initial epoch
            epoch_old = self.epoch

            # Evaluate relative mean angular velocity. If it's below zero chaser moves slower than target,
            # otherwise faster
            n_c = np.sqrt(mu_earth / chaser_mean.a ** 3)
            n_t = np.sqrt(mu_earth / target_mean.a ** 3)
            n_rel = n_c - n_t

            # Required true anomaly difference at the end of the manoeuvre, estimation assuming circular
            # orbit
            r_C = np.linalg.norm(chaser_cart.R)
            dv_req = checkpoint.rel_state.R[1] / r_C

            # Evaluate the actual true anomaly difference
            actual_dv = (chaser_mean.v + chaser_mean.w) % (2.0 * np.pi) - (target_mean.v + target_mean.w) % (
            2.0 * np.pi)

            # Millisecond tolerance to exit the loop
            tol = 1e-3

            chaser_tmp = Chaser()
            target_tmp = Satellite()

            manoeuvre_plan_old = self.manoeuvre_plan

            t_est = (2.0 * np.pi * F(dv_req, actual_dv, n_rel) + dv_req - actual_dv) / n_rel
            ellipsoid_flag = False
            dt = 10**np.floor(np.log10(t_est)) if t_est / (10**np.floor(np.log10(t_est))) >= 2.0 else 10**np.floor(np.log10(t_est) - 1.0)
            dr_next_old = 0.0
            dr_next = 0.0
            while dt > tol:
                # Store (i-1) chaser and target state
                chaser_tmp.set_from_other_satellite(self.chaser)
                target_tmp.set_from_other_satellite(self.target)
                epoch_tmp = self.epoch
                dr_next_tmp = dr_next
                manoeuvre_plan_tmp = self.manoeuvre_plan

                # Update epoch
                self.epoch = self.epoch + timedelta(seconds=dt)

                # Propagate
                chaser_prop = self.scenario.prop_chaser.propagate(self.epoch)
                target_prop = self.scenario.prop_target.propagate(self.epoch)

                self.chaser.abs_state.from_cartesian(chaser_prop[0])
                self.target.abs_state.from_cartesian(target_prop[0])
                self.chaser.rel_state.from_cartesian_pair(chaser_prop[0], target_prop[0])

                # Re-initialize propagators
                self.scenario.initialize_propagators(self.chaser.abs_state, self.target.abs_state, self.epoch)

                dr_next = self.chaser.rel_state.R[1] - checkpoint.rel_state.R[1]

                if dr_next <= 0.0 and dr_next_old <= 0.0:
                    # Correct plane in the middle of the drifting
                    tol_i = 0.5 / self.chaser.abs_state.a
                    tol_O = 0.5 / self.chaser.abs_state.a

                    chaser_mean.from_osc_elems(self.chaser.abs_state, self.scenario.settings)
                    target_mean.from_osc_elems(self.target.abs_state, self.scenario.settings)

                    # At this point, inclination and raan should match the one of the target
                    di = target_mean.i - chaser_mean.i
                    dO = target_mean.O - chaser_mean.O
                    if abs(di) > tol_i or abs(dO) > tol_O:
                        checkpoint_abs = AbsoluteCP()
                        checkpoint_abs.abs_state.i = target_mean.i
                        checkpoint_abs.abs_state.O = target_mean.O
                        self.plane_correction(checkpoint_abs)

                        dr_next = self.chaser.rel_state.R[1] - checkpoint.rel_state.R[1]

                        if dr_next >= 0.0:
                            # Overshoot due to plane adjustment => reduce dt and depropagate
                            dt /= 10.0
                            self.chaser.set_from_other_satellite(chaser_tmp)
                            self.target.set_from_other_satellite(target_tmp)
                            self.epoch = epoch_tmp
                            self.scenario.initialize_propagators(self.chaser.abs_state, self.target.abs_state, self.epoch)
                            self.manoeuvre_plan = manoeuvre_plan_tmp
                            # dr_next_old should be the same as the one at the beginning
                            dr_next = dr_next_tmp
                        else:
                            # Target point not overshooted, everything looks good as it is
                            c = RelativeMan()
                            c.dV = np.array([0.0, 0.0, 0.0])
                            c.set_abs_state(chaser_tmp.abs_state)
                            c.set_rel_state(chaser_tmp.rel_state)
                            c.duration = dt
                            c.description = 'Drift for ' + str(t_est) + ' seconds'
                            self.manoeuvre_plan.append(c)
                            dr_next_old = dr_next

                    else:
                        # No plane adjustment needed, add another dt and move forward
                        c = RelativeMan()
                        c.dV = np.array([0.0, 0.0, 0.0])
                        c.set_abs_state(chaser_tmp.abs_state)
                        c.set_rel_state(chaser_tmp.rel_state)
                        c.duration = dt
                        c.description = 'Drift for ' + str(dt) + ' seconds'
                        self.manoeuvre_plan.append(c)
                        dr_next_old = dr_next

                elif dr_next >= 0.0 and dr_next_old >= 0.0:
                    # Only useful for the case when chaser is on a higher orbit
                    pass

                elif (dr_next <= 0.0 and dr_next_old >= 0.0) or (dr_next >= 0.0 and dr_next_old <= 0.0):
                    dt /= 10.0
                    self.chaser.set_from_other_satellite(chaser_tmp)
                    self.target.set_from_other_satellite(target_tmp)
                    self.epoch = epoch_tmp
                    self.scenario.initialize_propagators(self.chaser.abs_state, self.target.abs_state, self.epoch)
                    # dr_next_old should be the same as the one at the beginning
                    dr_next = dr_next_tmp

                if abs(checkpoint.rel_state.R[1] - self.chaser.rel_state.R[1]) <= checkpoint.error_ellipsoid[1]:
                    # Almost in line with the checkpoint
                    if abs(checkpoint.rel_state.R[0] - self.chaser.rel_state.R[0]) <= checkpoint.error_ellipsoid[0]:
                        # Inside the tolerance, the point may be reached by drifting
                        ellipsoid_flag = True
                    elif abs(checkpoint.rel_state.R[1] - self.chaser.rel_state.R[1]) <= 0.05 and \
                        abs(checkpoint.rel_state.R[0] - self.chaser.rel_state.R[0]) > checkpoint.error_ellipsoid[0]:
                        # Outside tolerance, point may not be reached!
                        break

            if ellipsoid_flag:
                # It is possible to drift
                return
            else:
                # Drift is not possible, drop a warning and correct altitude!
                print "\n[WARNING]: Drifting to checkpoint nr. " + str(checkpoint.id) + " not possible!"
                print "           Correcting altitude automatically...\n"

                # Depropagate to initial conditions before drifting
                self.chaser.set_from_other_satellite(chaser_old)
                self.target.set_from_other_satellite(target_old)

                chaser_mean.from_osc_elems(self.chaser.abs_state, self.scenario.settings)
                target_mean.from_osc_elems(self.target.abs_state, self.scenario.settings)

                self.epoch = epoch_old

                self.scenario.initialize_propagators(self.chaser.abs_state, self.target.abs_state, self.epoch)

                self.manoeuvre_plan = manoeuvre_plan_old

                # Create new checkpoint
                checkpoint_new_abs = AbsoluteCP()
                checkpoint_new_abs.set_abs_state(chaser_mean)
                checkpoint_new_abs.abs_state.a = target_mean.a + checkpoint.rel_state.R[0]
                checkpoint_new_abs.abs_state.e = target_mean.a * target_mean.e / checkpoint_new_abs.abs_state.a

                self.adjust_eccentricity_semimajoraxis(checkpoint_new_abs)      #TODO: Maybe add also another plane correction there...

    def multi_lambert(self, checkpoint, approach_ellipsoid, safety_flag):
        """
            Solve the Multi-Lambert Problem.

        Args:
            chaser (Chaser): Chaser state.
            checkpoint (AbsoluteCP or RelativeCP): Next checkpoint (theoretically used only for RelativeCP).
            target (Satellite): Target state.
            approach_ellipsoid: Approach ellipsoid drawn around the target to be avoided during manoeuvering
        """

        # Check if trajectory is retrograde
        retrograde = False
        if self.chaser.abs_state.i > np.pi / 2.0:
            retrograde = True

        # Calculate the cartesian coordinates of target and chaser
        chaser_cart = Cartesian()
        target_cart = Cartesian()

        chaser_cart.from_keporb(self.chaser.abs_state)
        target_cart.from_keporb(self.target.abs_state)

        # Absolute position of chaser at t = t0
        R_C_i = chaser_cart.R
        V_C_i = chaser_cart.V

        # Absolute position of the target at t = t0
        R_T_i = target_cart.R
        V_T_i = target_cart.V

        # Create temporary target that will keep the initial conditions
        target_ic = Cartesian()
        chaser_ic = Cartesian()
        target_ic.R = R_T_i
        target_ic.V = V_T_i

        # Initialize best dV and dt
        best_dV = 1e12
        best_dt = 0.0

        # Minimum deltaV deliverable -> 5 mm/s
        dV_min = 5e-6

        # Check all the possible transfers time from tmin to tmax (seconds)
        t_min = int(checkpoint.t_min)
        t_max = int(checkpoint.t_max)
        for dt in xrange(t_min, t_max):
            r_T, v_T = pk.propagate_lagrangian(R_T_i, V_T_i, dt, mu_earth)

            target_prop = self.scenario.prop_target.propagate(self.epoch + timedelta(seconds=dt))

            target_cart.R = np.array(r_T)
            target_cart.V = np.array(v_T)

            # Transformation matrix from TEME to LVLH at time t1
            B_LVLH_TEME_f = target_cart.get_lof()

            # Evaluate final wanted absolute position of the chaser
            R_C_f = np.array(target_cart.R) + np.linalg.inv(B_LVLH_TEME_f).dot(checkpoint.rel_state.R)
            O_T_f = np.cross(target_cart.R, target_cart.V) / np.linalg.norm(target_cart.R)**2
            V_C_f = np.array(target_cart.V) + np.linalg.inv(B_LVLH_TEME_f).dot(checkpoint.rel_state.V) + \
                    np.cross(O_T_f, np.linalg.inv(B_LVLH_TEME_f).dot(checkpoint.rel_state.R))

            # Solve lambert in dt starting from the chaser position at t0 going to t1
            sol = pk.lambert_problem(R_C_i, R_C_f, dt, mu_earth, retrograde, 10)

            # Check for the best solution for this dt
            for i in xrange(0, len(sol.get_v1())):
                dV_1 = np.array(sol.get_v1()[i]) - V_C_i
                dV_2 = V_C_f - np.array(sol.get_v2()[i])
                dV_tot = np.linalg.norm(dV_1) + np.linalg.norm(dV_2)

                # Check if the deltaV is above the minimum deliverable by thrusters
                if np.linalg.norm(dV_1) > dV_min and np.linalg.norm(dV_2) > dV_min:
                    # Check if the new deltaV is less than previous
                    if dV_tot < best_dV and safety_flag:
                        # Approach ellipsoid can be entered
                        best_dV = dV_tot
                        best_dV_1 = dV_1
                        best_dV_2 = dV_2
                        best_dt = dt
                    elif dV_tot < best_dV and not safety_flag:
                        # Check if the trajectory is safe
                        chaser_ic.R = R_C_i
                        chaser_ic.V = np.array(sol.get_v1()[i])
                        if self.is_trajectory_safe(chaser_ic, target_ic, dt, approach_ellipsoid):
                            best_dV = dV_tot
                            best_dV_1 = dV_1
                            best_dV_2 = dV_2
                            best_dt = dt

        c1 = RelativeMan()
        c1.dV = best_dV_1
        c1.set_abs_state(self.chaser.abs_state)
        c1.set_rel_state(self.chaser.rel_state)
        c1.duration = 0
        c1.description = 'Multi-Lambert solution'
        self.manoeuvre_plan.append(c1)

        self._propagator(1e-3, best_dV_1)

        chaser_cart.from_keporb(self.chaser.abs_state)

        c2 = RelativeMan()
        c2.dV = best_dV_2
        c2.set_abs_state(self.chaser.abs_state)
        c2.set_rel_state(self.chaser.rel_state)
        c2.duration = best_dt
        c2.description = 'Multi-Lambert solution'
        self.manoeuvre_plan.append(c2)

        self._propagator(best_dt, best_dV_2)

    def clohessy_wiltshire(self, checkpoint):
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

    def absolute_solver(self, checkpoint):
        """
            Absolute solver. Calculate the manoeuvre needed to go from an absolute position to another.

        Args:
             chaser (Chaser):
             checkpoint (AbsoluteCP):
             target (Satellite):
        """
        # Define mean orbital elements
        chaser_mean = KepOrbElem()
        chaser_mean.from_osc_elems(self.chaser.abs_state, self.scenario.settings)

        # Define tolerances, if we get deviations greater than ~1 km then correct
        tol_i = 1.0 / self.chaser.abs_state.a
        tol_O = 1.0 / self.chaser.abs_state.a
        tol_w = 1.0 / self.chaser.abs_state.a
        tol_a = 0.2
        tol_e = 1.0 / self.chaser.abs_state.a

        # Evaluate differences
        di = checkpoint.abs_state.i - chaser_mean.i
        dO = checkpoint.abs_state.O - chaser_mean.O
        dw = checkpoint.abs_state.w - chaser_mean.w
        da = checkpoint.abs_state.a - chaser_mean.a
        de = checkpoint.abs_state.e - chaser_mean.e

        # Inclination and RAAN
        if abs(di) > tol_i or abs(dO) > tol_O:
            self.plane_correction(checkpoint)

        # Argument of Perigee
        if abs(dw) > tol_w:
            self.adjust_perigee(checkpoint)

        # Eccentricity and Semi-Major Axis
        if abs(da) > tol_a or abs(de) > tol_e:
            self.adjust_eccentricity_semimajoraxis(checkpoint)

    def relative_solver(self, checkpoint, approach_ellipsoid):

        # Mean orbital elements
        chaser_mean = KepOrbElem()
        chaser_mean.from_osc_elems(self.chaser.abs_state, self.scenario.settings)

        target_mean = KepOrbElem()
        target_mean.from_osc_elems(self.target.abs_state, self.scenario.settings)

        # Check if plane needs to be corrected again
        # TODO: Put as tolerance a number slightly bigger than the deviation of the estimation
        tol_i = 1.0 / self.chaser.abs_state.a
        tol_O = 1.0 / self.chaser.abs_state.a

        # At this point, inclination and raan should match the one of the target
        di = target_mean.i - chaser_mean.i
        dO = target_mean.O - chaser_mean.O
        if abs(di) > tol_i or abs(dO) > tol_O:
            checkpoint_abs = AbsoluteCP()
            checkpoint_abs.abs_state.i = target_mean.i
            checkpoint_abs.abs_state.O = target_mean.O
            self.plane_correction(checkpoint_abs)

        t_limit = 604800

        if checkpoint.manoeuvre_type == 'standard':
            self.multi_lambert(checkpoint, approach_ellipsoid, False)

        elif checkpoint.manoeuvre_type == 'radial':
            # Manoeuvre type is radial -> deltaT is calculated from CW-equations -> solved with multi-lambert
            a = self.target.abs_state.a
            dt = np.pi / np.sqrt(mu_earth / a ** 3.0)

            checkpoint.t_min = dt
            checkpoint.t_max = dt + 1.0

            self.multi_lambert(checkpoint, approach_ellipsoid, True)

        elif checkpoint.manoeuvre_type == 'drift':
            self.drift_to_new2(checkpoint)

    def _propagator(self, dt, dv=np.array([0, 0, 0])):
        """
            Propagate chaser and target to t* = now + dt.

        Args:
            chaser (Chaser): Chaser state.
            target (Satellite): Target state.
            dt (float64): Propagated time in seconds.
            dv (np.array): Amount of delta-v to be applied at the beginning of the propagation, in km/s in TEME
                reference frame.
        """

        # Update time
        self.epoch += timedelta(seconds=dt)

        # Propagate
        chaser_prop = self.scenario.prop_chaser.propagate(self.epoch)
        target_prop = self.scenario.prop_target.propagate(self.epoch)

        chaser_cart = chaser_prop[0]
        target_cart = target_prop[0]

        self.chaser.abs_state.from_cartesian(chaser_cart)

        # Add deltaV
        chaser_cart.V += dv

        # Update chaser and target
        self.chaser.abs_state.from_cartesian(chaser_cart)
        self.target.abs_state.from_cartesian(target_cart)
        self.chaser.rel_state.from_cartesian_pair(chaser_cart, target_cart)

        # Initialize propagator
        self.scenario.initialize_propagators(self.chaser.abs_state, self.target.abs_state, self.epoch)

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

        a = chaser.a
        e = chaser.e

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

    def is_trajectory_safe(self, chaser_cart, target_cart, t, approach_ellipsoid):
        """
            Check if a trajectory is safe.

        Args:
            chaser_cart:
            target_cart:
            dt:
            approach_ellipsoid:
        """

        t_SF = 7200.0
        dt = 10.0

        T = np.arange(0.0, t + t_SF, dt)

        target_new = Cartesian()

        r = chaser_cart.R
        v = chaser_cart.V

        r_t = target_cart.R
        v_t = target_cart.V
        target_new.R = r_t
        target_new.V = v_t

        for t in T:
            r, v = pk.propagate_lagrangian(r, v, dt, mu_earth)
            r_t, v_t = pk.propagate_lagrangian(r_t, v_t, dt, mu_earth)

            r = np.array(r)
            v = np.array(v)

            r_t = np.array(r_t)
            v_t = np.array(v_t)

            target_new.R = r_t
            target_new.V = v_t

            # Calculate relative position in LVLH frame
            dr_TEME = r - r_t
            dr_LVLH = target_new.get_lof().dot(dr_TEME)

            is_inside = (dr_LVLH[0]**2 / approach_ellipsoid[0]**2 + dr_LVLH[1]**2 / approach_ellipsoid[1]**2 + \
                        dr_LVLH[2]**2 / approach_ellipsoid[2]**2) <= 1.0

            if is_inside:
                return False

        return True

    def print_state(self, kep):
        print "      a :     " + str(kep.a)
        print "      e :     " + str(kep.e)
        print "      i :     " + str(kep.i)
        print "      O :     " + str(kep.O)
        print "      w :     " + str(kep.w)
        print "      v :     " + str(kep.v)

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

    def linearized_including_J2(self, checkpoint):

        a_0 = self.target.abs_state.a
        e_0 = self.target.abs_state.e
        M_0 = self.target.abs_state.m
        v_0 = self.target.abs_state.v

        target_mean = KepOrbElem()
        target_mean.from_osc_elems(self.target.abs_state)

        a_mean = target_mean.a
        i_mean = target_mean.i
        e_mean = target_mean.e

        # p. 1650
        # e_0: target eccentricity (or maybe chaser...)
        eta = np.sqrt(1.0 - e_0 ** 2)  # TODO CHECK
        eta_0 = np.sqrt(1.0 - e_0 ** 2)
        eta_mean = np.sqrt(1.0 - e_mean ** 2)

        p_mean = a_mean * eta_mean ** 2
        p = a_0 * eta ** 2

        # Nr. of orbits performed by spacecraft
        N_orb = ...

        # Eccentric anomaly
        E = lambda v: 2.0 * np.arctan(np.sqrt((1.0 - e_0)/(1.0 + e_0)) * np.tan(v / 2.0))

        # Mean anomaly
        M = lambda v: E(v) - e_0 * E(v)

        # Mean motion
        n_mean = np.sqrt(mu_earth / a_mean ** 3)

        # Constant C
        C = J_2 * n_mean * R_earth ** 2 / (4.0 * p)

        # p. 1651: Mean anomaly drift rate
        M_mean_dot = n_mean + 0.75 * J_2 * n_mean * (R_earth / p_mean) ** 2 * eta_mean * (3.0 * np.cos(i_mean) ** 2 - 1.0)

        # p. 1651: dMda
        dMda = -3.0 * n_mean / (2.0 * a_mean) - eta_mean / (4.0 * a_mean) * C * (63.0 * np.cos(2.0 * i_mean) - 21.0)

        # p. 1656: depsd-
        depsde = 

        # p. 1652: tau
        tau = lambda v: (2.0 * np.pi * N_orb + M(v) - M_0) / M_mean_dot

        # p. 1653: r_dot
        r_dot = lambda v: a_0 * e_0 * np.sin(v) / eta * M_mean_dot

        # p. 1653: v_dot
        # e_0: initial reference orbit eccentricity (chaser?)
        v_dot = lambda v: (1.0 + e_0 * np.cos(v)) ** 2 / eta ** 3 * M_mean_dot

        # p. 1656: k_x_dot
        k_x_dot = lambda v: a_0 * e_0 * v_dot(v) * np.cos(v) / eta_0

        # p.1656: phi_1
        phi_11 = lambda v: r_dot / a_0 + (k_x_dot * tau + a_0 * e_0 * np.sin(v) / eta_0) * dMda

        phi_12 = lambda v: (k_x_dot * tau + a_0 * e_0 * np.sin(v) / eta_0) * (dMda * depsde + dMda * depsdv * np.sin(v_0) / eta_0 ** 2 * (2.0 + e_0 * np.cos(e_0)))
