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
from datetime import timedelta, datetime

from org.orekit.propagation import SpacecraftState
from org.orekit.frames import FramesFactory
from org.orekit.orbits import CartesianOrbit
from org.orekit.utils import PVCoordinates
from org.orekit.utils import Constants as Cst
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.time import AbsoluteDate, TimeScalesFactory


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

        self.epoch = datetime.utcnow()

    def set_solver(self, scenario):
        self.scenario = scenario
        self.chaser.set_from_other_satellite(scenario.chaser_ic)
        self.target.set_from_other_satellite(scenario.target_ic)
        self.epoch = scenario.date

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
        chaser_mean.from_osc_elems(self.chaser.abs_state, self.scenario.prop_type)
        self.print_state(chaser_mean)
        print "\n >> LVLH:"
        print "     R: " + str(self.chaser.rel_state.R)
        print "     V: " + str(self.chaser.rel_state.V)
        print "\n--------------------Target initial state-------------------"
        print " >> Osc Elements:"
        self.print_state(self.target.abs_state)
        print "\n >> Mean Elements:"
        target_mean = KepOrbElem()
        target_mean.from_osc_elems(self.target.abs_state, self.scenario.prop_type)
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
            chaser_mean.from_osc_elems(self.chaser.abs_state, self.scenario.prop_type)
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
        chaser_mean.from_osc_elems(self.chaser.abs_state, self.scenario.prop_type)

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

        c1.initial_rel_state = self.chaser.rel_state.R
        # Propagate chaser and target
        self._propagator(c1.duration, deltaV_C_1)
        c1.final_rel_state = self.chaser.rel_state.R
        self.manoeuvre_plan.append(c1)

        chaser_mean.from_osc_elems(self.chaser.abs_state, self.scenario.prop_type)

        c2 = Manoeuvre()
        c2.dV = deltaV_C_2
        c2.set_abs_state(chaser_mean)
        c2.abs_state.v = theta_2
        c2.duration = np.pi * np.sqrt(chaser_mean.a**3 / mu_earth)
        c2.description = 'Apogee/Perigee raise'

        c2.initial_rel_state = self.chaser.rel_state.R
        # Propagate chaser and target
        self._propagator(c2.duration, deltaV_C_2)
        c2.final_rel_state = self.chaser.rel_state.R
        self.manoeuvre_plan.append(c2)

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
        chaser_mean.from_osc_elems(self.chaser.abs_state, self.scenario.prop_type)

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
        c.initial_rel_state = self.chaser.rel_state.R
        # Propagate chaser and target
        self._propagator(c.duration, deltaV_C)
        c.final_rel_state = self.chaser.rel_state.R
        self.manoeuvre_plan.append(c)

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
        chaser_mean.from_osc_elems(self.chaser.abs_state, self.scenario.prop_type)

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

        c.initial_rel_state = self.chaser.rel_state.R
        # Propagate chaser and target
        self._propagator(c.duration, deltaV_C)
        c.final_rel_state = self.chaser.rel_state.R
        self.manoeuvre_plan.append(c)

    def drift_to(self, checkpoint):
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
            chaser_mean.from_osc_elems(self.chaser.abs_state, self.scenario.prop_type)
            target_mean.from_osc_elems(self.target.abs_state, self.scenario.prop_type)

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
                self._change_propagator_ic(self.scenario.prop_chaser, chaser_prop[0], self.epoch, self.chaser.mass)
                self._change_propagator_ic(self.scenario.prop_target, target_prop[0], self.epoch, self.target.mass)

                dr_next = self.chaser.rel_state.R[1] - checkpoint.rel_state.R[1]

                if dr_next <= 0.0 and dr_next_old <= 0.0:
                    # Correct plane in the middle of the drifting
                    tol_i = 0.5 / self.chaser.abs_state.a
                    tol_O = 0.5 / self.chaser.abs_state.a

                    chaser_mean.from_osc_elems(self.chaser.abs_state, self.scenario.prop_type)
                    target_mean.from_osc_elems(self.target.abs_state, self.scenario.prop_type)

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

                            chaser_cartesian_tmp = Cartesian()
                            chaser_cartesian_tmp.from_keporb(self.chaser.abs_state)

                            target_cartesian_tmp = Cartesian()
                            target_cartesian_tmp.from_keporb(self.target.abs_state)

                            self._change_propagator_ic(self.scenario.prop_chaser, chaser_cartesian_tmp, self.epoch,
                                                       self.chaser.mass)
                            self._change_propagator_ic(self.scenario.prop_target, target_cartesian_tmp, self.epoch,
                                                       self.target.mass)

                            # self.scenario.initialize_propagators(self.chaser.abs_state, self.target.abs_state, self.epoch)


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
                            c.description = 'Drift for ' + str(dt) + ' seconds'

                            c.initial_rel_state = chaser_tmp.rel_state.R
                            # Propagate chaser and target
                            c.final_rel_state = self.chaser.rel_state.R
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

                        c.initial_rel_state = chaser_tmp.rel_state.R
                        # Propagate chaser and target
                        c.final_rel_state = self.chaser.rel_state.R
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

                    chaser_cart_tmp = Cartesian()
                    target_cart_tmp = Cartesian()
                    chaser_cart_tmp.from_keporb(chaser_tmp.abs_state)
                    target_cart_tmp.from_keporb(target_tmp.abs_state)

                    self._change_propagator_ic(self.scenario.prop_chaser, chaser_cart_tmp, self.epoch, chaser_tmp.mass)
                    self._change_propagator_ic(self.scenario.prop_target, target_cart_tmp, self.epoch, target_tmp.mass)
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

                chaser_mean.from_osc_elems(self.chaser.abs_state, self.scenario.prop_type)
                target_mean.from_osc_elems(self.target.abs_state, self.scenario.prop_type)

                self.epoch = epoch_old

                chaser_cart_old = Cartesian()
                target_cart_old = Cartesian()
                chaser_cart_old.from_keporb(chaser_old.abs_state)
                target_cart_old.from_keporb(target_old.abs_state)

                self._change_propagator_ic(self.scenario.prop_chaser, chaser_cart_old, self.epoch, chaser_old.mass)
                self._change_propagator_ic(self.scenario.prop_target, target_cart_old, self.epoch, target_old.mass)

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
                        if self.is_trajectory_safe(dt, approach_ellipsoid):
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
        chaser_mean.from_osc_elems(self.chaser.abs_state, self.scenario.prop_type)

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
        chaser_mean.from_osc_elems(self.chaser.abs_state, self.scenario.prop_type)

        target_mean = KepOrbElem()
        target_mean.from_osc_elems(self.target.abs_state, self.scenario.prop_type)

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
            # self.multi_lambert(checkpoint, approach_ellipsoid, False)
            self.linearized_including_J2(checkpoint, approach_ellipsoid)

        elif checkpoint.manoeuvre_type == 'radial':
            # Manoeuvre type is radial -> deltaT is calculated from CW-equations -> solved with multi-lambert
            a = self.target.abs_state.a
            dt = np.pi / np.sqrt(mu_earth / a ** 3.0)

            checkpoint.t_min = dt
            checkpoint.t_max = dt + 1.0

            # self.multi_lambert(checkpoint, approach_ellipsoid, True)
            self.linearized_including_J2(checkpoint, approach_ellipsoid)

        elif checkpoint.manoeuvre_type == 'drift':
            self.drift_to(checkpoint)

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

        # Update propagator initial conditions
        self._change_propagator_ic(self.scenario.prop_chaser, chaser_cart, self.epoch, self.chaser.mass)
        self._change_propagator_ic(self.scenario.prop_target, target_cart, self.epoch, self.target.mass)

    def travel_time(self, state, theta0, theta1):
        """
            Evaluate the travel time of a satellite from a starting true anomaly theta0 to an end anomaly theta1.

        Reference:
            Exercise of Nicollier's Lecture.
            David A. Vallado, Fundamentals of Astrodynamics and Applications, Second Edition, Algorithm 11 (p. 133)

        Args:
            state (KepOrbElem): Satellite state in keplerian orbital elements.
            theta0 (rad): Starting true anomaly.
            theta1 (rad): Ending true anomaly.

        Return:
            Travel time (seconds)
        """

        a = state.a
        e = state.e

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

    def is_trajectory_safe(self, t_man, keep_out_zone):
        """
            Check if a trajectory is safe, i.e if it never violates a given keep-out zone.

        Args:
            t_man: Duration of the manoeuvre.
            keep_out_zone: Zone around the target that the trajectory should not cross. Given as an array representing
                            the 3 axis of an imaginary ellipsoid drawn around the target (with respect to LVLH frame).
                            ex: np.array([0.5,0.5,0.5]) is a sphere of radius 500 meter.
        """

        # Safety time, propagation continues up to t + t_SF to see if the trajectory is safe also afterwards
        t_SF = 7200.0

        # Time step of propagation
        dt = 10.0

        # Time vector
        T = np.arange(0.0, t_man + t_SF, dt)

        # Extract propagators
        prop_chaser = self.scenario.prop_chaser
        prop_target = self.scenario.prop_target

        for t in T:
            chaser = prop_chaser.propagate(self.epoch + timedelta(seconds=t))
            target = prop_target.propagate(self.epoch + timedelta(seconds=t))

            # Calculate relative position in LVLH frame
            dr_TEME = chaser[0].R - target[0].R
            dr_LVLH = target[0].get_lof().dot(dr_TEME)

            if (dr_LVLH[0]**2 / keep_out_zone[0]**2 + dr_LVLH[1]**2 / keep_out_zone[1]**2 +
                dr_LVLH[2]**2 / keep_out_zone[2]**2) <= 1.0:
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
            print '--> Initial rel state: ' + str(command.initial_rel_state)
            print '--> Final rel state:   ' + str(command.final_rel_state)
            tot_dv += np.linalg.norm(command.dV)
            tot_dt += command.duration

        return tot_dv, tot_dt

    def linearized_including_J2(self, checkpoint, approach_ellipsoid):
        """
            Using the linearized solution including J2 and elliptical orbits to estimate the deltaV needed.

        Args:
            checkpoint (Checkpoint):
        """

        # Initial reference osculatin orbit
        a_0 = self.target.abs_state.a
        e_0 = self.target.abs_state.e
        i_0 = self.target.abs_state.i
        O_0 = self.target.abs_state.O
        w_0 = self.target.abs_state.w
        M_0 = self.target.abs_state.m
        v_0 = self.target.abs_state.v

        # Initial relative orbital elements
        de0_initial = np.array([
            self.chaser.abs_state.a - a_0,
            self.chaser.abs_state.e - e_0,
            self.chaser.abs_state.i - i_0,
            self.chaser.abs_state.O - O_0,
            self.chaser.abs_state.w - w_0,
            self.chaser.abs_state.m - M_0
        ])

        # Chaser cartesian initial state
        chaser_cart = Cartesian()
        chaser_cart.from_keporb(self.chaser.abs_state)

        eta_0 = np.sqrt(1.0 - e_0 ** 2)
        p_0 = a_0 * (1.0 - e_0 ** 2)
        r_0 = p_0 / (1.0 + e_0 * np.cos(v_0))

        # Initial reference mean orbit
        target_mean = KepOrbElem()
        target_mean.from_osc_elems(self.target.abs_state, 'real-world')

        a_mean = target_mean.a
        i_mean = target_mean.i
        e_mean = target_mean.e

        eta_mean = np.sqrt(1.0 - e_mean ** 2)
        p_mean = a_mean * (1.0 - e_mean ** 2)
        n_mean = np.sqrt(mu_earth / a_mean ** 3)
        T_mean = 2.0 * np.pi / n_mean

        # Mean orbital element drift
        w_mean_dot = 0.75 * J_2 * n_mean * (R_earth / p_mean) ** 2 * (5.0 * np.cos(i_mean) ** 2 - 1.0)
        M_mean_dot = n_mean + 0.75 * J_2 * n_mean * (R_earth / p_mean) ** 2 * eta_mean * \
                     (3.0 * np.cos(i_mean) ** 2 - 1.0)

        # Epsilon_a partial derivatives
        gamma_2 = -0.5 * J_2 * (R_earth / a_0) ** 2

        depsda = 1.0 - gamma_2 * ((3.0 * np.cos(i_0) ** 2 - 1.0) * ((a_0 / r_0) ** 3 - 1.0 / eta_0 ** 3) +
                                  3.0 * (1.0 - np.cos(i_0) ** 2) * (a_0 / r_0) ** 3 * np.cos(2.0 * w_0 + 2.0 * v_0))
        depsde = a_0 * gamma_2 * ((2.0 - 3.0 * np.sin(i_0) ** 2) *
                                  (3.0 * np.cos(v_0) * (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 6 + 6.0 * e_0 * (
                                              1.0 + e_0 * np.cos(v_0)) ** 3 / eta_0 ** 8 - 3.0 * e_0 / eta_0 ** 5) +
                                  9.0 * np.sin(i_0) ** 2 * np.cos(2.0 * w_0 + 2.0 * v_0) * np.cos(v_0) * (
                                              1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 6 +
                                  18.0 * np.sin(i_0) ** 2 * e_0 * np.cos(2.0 * w_0 + 2.0 * v_0) * (
                                              1.0 + e_0 * np.cos(v_0)) ** 3 / eta_0 ** 8)
        depsdi = -3.0 * a_0 * gamma_2 * np.sin(2.0 * i_0) * (
                    (a_0 / r_0) ** 3 * (1.0 - np.cos(2.0 * w_0 + 2.0 * v_0)) - 1.0 / eta_0 ** 3)
        depsdw = -6.0 * a_0 * gamma_2 * (1.0 - np.cos(i_0) ** 2) * (a_0 / r_0) ** 3 * np.sin(2.0 * w_0 + 2.0 * v_0)
        depsdv = a_0 * gamma_2 * (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 6 * \
                 ((-9.0 * np.cos(i_0) ** 2 + 3.0) * e_0 * np.sin(v_0) -
                  (9.0 - 9.0 * np.cos(i_0) ** 2) * np.cos(2.0 * w_0 + 2.0 * v_0) * e_0 * np.sin(v_0) -
                  (6.0 - 6.0 * np.cos(i_0) ** 2) * (1.0 + e_0 * np.cos(v_0)) * np.sin(2.0 * w_0 + 2.0 * v_0))

        # Mean elements partial derivatives
        C = J_2 * n_mean * R_earth ** 2 / (4.0 * p_mean ** 2)
        dOda = 21.0 / a_mean * C * np.cos(i_mean)
        dOde = 24.0 * e_mean / eta_mean ** 2 * C * np.cos(i_mean)
        dOdi = 6.0 * C * np.sin(i_mean)
        dwda = -10.5 * C * (5.0 * np.cos(i_mean) ** 2 - 1.0) / a_mean
        dwde = 12.0 * e_mean * C * (5.0 * np.cos(i_mean) ** 2 - 1.0) / eta_mean ** 2
        dwdi = -15.0 * C * np.sin(2.0 * i_mean)
        dMda = -3.0 * n_mean / (2.0 * a_mean) - eta_mean / (2.0 * a_mean) * C * (63.0 * np.cos(i_mean) ** 2 - 21.0)
        dMde = 9.0 * e_mean * C * (3.0 * np.cos(i_mean) ** 2 - 1.0) / eta_mean
        dMdi = -9.0 * eta_mean * C * np.sin(2.0 * i_mean)

        # Initialize best dV and dt
        best_dV = 1e12
        best_dt = 0.0

        # Minimum deltaV deliverable -> 5 mm/s
        dV_min = 5e-6

        # Check all the possible transfers time from tmin to tmax (seconds)
        t_min = int(checkpoint.t_min)
        t_max = int(checkpoint.t_max)

        for dt in xrange(t_min, t_max):
            # Estimate flight time
            E = lambda v: 2.0 * np.arctan(np.sqrt((1.0 - e_0) / (1.0 + e_0)) * np.tan(v / 2.0))
            M = lambda v: (E(v) - e_0 * np.sin(E(v))) % (2.0 * np.pi)

            N_orb = np.floor(dt / T_mean) + int((self.travel_time(self.target.abs_state, v_0, 0.0) < dt) and (dt / T_mean < 1.0))

            # propagated_tg = self.scenario.prop_target.propagate(self.epoch + timedelta(seconds=dt))


            def tau(v):
                if v != v_0:
                    return (2.0 * np.pi * N_orb + M(v) - M_0) / M_mean_dot
                else:
                    return 0.0

            M_f = (dt * M_mean_dot) + M_0 - 2.0 * np.pi * N_orb
            E_f = self._calc_E_from_m(e_0, M_f)
            v_f = (2.0 * np.arctan(np.sqrt((1.0 + e_0) / (1.0 - e_0)) * np.tan(E_f / 2.0))) % (2.0 * np.pi)

            # Position
            r = lambda v: p_0 / (1.0 + e_0 * np.cos(v))

            # Position and true anomaly derivatives
            r_dot = lambda v: a_0 * e_0 * np.sin(v) / eta_0 * M_mean_dot
            v_dot = lambda v: (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3 * M_mean_dot

            # Phi_1
            k_x_dot = lambda v: a_0 * e_0 * v_dot(v) * np.cos(v) / eta_0
            phi_11 = lambda v: r_dot(v) / a_0 + (k_x_dot(v) * tau(v) + a_0 * e_0 * np.sin(v) / eta_0) * dMda
            phi_12 = lambda v: a_0 * v_dot(v) * np.sin(v) + (k_x_dot(v) * tau(v) + a_0 * e_0 * np.sin(v) / eta_0) * \
                               (dMde + dMda * depsde + dMda * depsdv * np.sin(v_0) / eta_0 ** 2 * (2.0 + e_0 * np.cos(e_0)))
            phi_13 = lambda v: (k_x_dot(v) * tau(v) + a_0 * e_0 * np.sin(v) / eta_0) * (dMda * depsdi + dMdi)
            phi_14 = 0.0
            phi_15 = lambda v: (k_x_dot(v) * tau(v) + a_0 * e_0 * np.sin(v) / eta_0) * dMda * depsdw
            phi_16 = lambda v: k_x_dot(v) + (k_x_dot(v) * tau(v) + a_0 * e_0 * np.sin(v) / eta_0) * dMda * depsdv * \
                               (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 3

            # Phi 2
            k_y_dot = lambda v: r_dot(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3 - 2.0 * e_0 * v_dot(v) * np.sin(v) * (
                        1.0 + e_0 * np.cos(v)) / eta_0 ** 3
            phi_21 = lambda v: (r_dot(v) * np.cos(i_0) * tau(v) + r(v) * np.cos(i_0)) * dOda + (
                        r_dot(v) * tau(v) + r(v)) * dwda + \
                               (k_y_dot(v) * tau(v) + r(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3) * dMda
            phi_22 = lambda v: 1.0 / eta_0 ** 2 * (
                        r(v) * v_dot(v) * np.cos(v) * (2.0 + e_0 * np.cos(v)) - r(v) * e_0 * v_dot(v) * np.sin(v) ** 2 +
                        r_dot(v) * np.sin(v) * (2.0 + e_0 * np.cos(v))) + (
                                           r_dot(v) * np.cos(i_0) * tau(v) + r(v) * np.cos(i_0)) * \
                               (dOda * depsde + dOda * depsdv * np.sin(v_0) / eta_0 ** 2 * (
                                           2.0 + e_0 * np.cos(e_0)) + dOde) + \
                               (r_dot(v) * tau(v) + r(v)) * (dwda * depsde + dwda * depsdv * np.sin(v_0) / eta_0 ** 2 * (
                        2.0 + e_0 * np.cos(e_0)) + dwde) + \
                               (k_y_dot(v) * tau(v) + r(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3) * \
                               (dMda * depsde + dMda * depsdv * np.sin(v_0) * (2.0 + e_0 * np.cos(e_0)) / eta_0 ** 2 + dMde)
            phi_23 = lambda v: (r_dot(v) * np.cos(i_0) * tau(v) + r(v) * np.cos(i_0)) * (dOda * depsdi + dOdi) + \
                               (r_dot(v) * tau(v) + r(v)) * (dwda * depsdi + dwdi) + \
                               (k_y_dot(v) * tau(v) + r(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3) * (
                                           dMda * depsdi + dMdi)
            phi_24 = lambda v: r_dot(v) * np.cos(i_0)
            phi_25 = lambda v: r_dot(v) + (r_dot(v) * np.cos(i_0) * tau(v) + r(v) * np.cos(i_0)) * dOda * depsdw + \
                               (r_dot(v) * tau(v) + r(v)) * dwda * depsdw + (k_y_dot(v) * tau(v) + r(v) * (
                        1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3) * dMda * depsdw
            phi_26 = lambda v: k_y_dot(v) + (r_dot(v) * np.cos(i_0) * tau(v) + r(v) * np.cos(i_0)) * dOda * depsdv * (
                        1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 3 + \
                               (r_dot(v) * tau(v) + r(v)) * dwda * depsdv * (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 3 + \
                               (k_y_dot(v) * tau(v) + r(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3) * dOda * depsdv * \
                               (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 3

            # Phi 3
            k_z_dot = lambda v: -r_dot(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0) + \
                                r(v) * np.sin(v + w_0 + w_mean_dot * tau(v)) * (v_dot(v) + w_mean_dot) * np.sin(i_0)
            phi_31 = lambda v: (k_z_dot(v) * tau(v) - r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0)) * dOda
            phi_32 = lambda v: (k_z_dot(v) * tau(v) - r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0)) * \
                               (dOda * depsde + dOda * depsdv * np.sin(v_0) / eta_0 ** 2 * (2.0 + e_0 * np.cos(e_0)) + dOde)
            phi_33 = lambda v: r_dot(v) * np.sin(v + w_0 + w_mean_dot * tau(v)) + r(v) * np.cos(
                v + w_0 + w_mean_dot * tau(v)) * \
                               (v_dot(v) + w_mean_dot) + (
                                           k_z_dot(v) * tau(v) - r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(
                                       i_0)) * dOda
            phi_34 = lambda v: k_z_dot(v)
            phi_35 = lambda v: (k_z_dot(v) * tau(v) - r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(
                i_0)) * dOda * depsdw
            phi_36 = lambda v: (k_z_dot(v) * tau(v) - r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(
                i_0)) * dOda * depsdv * \
                               (1.0 + e_0 * np.cos(e_0)) ** 2 / eta_0 ** 3

            # Phi 4
            phi_41 = lambda v: r(v) / a_0 + a_0 * e_0 * np.sin(v) / eta_0 * dMda * tau(v)
            phi_42 = lambda v: a_0 * e_0 * np.sin(v) / eta_0 * (
                        dMda * depsde + dMda * depsdv * np.sin(v_0) / eta_0 ** 2 * (2.0 + e_0 * np.cos(e_0)) +
                        dMde) * tau(v) - a_0 * np.cos(v)
            phi_43 = lambda v: a_0 * e_0 * np.sin(v) / eta_0 * (dMda * depsdi + dMdi) * tau(v)
            phi_44 = 0.0
            phi_45 = lambda v: a_0 * e_0 * np.sin(v) / eta_0 * dMda * depsdw * tau(v)
            phi_46 = lambda v: a_0 * e_0 * np.sin(v) / eta_0 + a_0 * e_0 * np.sin(v) / eta_0 * dMda * depsdw * \
                               (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 3 * tau(v)

            # Phi 5
            phi_51 = lambda v: r(v) * np.cos(i_0) * dOda * tau(v) + r(v) * dwda * tau(v) + r(v) * (
                        1.0 + e_0 * np.cos(v)) ** 2 \
                               / eta_0 ** 3 * dMda * tau(v)
            phi_52 = lambda v: r(v) * np.sin(v) / eta_0 ** 2 * (2.0 + e_0 * np.cos(v)) + r(v) * np.cos(i_0) * \
                               (dOda * depsde + dOda * depsdv * np.sin(v_0) / eta_0 ** 2 * (
                                           2.0 + e_0 * np.cos(e_0)) + dOde) * tau(v) + \
                               r(v) * (dwda * depsde + dwda * depsdv * np.sin(v_0) / eta_0 ** 2 * (
                        2.0 + e_0 * np.cos(e_0)) + dwde) * tau(v) + \
                               r(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3 * \
                               (dMda * depsde + dMda * depsdv * np.sin(v_0) / eta_0 ** 2 * (
                                           2.0 + e_0 * np.cos(e_0)) + dMde) * tau(v)
            phi_53 = lambda v: r(v) * np.cos(i_0) * (dOda * depsdi + dOdi) * tau(v) + r(v) * (dwda * depsdi + dwdi) * tau(
                v) + \
                               r(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3 * (dMda * depsdi + dMdi) * tau(v)
            phi_54 = lambda v: r(v) * np.cos(i_0)
            phi_55 = lambda v: r(v) + r(v) * np.cos(i_0) * dOda * depsdw * tau(v) + r(v) * dwda * depsdw * tau(v) + \
                               r(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3 * dMda * depsdw * tau(v)
            phi_56 = lambda v: r(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3 + r(v) * np.cos(i_0) * dOda * depsdv * \
                               (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 3 * tau(v) + r(v) * dwda * depsdv * (
                                           1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 3 * tau(v) + \
                               r(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3 * dMda * depsdv * (
                                           1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 3 * tau(v)

            # Phi 6
            phi_61 = lambda v: -r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0) * dOda * tau(v)
            phi_62 = lambda v: -r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0) * (
                        dOda * depsde + dOda * depsdv * np.sin(v_0) /
                        eta_0 ** 2 * (2.0 + e_0 * np.cos(e_0)) + dOde) * tau(v)
            phi_63 = lambda v: r(v) * np.sin(v + w_0 + w_mean_dot * tau(v)) - r(v) * np.cos(
                v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0) * (dOda * depsdi + dOdi) * tau(v)
            phi_64 = lambda v: -r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0)
            phi_65 = lambda v: -r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0) * dOda * depsdw * tau(v)
            phi_66 = lambda v: -r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0) * dOda * depsdv * (
                        1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 3 * tau(v)

            phi_i = np.array([
                [phi_11(v_0), phi_12(v_0), phi_13(v_0), phi_14, phi_15(v_0), phi_16(v_0)],
                [phi_21(v_0), phi_22(v_0), phi_23(v_0), phi_24(v_0), phi_25(v_0), phi_26(v_0)],
                [phi_31(v_0), phi_32(v_0), phi_33(v_0), phi_34(v_0), phi_35(v_0), phi_36(v_0)],
                [phi_41(v_0), phi_42(v_0), phi_43(v_0), phi_44, phi_45(v_0), phi_46(v_0)],
                [phi_51(v_0), phi_52(v_0), phi_53(v_0), phi_54(v_0), phi_55(v_0), phi_56(v_0)],
                [phi_61(v_0), phi_62(v_0), phi_63(v_0), phi_64(v_0), phi_65(v_0), phi_66(v_0)]
            ])

            phi_f = np.array([
                [phi_11(v_f), phi_12(v_f), phi_13(v_f), phi_14, phi_15(v_f), phi_16(v_f)],
                [phi_21(v_f), phi_22(v_f), phi_23(v_f), phi_24(v_f), phi_25(v_f), phi_26(v_f)],
                [phi_31(v_f), phi_32(v_f), phi_33(v_f), phi_34(v_f), phi_35(v_f), phi_36(v_f)],
                [phi_41(v_f), phi_42(v_f), phi_43(v_f), phi_44, phi_45(v_f), phi_46(v_f)],
                [phi_51(v_f), phi_52(v_f), phi_53(v_f), phi_54(v_f), phi_55(v_f), phi_56(v_f)],
                [phi_61(v_f), phi_62(v_f), phi_63(v_f), phi_64(v_f), phi_65(v_f), phi_66(v_f)]
            ])

            phi_comb = np.array([
                phi_i[0:6][3],
                phi_i[0:6][4],
                phi_i[0:6][5],
                phi_f[0:6][3],
                phi_f[0:6][4],
                phi_f[0:6][5]
            ])

            state_comb = np.array([self.chaser.rel_state.R[0],
                                   self.chaser.rel_state.R[1],
                                   self.chaser.rel_state.R[2],
                                   checkpoint.rel_state.R[0],
                                   checkpoint.rel_state.R[1],
                                   checkpoint.rel_state.R[2]])

            de0_wanted = np.linalg.inv(phi_comb).dot(state_comb)
            de0_diff = de0_wanted - de0_initial

            chaser_kep_wanted = KepOrbElem()
            chaser_kep_wanted.a = self.chaser.abs_state.a + de0_diff[0]
            chaser_kep_wanted.e = self.chaser.abs_state.e + de0_diff[1]
            chaser_kep_wanted.i = self.chaser.abs_state.i + de0_diff[2]
            chaser_kep_wanted.O = self.chaser.abs_state.O + de0_diff[3]
            chaser_kep_wanted.w = self.chaser.abs_state.w + de0_diff[4]
            chaser_kep_wanted.m = self.chaser.abs_state.m + de0_diff[5]

            chaser_cart_wanted = Cartesian()
            chaser_cart_wanted.from_keporb(chaser_kep_wanted)

            # Evaluate delta_V_1
            delta_V_1 = chaser_cart_wanted.V - chaser_cart.V


            # Evaluate delta_V_2
            state_f = phi_f.dot(de0_wanted)
            delta_V_2 = checkpoint.rel_state.V - state_f[0:3]

            # self._change_propagator_ic(self.scenario.prop_chaser, chaser_cart_wanted, self.epoch, self.chaser.mass)
            #
            # prop_chaser = self.scenario.prop_chaser.propagate(self.epoch + timedelta(seconds=dt))
            # prop_target = self.scenario.prop_target.propagate(self.epoch + timedelta(seconds=dt))
            #
            #
            # dr_TEME = prop_chaser[0].R - prop_target[0].R
            # dr_LVLH = prop_target[0].get_lof().dot(dr_TEME)

            deltaV_tot = np.linalg.norm(delta_V_1) + np.linalg.norm(delta_V_2)

            if best_dV > deltaV_tot:
                # if self.is_trajectory_safe(dt, approach_ellipsoid):
                best_dV = deltaV_tot
                best_dV_1 = delta_V_1
                best_dV_2 = delta_V_2
                best_dt = dt

        c1 = RelativeMan()
        c1.dV = best_dV_1
        c1.set_abs_state(self.chaser.abs_state)
        c1.set_rel_state(self.chaser.rel_state)
        c1.duration = 0
        c1.description = 'Linearized-J2 solution'
        self.manoeuvre_plan.append(c1)

        self._propagator(1e-3, best_dV_1)

        chaser_cart.from_keporb(self.chaser.abs_state)

        c2 = RelativeMan()
        c2.dV = best_dV_2
        c2.set_abs_state(self.chaser.abs_state)
        c2.set_rel_state(self.chaser.rel_state)
        c2.duration = best_dt
        c2.description = 'Linearized-J2 solution'
        self.manoeuvre_plan.append(c2)

        self._propagator(best_dt, best_dV_2)

    def _change_propagator_ic(self, propagator, initial_state, epoch, mass):
        """
            Allows to change the initial conditions given to the propagator without initializing it again.

        Args:
            propagator (OrekitPropagator): The propagator that has to be changed.
            initial_state (Cartesian): New cartesian coordinates of the initial state.
            epoch (datetime): New starting epoch.
            mass (float64): Satellite mass.
        """

        # Create position and velocity vectors as Vector3D
        p = Vector3D(float(initial_state.R[0]) * 1e3, float(initial_state.R[1]) * 1e3,
                     float(initial_state.R[2]) * 1e3)
        v = Vector3D(float(initial_state.V[0]) * 1e3, float(initial_state.V[1]) * 1e3,
                     float(initial_state.V[2]) * 1e3)

        # Initialize orekit date
        seconds = float(epoch.second) + float(epoch.microsecond) / 1e6
        orekit_date = AbsoluteDate(epoch.year,
                                   epoch.month,
                                   epoch.day,
                                   epoch.hour,
                                   epoch.minute,
                                   seconds,
                                   TimeScalesFactory.getUTC())

        # Extract frame
        inertialFrame = FramesFactory.getEME2000()

        # Evaluate new initial orbit
        initialOrbit = CartesianOrbit(PVCoordinates(p, v), inertialFrame, orekit_date, Cst.WGS84_EARTH_MU)

        # Create new spacecraft state
        newSpacecraftState = SpacecraftState(initialOrbit, mass)

        # Rewrite propagator initial conditions
        propagator._propagator_num.setInitialState(newSpacecraftState)

    def find_best_deltaV(self, checkpoint):
        """
            Find the best deltaV given the checkpoint that the chaser has to reach next.

        Args:
            checkpoint (Checkpoint):
        """


        if self.scenario.prop_type == 'real-world':
            pass
        elif self.scenario.prop_type == '2-body':
            pass
        else:
            print "Propagator type not known!"
            raise TypeError

    def _calc_E_from_m(self, e, m):
        """Calculates Eccentric anomaly from Mean anomaly

        Uses a Newton-Raphson iteration to solve Kepler's Equation.
        Source: Algorithm 3.1 in [1]

        Prerequisites: m and e

        """
        if m < np.pi:
            E = m + e / 2.0
        else:
            E = m - e / 2.0

        max_int = 20  # maximum number of iterations

        while max_int > 1:
            fE = E - e * np.sin(E) - m
            fpE = 1.0 - e * np.cos(E)
            ratio = fE / fpE
            max_int = max_int - 1

            # check if ratio is small enough
            if abs(ratio) > 1e-15:
                E = E - ratio
            else:
                break

        if E < 0:
            E = E + np.pi * 2.0

        return E