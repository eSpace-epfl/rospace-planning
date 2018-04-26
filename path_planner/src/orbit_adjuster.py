# @copyright Copyright (c) 2017, Davide Frey (frey.davide.ae@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

"""Class to create an object capable of calculating the correction from an orbit to another."""

import numpy as np
import pykep as pk

from state import Satellite, Chaser
from rospace_lib import mu_earth
from datetime import timedelta
from manoeuvre import Manoeuvre, RelativeMan
from rospace_lib import CartesianTEME, KepOrbElem
from checkpoint import AbsoluteCP


class OrbitAdjuster(object):
    """
        Base class of an orbit adjuster.
    """

    def travel_time(self, abs_state, theta0, theta1):
        """
            Evaluate the travel time of a satellite from a starting true anomaly theta0 to an end anomaly theta1.

        Reference:
            Exercise of Nicollier's Lecture.
            David A. Vallado, Fundamentals of Astrodynamics and Applications, Second Edition, Algorithm 11 (p. 133)

        Args:
            abs_state (KepOrbElem): Satellite state in keplerian orbital elements.
            theta0 (rad): Starting true anomaly.
            theta1 (rad): Ending true anomaly.

        Return:
            Travel time (seconds)
        """

        a = abs_state.a
        e = abs_state.e

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

    def create_and_apply_manoeuvre(self, chaser, target, deltaV, dt):
        """
            Take the amount of deltaV needed and the waiting time up to when the manoeuvre should be executed, create
            the manoeuvre and apply it, propagating chaser and target.

        Args:
            chaser (Chaser)
            target (Satellite)
            deltaV (np.array): Array in TEME reference frame cointaing the deltaV in [km/s]
            dt (float): Waiting time in [s] up to when the burn should be executed.

        Return:
            man (Manoeuvre): The manoeuvre to be added to manoeuvre plan
        """

        # Define new starting epoch
        new_epoch = chaser.prop.date + timedelta(seconds=dt)

        # Propagate both satellite and target to the new epoch
        satellite_prop = chaser.prop.orekit_prop.propagate(new_epoch)
        target_prop = target.prop.orekit_prop.propagate(new_epoch)

        # Apply deltaV to satellite and update absolute and relative states of target and satellite
        satellite_prop[0].V += deltaV
        chaser.set_abs_state_from_cartesian(satellite_prop[0])
        target.set_abs_state_from_cartesian(target_prop[0])
        chaser.rel_state.from_cartesian_pair(chaser.abs_state, target.abs_state)

        # Reset propagators initial conditions
        chaser.prop.change_initial_conditions(satellite_prop[0], new_epoch, chaser.mass)
        target.prop.change_initial_conditions(target_prop[0], new_epoch, target.mass)

        # Reset propagators initial starting date
        chaser.prop.date = new_epoch
        target.prop.date = new_epoch

        # Create manoeuvre
        man = Manoeuvre()
        man.deltaV = deltaV
        man.execution_epoch = new_epoch

        return man


class HohmannTransfer(OrbitAdjuster):
    """
        Subclass holding the function to evaluate a Hohmann Transfer.
        Can be used to do corrections during absolute navigation.
    """

    def is_necessary(self, chaser, checkpoint):
        """
            Function to test if this type of orbit adjuster is needed.
        """

        mean_oe = chaser.get_mean_oe()

        da = checkpoint.abs_state.a - mean_oe.a
        de = checkpoint.abs_state.e - mean_oe.e

        tol_a = 0.2
        tol_e = 1.0 / mean_oe.a

        if abs(da) > tol_a or abs(de) > tol_e:
            return True
        else:
            return False

    def evaluate_manoeuvre(self, chaser, checkpoint, target):
        """
            Adjust eccentricity and semi-major axis at the same time with an Hohmann-Transfer like manoeuvre:
            1) Burn at perigee to match the needed intermediate orbit
            2) Burn at apogee to arrive at the final, wanted orbit

        References:
            Howard Curtis, Orbital Mechanics for Engineering Students, Chapter 6
            David A. Vallado, Fundamentals of Astrodynamics and Applications, Second Edition, Chapter 6
        """

        # Evaluating mean orbital elements
        mean_oe = chaser.get_mean_oe()

        # Extract initial and final semi-major axis and eccentricities
        a_i = mean_oe.a
        e_i = mean_oe.e
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
        deltaV_C_1 = np.linalg.inv(mean_oe.get_pof()).dot(V_PERI_f_1 - V_PERI_i_1)

        # Apply first deltaV and give some amount of time (at least 10 seconds) to react
        dt = self.travel_time(mean_oe, mean_oe.v, theta_1)
        if dt < 10.0:
            dt += self.travel_time(mean_oe, 0.0, 2.0*np.pi)

        man1 = self.create_and_apply_manoeuvre(chaser, target, deltaV_C_1, dt)

        # Second burn
        V_PERI_i_2 = np.sqrt(mu_earth / (a_int * (1.0 - e_int ** 2))) * np.array([-np.sin(theta_2), e_int + np.cos(theta_2), 0.0])
        V_PERI_f_2 = np.sqrt(mu_earth / (a_f * (1.0 - e_f ** 2))) * np.array([-np.sin(theta_2), e_f + np.cos(theta_2), 0.0])
        deltaV_C_2 = np.linalg.inv(mean_oe.get_pof()).dot(V_PERI_f_2 - V_PERI_i_2)

        # Apply second deltaV and give some amount of time (at least 10 seconds) to react
        mean_oe = chaser.get_mean_oe()
        dt = self.travel_time(mean_oe, mean_oe.v, theta_2)
        if dt < 10.0:
            dt += self.travel_time(mean_oe, 0.0, 2.0*np.pi)

        man2 = self.create_and_apply_manoeuvre(chaser, target, deltaV_C_2, dt)

        return [man1, man2]


class ArgumentOfPerigee(OrbitAdjuster):
    """
        Subclass holding the function to evaluate the deltaV needed to change the argument of perigee.
        Can be used to do corrections during absolute navigation.
    """

    def is_necessary(self, chaser, checkpoint):
        """
            Function to test if this type of orbit adjuster is needed.
        """

        mean_oe = chaser.get_mean_oe()

        dw = checkpoint.abs_state.w - mean_oe.w

        tol_w = 1.0 / mean_oe.a

        if abs(dw) > tol_w:
            return True
        else:
            return False

    def evaluate_manoeuvre(self, chaser, checkpoint, target):
        """
            Given the chaser relative orbital elements with respect to the target adjust the perigee argument.

        References:
            Howard Curtis, Orbital Mechanics for Engineering Students, Chapter 6
            David A. Vallado, Fundamentals of Astrodynamics and Applications, Second Edition, Chapter 6
        """
        # Mean orbital elements
        mean_oe = chaser.get_mean_oe()

        # Extract constants
        a = mean_oe.a
        e = mean_oe.e

        # Evaluate perigee difference to correct
        dw = (checkpoint.abs_state.w - mean_oe.w) % (2.0 * np.pi)

        # Positions where burn can occur
        theta_i_1 = dw / 2.0
        theta_i_2 = theta_i_1 + np.pi
        theta_f_1 = 2.0 * np.pi - theta_i_1
        theta_f_2 = theta_f_1 - np.pi

        # Check which one is the closest TODO: Check the least consuming instead of the closest
        if theta_i_1 < mean_oe.v:
            dv1 = 2.0 * np.pi + theta_i_1 - mean_oe.v
        else:
            dv1 = theta_i_1 - mean_oe.v

        if theta_i_2 < mean_oe.v:
            dv2 = 2.0 * np.pi + theta_i_2 - mean_oe.v
        else:
            dv2 = theta_i_2 - mean_oe.v

        if dv1 > dv2:
            theta_i = theta_i_2
            theta_f = theta_f_2
        else:
            theta_i = theta_i_1
            theta_f = theta_f_1

        # Initial velocity
        V_PERI_i = np.sqrt(mu_earth / (a * (1.0 - e**2))) * np.array([-np.sin(theta_i), e + np.cos(theta_i), 0.0])
        V_TEM_i = np.linalg.inv(mean_oe.get_pof()).dot(V_PERI_i)

        # Final velocity
        V_PERI_f = np.sqrt(mu_earth / (a * (1.0 - e**2))) * np.array([-np.sin(theta_f), e + np.cos(theta_f), 0.0])
        V_TEM_f = np.linalg.inv(checkpoint.abs_state.get_pof()).dot(V_PERI_f)

        # Delta-V
        deltaV_C = V_TEM_f - V_TEM_i

        # Apply deltaV
        dt = self.travel_time(mean_oe, mean_oe.v, theta_i)

        man = self.create_and_apply_manoeuvre(chaser, target, deltaV_C, dt)

        return [man]


class PlaneOrientation(OrbitAdjuster):
    """
        Subclass holding the function to evaluate the deltaV needed to change in plane orientation.
        Can be used to do corrections during absolute navigation.
    """

    def is_necessary(self, chaser, checkpoint):
        """
            Function to test if this type of orbit adjuster is needed.
        """

        mean_oe = chaser.get_mean_oe()

        di = checkpoint.abs_state.i - mean_oe.i
        dO = checkpoint.abs_state.O - mean_oe.O

        tol_i = 1.0 / mean_oe.a
        tol_O = 1.0 / mean_oe.a

        if abs(di) > tol_i or abs(dO) > tol_O:
            return True
        else:
            return False

    def evaluate_manoeuvre(self, chaser, checkpoint, target):
        """
            Correct plane inclination and RAAN with a single manoeuvre at the node between the two orbital planes.

        References:
            Howard Curtis, Orbital Mechanics for Engineering Students, Chapter 6
            David A. Vallado, Fundamentals of Astrodynamics and Applications, Second Edition, Chapter 6
        """
        # Mean orbital elements
        mean_oe = chaser.get_mean_oe()

        # Extract values
        a = mean_oe.a
        e = mean_oe.e
        i_i = mean_oe.i
        O_i = mean_oe.O

        # Final values
        O_f = checkpoint.abs_state.O
        i_f = checkpoint.abs_state.i

        # Difference between initial and final values
        dO = O_f - O_i
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
        theta_1 = (2.0 * np.pi - A_Li - mean_oe.w) % (2.0 * np.pi)
        theta_2 = (theta_1 + np.pi) % (2.0 * np.pi)

        # Choose which of the two position is the closest
        # They consume different dV, the decision has to be taken then depending on if you want to spent a bit more
        # and burn in a specific point, or if you can born anywhere regardless on how much it will cost.
        # Now it's just taking the closest point to do the burn, to decrease the total time of the mission.
        if theta_1 < mean_oe.v:
            dv1 = 2*np.pi + theta_1 - mean_oe.v
        else:
            dv1 = theta_1 - mean_oe.v

        if theta_2 < mean_oe.v:
            dv2 = 2*np.pi + theta_2 - mean_oe.v
        else:
            dv2 = theta_2 - mean_oe.v

        if dv1 > dv2:
            theta_i = theta_2
        else:
            theta_i = theta_1

        # Define vector c in Earth-Inertial frame of reference
        cx = np.cos(psi) * np.cos(phi)
        cy = np.cos(psi) * np.sin(phi)
        cz = np.sin(psi)

        # TODO: check this, for some cases it seems to be wrong
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
        V_TEM_i = np.linalg.inv(mean_oe.get_pof()).dot(V_PERI_i)

        # Rotate vector around c by alpha radiants
        V_TEM_f = R_c.dot(V_TEM_i)

        # Evaluate deltaV
        deltaV_C = V_TEM_f - V_TEM_i

        # Apply first deltaV
        dt = self.travel_time(mean_oe, mean_oe.v, theta_1)

        man = self.create_and_apply_manoeuvre(chaser, target, deltaV_C, dt)

        return [man]


class AnomalySynchronisation(OrbitAdjuster):

    def evaluate_manoeuvre(self):
        pass


class Drift(OrbitAdjuster):
    """
        Subclass holding the function to drift from one point to another.
        Can be used to do corrections during relative navigation.
    """

    def evaluate_manoeuvre(self, chaser, checkpoint, target, manoeuvre_plan):
        """
            Algorithm that tries to drift to the next checkpoint, staying within a certain error ellipsoid.

        Args:
            chaser (Chaser)
            checkpoint (RelativeCP): Next checkpoint.
            target (Satellite)

        Return:
            man (Manoeuvre)
        """

        # Creating old chaser and target objects to store their temporary value
        chaser_old = Chaser()
        target_old = Satellite()

        # Define a function F for the angle calculation
        F = lambda dv_req, dv, n: int((dv - dv_req) / n > 0.0) * np.sign(n)

        # Correct altitude at every loop until drifting is possible
        while 1:
            # Assign mean values from osculating
            chaser_mean = chaser.get_mean_oe()
            target_mean = target.get_mean_oe()

            # Assign information to the new chaser and target objects
            chaser_old.set_from_satellite(chaser)
            target_old.set_from_satellite(target)

            # Store initial epoch
            epoch_old = chaser.prop.date

            # Evaluate relative mean angular velocity. If it's below zero chaser moves slower than target,
            # otherwise faster
            n_c = np.sqrt(mu_earth / chaser_mean.a ** 3)
            n_t = np.sqrt(mu_earth / target_mean.a ** 3)
            n_rel = n_c - n_t

            # Required true anomaly difference at the end of the manoeuvre, estimation assuming circular
            # orbit
            dv_req = checkpoint.rel_state.R[1] / np.linalg.norm(chaser.abs_state.R)

            # Evaluate the actual true anomaly difference
            actual_dv = (chaser_mean.v + chaser_mean.w) % (2.0 * np.pi) - (target_mean.v + target_mean.w) % (
            2.0 * np.pi)

            # Millisecond tolerance to exit the loop
            tol = 1e-3

            chaser_tmp = Chaser()
            target_tmp = Satellite()

            manoeuvre_plan_old = manoeuvre_plan

            t_est = (2.0 * np.pi * F(dv_req, actual_dv, n_rel) + dv_req - actual_dv) / n_rel
            ellipsoid_flag = False
            dt = 10**np.floor(np.log10(t_est)) if t_est / (10**np.floor(np.log10(t_est))) >= 2.0 else 10**np.floor(np.log10(t_est) - 1.0)
            dr_next_old = 0.0
            dr_next = 0.0
            while dt > tol:
                # Store (i-1) chaser and target state
                chaser_tmp.set_from_satellite(chaser)
                target_tmp.set_from_satellite(target)
                epoch_tmp = chaser.prop.date
                dr_next_tmp = dr_next
                manoeuvre_plan_tmp = manoeuvre_plan

                # Update epoch
                chaser.prop.date += timedelta(seconds=dt)
                target.prop.date += timedelta(seconds=dt)

                # Propagate
                chaser_prop = chaser.prop.orekit_prop.propagate(chaser.prop.date)
                target_prop = target.prop.orekit_prop.propagate(target.prop.date)

                chaser.set_abs_state_from_cartesian(chaser_prop[0])
                target.set_abs_state_from_cartesian(target_prop[0])
                chaser.rel_state.from_cartesian_pair(chaser_prop[0], target_prop[0])

                # Re-initialize propagators
                chaser.prop.change_initial_conditions(chaser_prop[0], chaser.prop.date, chaser.mass)
                target.prop.change_initial_conditions(target_prop[0], target.prop.date, target.mass)

                dr_next = chaser.rel_state.R[1] - checkpoint.rel_state.R[1]

                if dr_next <= 0.0 and dr_next_old <= 0.0:
                    # Correct plane in the middle of the drifting
                    tol_i = 0.5 / chaser_mean.a
                    tol_O = 0.5 / chaser_mean.a

                    chaser_mean = chaser.get_mean_oe()
                    target_mean = target.get_mean_oe()

                    # At this point, inclination and raan should match the one of the target
                    di = target_mean.i - chaser_mean.i
                    dO = target_mean.O - chaser_mean.O
                    if abs(di) > tol_i or abs(dO) > tol_O:
                        checkpoint_abs = AbsoluteCP()
                        checkpoint_abs.abs_state.i = target_mean.i
                        checkpoint_abs.abs_state.O = target_mean.O

                        orbit_adj = PlaneOrientation()
                        orbit_adj.evaluate_manoeuvre(chaser, checkpoint_abs, target)

                        dr_next = chaser.rel_state.R[1] - checkpoint.rel_state.R[1]

                        if dr_next >= 0.0:
                            # Overshoot due to plane adjustment => reduce dt and depropagate
                            dt /= 10.0
                            chaser.set_from_satellite(chaser_tmp)
                            target.set_from_satellite(target_tmp)

                            chaser.prop.date = epoch_tmp
                            target.prop.date = epoch_tmp

                            chaser_cartesian_tmp = CartesianTEME()
                            chaser_cartesian_tmp.R = chaser.abs_state.R
                            chaser_cartesian_tmp.V = chaser.abs_state.V

                            target_cartesian_tmp = CartesianTEME()
                            target_cartesian_tmp.R = target.abs_state.R
                            target_cartesian_tmp.V = target.abs_state.V

                            chaser.prop.change_initial_conditions(chaser_cartesian_tmp, chaser.prop.date, chaser.mass)
                            target.prop.change_initial_conditions(target_cartesian_tmp, target.prop.date, target.mass)

                            manoeuvre_plan = manoeuvre_plan_tmp

                            # dr_next_old should be the same as the one at the beginning
                            dr_next = dr_next_tmp
                        else:
                            # Target point not overshooted, everything looks good as it is
                            man = RelativeMan()
                            man.deltaV = np.array([0.0, 0.0, 0.0])
                            man.set_initial_rel_state(chaser_tmp.rel_state)
                            man.execution_epoch = chaser.prop.date
                            man.description = 'Drift for ' + str(dt) + ' seconds'

                            manoeuvre_plan.append(man)

                            dr_next_old = dr_next

                    else:
                        # No plane adjustment needed, add another dt and move forward
                        man = RelativeMan()
                        man.deltaV = np.array([0.0, 0.0, 0.0])
                        man.set_initial_rel_state(chaser_tmp.rel_state)
                        man.execution_epoch = chaser.prop.date
                        man.description = 'Drift for ' + str(dt) + ' seconds'

                        manoeuvre_plan.append(man)

                        dr_next_old = dr_next

                elif dr_next >= 0.0 and dr_next_old >= 0.0:
                    # Only useful for the case when chaser is on a higher orbit
                    pass

                elif (dr_next <= 0.0 and dr_next_old >= 0.0) or (dr_next >= 0.0 and dr_next_old <= 0.0):
                    dt /= 10.0
                    chaser.set_from_satellite(chaser_tmp)
                    target.set_from_satellite(target_tmp)

                    chaser.prop.date = epoch_tmp
                    target.prop.date = epoch_tmp

                    chaser_cart_tmp = CartesianTEME()
                    target_cart_tmp = CartesianTEME()

                    chaser_cart_tmp.R = chaser_tmp.abs_state.R
                    chaser_cart_tmp.V = chaser_tmp.abs_state.V
                    target_cart_tmp.R = target_tmp.abs_state.R
                    target_cart_tmp.V = target_tmp.abs_state.V

                    chaser.prop.change_initial_conditions(chaser_cart_tmp, chaser.prop.date, chaser_tmp.mass)
                    target.prop.change_initial_conditions(target_cart_tmp, target.prop.date, target_tmp.mass)
                    # dr_next_old should be the same as the one at the beginning
                    dr_next = dr_next_tmp

                if abs(checkpoint.rel_state.R[1] - chaser.rel_state.R[1]) <= checkpoint.error_ellipsoid[1]:
                    # Almost in line with the checkpoint
                    if abs(checkpoint.rel_state.R[0] - chaser.rel_state.R[0]) <= checkpoint.error_ellipsoid[0]:
                        # Inside the tolerance, the point may be reached by drifting
                        ellipsoid_flag = True
                    elif abs(checkpoint.rel_state.R[1] - chaser.rel_state.R[1]) <= 0.05 and \
                        abs(checkpoint.rel_state.R[0] - chaser.rel_state.R[0]) > checkpoint.error_ellipsoid[0]:
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
                chaser.set_from_satellite(chaser_old)
                target.set_from_satellite(target_old)

                chaser_mean = chaser.get_mean_oe()
                target_mean = target.get_mean_oe()

                chaser.prop.date = epoch_old
                target.prop.date = epoch_old

                chaser.prop.change_initial_conditions(chaser_old.abs_state, chaser.prop.date, chaser_old.mass)
                target.prop.change_initial_conditions(target_old.abs_state, target.prop.date, target_old.mass)

                manoeuvre_plan = manoeuvre_plan_old

                # Create new checkpoint
                checkpoint_new_abs = AbsoluteCP()
                checkpoint_new_abs.set_abs_state(chaser_mean)
                checkpoint_new_abs.abs_state.a = target_mean.a + checkpoint.rel_state.R[0]
                checkpoint_new_abs.abs_state.e = target_mean.a * target_mean.e / checkpoint_new_abs.abs_state.a

                orbit_adj = HohmannTransfer()
                orbit_adj.evaluate_manoeuvre(chaser, checkpoint_new_abs, target)


class MultiLambert(OrbitAdjuster):
    """
        Subclass holding the function to evaluate the deltaV needed to do a manoeuvre in LVLH frame, using the
        exact solution to the 2-body problem through a multi-lambert solver implemented in the pykep library.

        NOT suited when using real-world propagator, error can become very large!

        Can be used to do corrections during relative navigation.
    """

    def evaluate_manoeuvre(self, chaser, checkpoint, target, approach_ellipsoid, safety_flag):
        """
            Solve the Multi-Lambert Problem.

        Args:
            chaser (Chaser): Chaser state.
            checkpoint (RelativeCP): Next checkpoint.
            target (Satellite): Target state.
            approach_ellipsoid: Approach ellipsoid drawn around the target to be avoided during manoeuvering.
        """

        mean_oe = chaser.get_mean_oe()

        # Check if trajectory is retrograde
        retrograde = False
        if mean_oe.i > np.pi / 2.0:
            retrograde = True

        # Absolute position of chaser at t = t0
        R_C_i = chaser.abs_state.R
        V_C_i = chaser.abs_state.V

        # Absolute position of the target at t = t0
        R_T_i = target.abs_state.R
        V_T_i = target.abs_state.V

        # Create temporary target that will keep the initial conditions
        target_ic = CartesianTEME()
        chaser_ic = CartesianTEME()
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
            # r_T, v_T = pk.propagate_lagrangian(R_T_i, V_T_i, dt, mu_earth)

            target_prop = target.prop.orekit_prop.propagate(chaser.prop.date + timedelta(seconds=dt))

            target.set_abs_state_from_cartesian(target_prop[0])

            # Transformation matrix from TEME to LVLH at time t1
            B_LVLH_TEME_f = target.abs_state.get_lof()

            # Evaluate final wanted absolute position of the chaser
            R_C_f = np.array(target.abs_state.R) + np.linalg.inv(B_LVLH_TEME_f).dot(checkpoint.rel_state.R)
            O_T_f = np.cross(target.abs_state.R, target.abs_state.V) / np.linalg.norm(target.abs_state.R) ** 2
            V_C_f = np.array(target.abs_state.V) + np.linalg.inv(B_LVLH_TEME_f).dot(checkpoint.rel_state.V) + \
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

        man1 = self.create_and_apply_manoeuvre(chaser, target, best_dV_1, 1e-3)
        man2 = self.create_and_apply_manoeuvre(chaser, target, best_dV_2, best_dt)

        return [man1, man2]


class ClohessyWiltshire(OrbitAdjuster):
    """
        Subclass holding the function to evaluate the deltaV needed to do a manoeuvre in LVLH frame, using the
        linearized solution of Clohessy-Wiltshire.

        NOT suited when using real-world propagator, error can become very large. Moreover not suited also when the
        distance over the orbit radius is too big!

        Can be used to do corrections during relative navigation.
    """

    def evaluate_manoeuvre(self, chaser, checkpoint, target):
        """
            Solve Hill's Equation to get the amount of DeltaV needed to go to the next checkpoint.

        References:
            David A. Vallado, Fundamentals of Astrodynamics and Applications, Second Edition, Algorithm 47 (p. 382)

        Args:
            target (Satellite): Target state.
        """

        #TODO: Fix it -> not giving the right result at the moment

        target_mean = target.get_mean_oe()

        t_min = int(checkpoint.t_min)
        t_max = int(checkpoint.t_max)

        R_C_LVLH = chaser.rel_state.R
        V_C_LVLH = chaser.rel_state.V

        R_LVLH = checkpoint.rel_state.R
        V_LVLH = checkpoint.rel_state.V

        n = np.sqrt(mu_earth / target_mean.a ** 3.0)

        phi_rr = lambda t: np.array([
            [4.0 - 3.0 * np.cos(n * t), 0.0, 0.0],
            [6.0 * (np.sin(n * t) - n * t), 1.0, 0.0],
            [0.0, 0.0, np.cos(n * t)]
        ])

        phi_rv = lambda t: np.array([
            [1.0 / n * np.sin(n * t), 2.0 / n * (1 - np.cos(n * t)), 0.0],
            [2.0 / n * (np.cos(n * t) - 1.0), 1.0 / n * (4.0 * np.sin(n * t) - 3.0 * n * t), 0.0],
            [0.0, 0.0, 1.0 / n * np.sin(n * t)]
        ])

        phi_vr = lambda t: np.array([
            [3.0 * n * np.sin(n * t), 0.0, 0.0],
            [6.0 * n * (np.cos(n * t) - 1), 0.0, 0.0],
            [0.0, 0.0, -n * np.sin(n * t)]
        ])

        phi_vv = lambda t: np.array([
            [np.cos(n * t), 2.0 * np.sin(n * t), 0.0],
            [-2.0 * np.sin(n * t), 4.0 * np.cos(n * t) - 3.0, 0.0],
            [0.0, 0.0, np.cos(n * t)]
        ])

        best_deltaV = 1e12
        delta_T = 0

        for t_ in xrange(t_min, t_max):
            rv_t = phi_rv(t_)
            deltaV_1 = np.linalg.inv(rv_t).dot(R_LVLH - phi_rr(t_).dot(R_C_LVLH)) - V_C_LVLH
            deltaV_2 = np.dot(phi_vr(t_), R_C_LVLH) + np.dot(phi_vv(t_), V_C_LVLH + deltaV_1) - V_LVLH

            deltaV_tot = np.linalg.norm(deltaV_1) + np.linalg.norm(deltaV_2)

            if best_deltaV > deltaV_tot:
                best_deltaV = deltaV_tot
                best_deltaV_1 = deltaV_1
                best_deltaV_2 = deltaV_2
                delta_T = t_

        # Change frame of reference of deltaV. From LVLH to Earth-Inertial
        B = target.abs_state.get_lof()
        deltaV_C_1 = np.linalg.inv(B).dot(best_deltaV_1)

        man1 = self.create_and_apply_manoeuvre(chaser, target, deltaV_C_1, 1e-3)

        R = target.abs_state.get_lof()
        deltaV_C_2 = np.linalg.inv(R).dot(best_deltaV_2)

        man2 = self.create_and_apply_manoeuvre(chaser, target, deltaV_C_2, delta_T)

        return [man1, man2]


class TschaunerHempel(OrbitAdjuster):

    def evaluate_manoeuvre(self, chaser, checkpoint, target):
        """
            Tschauner Hempel solver.


        Reference:
            Guidance, Navigation and Control for Satellite Proximity Operations using Tschauner-Hempel equations.
        """

        # TODO: Fix it -> not giving the right result at the moment

        # Get initial mean orbital elements of target
        mean_oe_target = target.get_mean_oe()

        a = mean_oe_target.a
        e = mean_oe_target.e
        v_0 = mean_oe_target.v

        # Get initial cartesian position of the target
        R_T_0 = target.abs_state.R
        V_T_0 = target.abs_state.V
        B_0 = target.abs_state.get_lof()

        p = a * (1.0 - e ** 2)
        h = np.sqrt(p * mu_earth)

        eta = np.sqrt(1.0 - e ** 2)
        n = np.sqrt(mu_earth / a ** 3)
        v_dot = h / np.linalg.norm(target.abs_state.R) ** 2

        rho = lambda v: 1.0 + e * np.cos(v)
        c2 = lambda v: e * np.sin(v)

        K = lambda t: n * t
        A = lambda v: rho(v) / p
        B = lambda v: -c2(v) / p
        C = lambda v: A(v) / v_dot

        best_deltaV = 1e12

        # Extract max and min manoeuvre time
        t_min = int(checkpoint.t_min)
        t_max = int(checkpoint.t_max)

        for dt in xrange(t_min, t_max):
            # Propagate target
            target_prop = target.prop.orekit_prop.propagate(target.prop.date + timedelta(seconds=dt))

            target.abs_state.R = target_prop[0].R
            target.abs_state.V = target_prop[0].V

            # Get new mean orbital elements
            mean_oe_target = target.get_mean_oe()

            v_1 = mean_oe_target.v

            L = lambda v: np.array([
                [np.cos(v) * rho(v), np.sin(v) * rho(v),
                 2.0 / eta ** 2 * (1.0 - 1.5 * c2(v) * K(dt) * rho(v) / eta ** 3), 0.0, 0.0, 0.0],
                [-np.sin(v) * (1.0 + rho(v)), np.cos(v) * (1.0 + rho(v)), -3.0 / eta ** 5 * rho(v) ** 2 * K(dt), 1.0,
                 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, np.cos(v), np.sin(v)],
                [-(np.sin(v) + c2(2.0 * v)), np.cos(v) + e * np.cos(2.0 * v), -3.0 / eta ** 2 * e * (
                    np.sin(v) / rho(v) + 1.0 / eta ** 3 * K(dt) * (np.cos(v) + e * np.cos(2.0 * v))), 0.0, 0.0, 0.0],
                [-(2.0 * np.cos(v) + e * np.cos(2.0 * v)), -(2.0 * np.sin(v) + c2(2.0 * v)),
                 -3.0 / eta ** 2 * (1.0 - e / eta ** 3 * (2.0 * np.sin(v) + c2(2.0 * v)) * K(dt)), 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -np.sin(v), np.cos(v)]
            ])

            M = lambda v: np.array([
                [-3.0 / eta ** 2 * (e + np.cos(v)), 0.0, 0.0, -1.0 / eta ** 2 * np.sin(v) * rho(v),
                 -1.0 / eta ** 2 * (2.0 * np.cos(v) + e + e * np.cos(v) ** 2), 0.0],
                [-3.0 / eta ** 2 * np.sin(v) * (rho(v) + e ** 2) / rho(v), 0.0, 0.0,
                 1.0 / eta ** 2 * (np.cos(v) - 2.0 * e + e * np.cos(v) ** 2),
                 -1.0 / eta ** 2 * np.sin(v) * (1.0 + rho(v)), 0.0],
                [2.0 + 3.0 * e * np.cos(v) + e ** 2, 0.0, 0.0, c2(v) * rho(v), rho(v) ** 2, 0.0],
                [-3.0 / eta ** 2 * (1.0 + rho(v)) * c2(v) / rho(v), 1.0, 0.0,
                 -1.0 / eta ** 2 * (1.0 + rho(v)) * (1.0 - e * np.cos(v)), -1.0 / eta ** 2 * (1.0 + rho(v)) * c2(v),
                 0.0],
                [0.0, 0.0, np.cos(v), 0.0, 0.0, -np.sin(v)],
                [0.0, 0.0, np.sin(v), 0.0, 0.0, np.cos(v)]
            ])

            phi = L(v_1).dot(M(v_0))

            phi_rr = phi[0:3, 0:3]
            phi_rv = phi[0:3, 3:6]
            phi_vr = phi[3:6, 0:3]
            phi_vv = phi[3:6, 3:6]

            phi_rr_t = (A(v_0) * phi_rr + B(v_0) * phi_rv) / A(v_1)
            phi_rv_t = C(v_0) / A(v_1) * phi_rv
            phi_vr_t = (A(v_0) * phi_vr + B(v_0) * phi_vv - B(v_1) * phi_rr_t) / C(v_1)
            phi_vv_t = (C(v_0) * phi_rv - B(v_1) * phi_rv_t) / C(v_1)

            dv_0_transfer = np.linalg.inv(phi_rv_t).dot(checkpoint.rel_state.R - phi_rr_t.dot(chaser.rel_state.R))

            DeltaV_1_LVLH = dv_0_transfer - chaser.rel_state.V
            DeltaV_1_TEME = np.linalg.inv(B_0).dot(DeltaV_1_LVLH)

            DeltaV_2_LVLH = checkpoint.rel_state.V - phi_vr_t.dot(chaser.rel_state.R) + phi_vv_t.dot(dv_0_transfer)
            DeltaV_2_TEME = np.linalg.inv(target.abs_state.get_lof()).dot(DeltaV_2_LVLH)

            deltaV_tot = np.linalg.norm(DeltaV_1_TEME) + np.linalg.norm(DeltaV_2_TEME)

            if deltaV_tot < best_deltaV:
                best_deltaV = deltaV_tot
                best_deltaV_1 = DeltaV_1_TEME
                best_deltaV_2 = DeltaV_2_TEME
                best_dT = dt

            # Depropagate target
            target.abs_state.R = R_T_0
            target.abs_state.V = V_T_0

            target.prop.change_initial_conditions(target.abs_state, target.prop.date, target.mass)

        man1 = self.create_and_apply_manoeuvre(chaser, target, best_deltaV_1, 1e-3)

        man2 = self.create_and_apply_manoeuvre(chaser, target, best_deltaV_2, best_dT)

        return [man1, man2]


class HamelDeLafontaine(OrbitAdjuster):

    def __init__(self):
        pass

    def evaluate_manoeuvre(self):
        pass


class GeneticAlgorithm(OrbitAdjuster):

    def __init__(self):
        pass

    def evaluate_manoeuvre(self):
        pass
