# @copyright Copyright (c) 2017, Davide Frey (frey.davide.ae@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

"""Class holding the definition of the Solver, which outputs a manoeuvre plan given a scenario."""

import numpy as np

from space_tf import Cartesian, KepOrbElem, CartesianLVLH, mu_earth
from manoeuvre import Manoeuvre
from state import Satellite, Chaser
from checkpoint import CheckPoint
from scenario import Scenario
from datetime import timedelta, datetime
from propagator_class import Propagator
from orbit_adjuster import HohmannTransfer, PlaneOrientation, ArgumentOfPerigee


class Solver(object):
    """
        Base solver class, which takes a predefined scenario and some initial conditions and outputs one possible
        manoeuvre plan that can be executed in order to achieve the final wanted position.

        NOTE: This solver works with optimal solutions for each single manoeuvre, but it has to be noted that the
            combination of all those optimal solutions may be sub-optimal!
            For each manoeuvre it choose to apply the one that consumes the least delta-V within a certain given time
            interval. Therefore, depending on the time interval chosen, the solution may not be the optimal one!

    Attributes:
        manoeuvre_plan (list): List of the manoeuvre that has to be executed to perform the scenario.
        scenario (Scenario): The scenario that has to be solved.
        target (Satellite): Target actual state, evolving in time according to the solver.
        prop_target (Propagator): Target's propagator.
        epoch (datetime): Actual epoch, evolving in time according to the solver.
    """

    def __init__(self):
        self.manoeuvre_plan = []
        self.scenario = None
        self.target = Satellite()
        self.prop_target = Propagator()
        self.epoch = datetime.utcnow()

    def initialize_solver(self, scenario):
        """
            Given the scenario to be solved initialize the solver attributes.

        Args:
            scenario (Scenario)
        """

        self.scenario = scenario
        self.epoch = scenario.date
        self.target.set_from_satellite(scenario.target_ic)
        self.prop_target.initialize_propagator('target', self.target.get_osc_oe(), self.epoch)

    def apply_manoeuvre(self, manoeuvre):
        """
            Given a manoeuvre the satellite is propagated according to it.

        Args:
            manoeuvre (Manoeuvre)
        """

        # Waiting time to get to the proper state in seconds
        idle_time = (manoeuvre.execution_epoch - self.epoch).total_seconds()

        # Divide propagation time in steps of dt seconds to increase accuracy
        dt = 100.0
        steps = int(np.floor(idle_time / 100.0))
        dt_rest = idle_time - 100.0 * steps

        for i in xrange(0, steps):
            # Update epoch
            self.epoch += timedelta(seconds=100.0)

            # Propagate
            self.prop_target.propagator.propagate(self.epoch)

        # Update epoch
        self.epoch += timedelta(seconds=dt_rest)

        # Propagate to the execution epoch
        target_prop = self.prop_target.propagator.propagate(self.epoch)

        # Apply impulsive deltaV and apply it to the propagator initial conditions
        target_prop[0].V += manoeuvre.deltaV
        self.prop_target.change_initial_conditions(target_prop[0], self.epoch, self.target.mass)

        # Update target state
        self.target.set_abs_state_from_cartesian(target_prop[0])

    def create_manoeuvres(self, deltaV_list):

        for deltaV in deltaV_list:
            # Create manoeuvre
            man = Manoeuvre()
            man.deltaV = deltaV[0]
            man.initial_state = self.target.get_mean_oe(self.scenario.prop_type)
            man.execution_epoch = self.epoch + timedelta(seconds=self.travel_time(man.initial_state, man.initial_state.v, deltaV[1]))

            # Apply manoeuvre
            self.apply_manoeuvre(man)

            # Add manoeuvre to the plan
            self.manoeuvre_plan.append(man)

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

        print "\n------------------------Target initial state------------------------"
        self._print_state(self.target)
        print "---------------------------------------------------------------------\n"

        # Start solving scenario by popping positions from position list
        for checkpoint in checkpoints:
            print "\n\n======================================================================="
            print "[GOING TO CHECKPOINT NR. " + str(checkpoint.id) + "]"
            print "======================================================================="
            print "[CHECKPOINT]:"
            self._print_checkpoint(checkpoint)
            print "======================================================================="
            self.absolute_solver(checkpoint)
            print "======================================================================="
            print "[REACHED STATE]:"
            print "\n--------------------Target-------------------"
            self._print_state(self.target)
            print "=======================================================================\n"

        tot_dV, tot_dt = self._print_result()

        print "\n\n-----------------> Manoeuvre elaborated <--------------------\n"
        print "---> Manoeuvre duration:    " + str(tot_dt) + " seconds"
        print "---> Total deltaV:          " + str(tot_dV) + " km/s"

    def absolute_solver(self, checkpoint):
        """
            Absolute solver. Calculate the manoeuvre needed to go from an absolute position to another.

        Args:
             checkpoint (CheckPoint): CheckPoint with the state defined in terms of Mean Orbital Elements.
        """

        orbit_adj = PlaneOrientation(self.target, checkpoint, self.scenario.prop_type)
        if orbit_adj.is_necessary():
            man = orbit_adj.evaluate_manoeuvre()
            self.create_manoeuvres(man)

        orbit_adj = ArgumentOfPerigee(self.target, checkpoint, self.scenario.prop_type)
        if orbit_adj.is_necessary():
            man = orbit_adj.evaluate_manoeuvre()
            self.create_manoeuvres(man)

        orbit_adj = HohmannTransfer(self.target, checkpoint, self.scenario.prop_type)
        if orbit_adj.is_necessary():
            man = orbit_adj.evaluate_manoeuvre()
            self.create_manoeuvres(man)

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

    def _print_state(self, satellite):
        """Print out satellite state.

        Args:
            satellite (Satellite)
        """

        print " >> Cartesian: "
        print "      R :      " + str(satellite.abs_state.R) + "   [km]"
        print "      V :      " + str(satellite.abs_state.V) + "   [km/s]"
        print ""

        kep_osc = satellite.get_osc_oe()

        print " >> Osculating orbital elements: "
        print "      a :      " + str(kep_osc.a)
        print "      e :      " + str(kep_osc.e)
        print "      i :      " + str(kep_osc.i)
        print "      O :      " + str(kep_osc.O)
        print "      w :      " + str(kep_osc.w)
        print "      v :      " + str(kep_osc.v)
        print ""

        kep_mean = satellite.get_mean_oe(self.scenario.prop_type)

        print " >> Mean orbital elements: "
        print "      a :      " + str(kep_mean.a)
        print "      e :      " + str(kep_mean.e)
        print "      i :      " + str(kep_mean.i)
        print "      O :      " + str(kep_mean.O)
        print "      w :      " + str(kep_mean.w)
        print "      v :      " + str(kep_mean.v)

        if hasattr(satellite, 'rel_state'):
            print ""
            print " >> Cartesian LVLH: "
            print "      R :      " + str(satellite.rel_state.R) + "   [km]"
            print "      V :      " + str(satellite.rel_state.V) + "   [km/s]"

    def _print_checkpoint(self, checkpoint):
        """Print out checkpoint informations.

        Args:
            checkpoint (CheckPoint)
        """

        state_type = type(checkpoint.state)

        if state_type == Cartesian:
            print " >> Cartesian: "
            print "      R :      " + str(checkpoint.state.R) + "   [km]"
            print "      V :      " + str(checkpoint.state.V) + "   [km/s]"
            print ""
        elif state_type == KepOrbElem:
            print " >> Mean orbital elements: "
            print "      a :      " + str(checkpoint.state.a)
            print "      e :      " + str(checkpoint.state.e)
            print "      i :      " + str(checkpoint.state.i)
            print "      O :      " + str(checkpoint.state.O)
            print "      w :      " + str(checkpoint.state.w)
            print "      v :      " + str(checkpoint.state.v)
        elif state_type == CartesianLVLH:
            print " >> Cartesian LVLH: "
            print "      R :      " + str(checkpoint.state.R) + "   [km]"
            print "      V :      " + str(checkpoint.state.V) + "   [km/s]"
            print ""
        else:
            raise TypeError('CheckPoint state type not recognized!')

    def _print_result(self):
        tot_dv = 0

        for it, man in enumerate(self.manoeuvre_plan):
            print '\n Manoeuvre nr. ' + str(it) + ':'
            print '--> DeltaV:            ' + str(man.deltaV)
            print '--> Normalized DeltaV: ' + str(np.linalg.norm(man.deltaV))
            tot_dv += np.linalg.norm(man.deltaV)

        return tot_dv, (man.execution_epoch - self.scenario.date).total_seconds()
