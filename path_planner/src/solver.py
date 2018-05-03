# @copyright Copyright (c) 2017, Davide Frey (frey.davide.ae@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

"""Class holding the definition of the Solver, which outputs a manoeuvre plan given a scenario."""

import numpy as np

from rospace_lib import Cartesian, KepOrbElem, CartesianLVLH, mu_earth
from manoeuvre import Manoeuvre
from state import Satellite, Chaser
from checkpoint import AbsoluteCP, RelativeCP
from scenario import Scenario
from datetime import timedelta, datetime
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
        chaser (Chaser): Chaser actual state, evolving in time according to the solver.
        target (Satellite): Target actual state, evolving in time according to the solver.
        epoch (datetime): Actual epoch, evolving in time according to the solver.
    """

    def __init__(self, date):
        self.manoeuvre_plan = []
        self.scenario = None
        self.chaser = Chaser(date)
        self.target = Satellite(date)
        self.epoch = date

    def initialize_solver(self, scenario):
        """
            Given the scenario to be solved initialize the solver attributes.

        Args:
            scenario (Scenario)
        """

        self.scenario = scenario
        self.epoch = scenario.target_ic.prop.date

        self.chaser.set_from_satellite(scenario.chaser_ic)
        self.target.set_from_satellite(scenario.target_ic)

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
        steps = int(np.floor(idle_time / dt))
        dt_rest = idle_time - dt * steps

        for i in xrange(0, steps):
            # Update epoch
            self.epoch += timedelta(seconds=dt)

            # Propagate
            self.target.prop.orekit_prop.propagate(self.epoch)
            self.chaser.prop.orekit_prop.propagate(self.epoch)

        # Update epoch
        self.epoch += timedelta(seconds=dt_rest)

        # Propagate to the execution epoch
        target_prop = self.target.prop.orekit_prop.propagate(self.epoch)
        chaser_prop = self.chaser.prop.orekit_prop.propagate(self.epoch)

        # Apply impulsive deltaV and apply it to the propagator initial conditions
        chaser_prop[0].V += manoeuvre.deltaV
        self.chaser.prop.change_initial_conditions(chaser_prop[0], self.epoch, self.chaser.mass)

        # Update target and chaser states
        self.chaser.set_abs_state_from_cartesian(chaser_prop[0])
        self.target.set_abs_state_from_cartesian(target_prop[0])

    def create_manoeuvres(self, deltaV_list):
        """
            Given a list of deltaV's and true anomalies where they has to be executed, this function creates and add the
            manoeuvres to the plan, while also applying them to keep the satellite state and propagator up to date.

        Args:
            deltaV_list (list)
        """

        for deltaV in deltaV_list:
            mean_oe = self.chaser.get_mean_oe()

            # Create manoeuvre
            man = Manoeuvre()
            man.deltaV = deltaV[0]
            man.execution_epoch = self.epoch + timedelta(seconds=self.travel_time(mean_oe, mean_oe.v, deltaV[1]))

            # Apply manoeuvre
            self.apply_manoeuvre(man)

            # Add manoeuvre to the plan
            self.manoeuvre_plan.append(man)

    def solve_scenario(self):
        """
            Function that solve the scenario given in the solver object.
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

        print "\n------------------------Chaser initial state------------------------"
        self._print_state(self.chaser)
        print "---------------------------------------------------------------------\n"

        # Start solving scenario by popping positions from position list
        for checkpoint in checkpoints:
            print "\n\n======================================================================="
            print "[GOING TO CHECKPOINT NR. " + str(checkpoint.id) + "]"
            print "======================================================================="
            print "[CHECKPOINT]:"
            self._print_checkpoint(checkpoint)
            print "======================================================================="

            if type(checkpoint) == AbsoluteCP:
                self.absolute_solver(checkpoint)
            elif type(checkpoint) == RelativeCP:
                self.relative_solver(checkpoint)
            else:
                raise TypeError()

            print "======================================================================="
            print "[REACHED STATE]:"
            print "\n--------------------Chaser-------------------"
            self._print_state(self.chaser)

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
             checkpoint (AbsoluteCP): Absolute checkpoint with the state defined as Mean Orbital Elements.
        """

        orbit_adj = PlaneOrientation(self.chaser, checkpoint)
        if orbit_adj.is_necessary():
            man = orbit_adj.evaluate_manoeuvre()
            self.create_manoeuvres(man)

        orbit_adj = ArgumentOfPerigee(self.chaser, checkpoint)
        if orbit_adj.is_necessary():
            man = orbit_adj.evaluate_manoeuvre()
            self.create_manoeuvres(man)

        orbit_adj = HohmannTransfer(self.chaser, checkpoint)
        if orbit_adj.is_necessary():
            man = orbit_adj.evaluate_manoeuvre()
            self.create_manoeuvres(man)

    def relative_solver(self, checkpoint):
        pass

    def _print_result(self):
        """
            Print out results of the simulation and all the manoeuvres.
        """
        tot_dv = 0

        for it, man in enumerate(self.manoeuvre_plan):
            print '\n Manoeuvre nr. ' + str(it) + ':'
            print '--> DeltaV:            ' + str(man.deltaV)
            print '--> Normalized DeltaV: ' + str(np.linalg.norm(man.deltaV))
            tot_dv += np.linalg.norm(man.deltaV)

        return tot_dv, (man.execution_epoch - self.scenario.date).total_seconds()

    @staticmethod
    def _print_state(satellite):
        """
            Print out satellite state.

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

        kep_mean = satellite.get_mean_oe()

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

    @staticmethod
    def _print_checkpoint(checkpoint):
        """
            Print out checkpoint informations.

        Args:
            checkpoint (CheckPoint)
        """

        checkpoint_type = type(checkpoint)

        if checkpoint_type == RelativeCP:
            print " >> Cartesian LVLH: "
            print "      R :      " + str(checkpoint.rel_state.R) + "   [km]"
            print "      V :      " + str(checkpoint.rel_state.V) + "   [km/s]"
            print ""
        elif checkpoint_type == AbsoluteCP:
            print " >> Mean orbital elements: "
            print "      a :      " + str(checkpoint.abs_state.a)
            print "      e :      " + str(checkpoint.abs_state.e)
            print "      i :      " + str(checkpoint.abs_state.i)
            print "      O :      " + str(checkpoint.abs_state.O)
            print "      w :      " + str(checkpoint.abs_state.w)
            print "      v :      " + str(checkpoint.abs_state.v)
        else:
            raise TypeError('CheckPoint type not recognized!')

    @staticmethod
    def travel_time(state, theta0, theta1):
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
