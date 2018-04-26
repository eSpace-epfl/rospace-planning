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
from orbit_adjuster import *


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

    def __init__(self):
        self.manoeuvre_plan = []
        self.scenario = None
        self.chaser = Chaser()
        self.target = Satellite()
        self.epoch = datetime.utcnow()

        self.tot_dV = 0.0

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
                self.relative_solver(checkpoint, self.scenario.approach_ellipsoid)
            else:
                raise TypeError()

            print "======================================================================="
            print "[REACHED STATE]:"
            print "\n--------------------Chaser-------------------"
            self._print_state(self.chaser)

            print "\n--------------------Target-------------------"
            self._print_state(self.target)
            print "=======================================================================\n"

        self.tot_dV, tot_dt = self._print_result()

        print "\n\n-----------------> Manoeuvre elaborated <--------------------\n"
        print "---> Manoeuvre duration:    " + str(tot_dt) + " seconds"
        print "---> Total deltaV:          " + str(self.tot_dV) + " km/s"

    def absolute_solver(self, checkpoint):
        """
            Absolute solver. Calculate the manoeuvre needed to go from an absolute position to another.

        Args:
             checkpoint (AbsoluteCP): Absolute checkpoint with the state defined as Mean Orbital Elements.
        """

        orbit_adj = PlaneOrientation()
        if orbit_adj.is_necessary(self.chaser, checkpoint):
            self.manoeuvre_plan += orbit_adj.evaluate_manoeuvre(self.chaser, checkpoint, self.target)

        orbit_adj = ArgumentOfPerigee()
        if orbit_adj.is_necessary(self.chaser, checkpoint):
            self.manoeuvre_plan += orbit_adj.evaluate_manoeuvre(self.chaser, checkpoint, self.target)

        orbit_adj = HohmannTransfer()
        if orbit_adj.is_necessary(self.chaser, checkpoint):
            self.manoeuvre_plan += orbit_adj.evaluate_manoeuvre(self.chaser, checkpoint, self.target)

    def relative_solver(self, checkpoint, approach_ellipsoid):
        """
            Relative solver. Calculate the manoeuvre needed to go from a relative position to another.

        Args:
            checkpoint (RelativeCP)
            approach_ellipsoid (np.array): Allowed error on a checkpoint.
        """

        # Mean orbital elements
        chaser_mean = self.chaser.get_mean_oe()
        target_mean = self.target.get_mean_oe()

        # Check if plane needs to be corrected again
        # TODO: Put as tolerance a number slightly bigger than the deviation of the estimation
        # TODO: Remove changes in plane if it is drifting autonomously to the wanted direction
        tol_i = 1.0 / chaser_mean.a
        tol_O = 1.0 / chaser_mean.a

        # At this point, inclination and raan should match the one of the target
        di = target_mean.i - chaser_mean.i
        dO = target_mean.O - chaser_mean.O
        if abs(di) > tol_i or abs(dO) > tol_O:
            checkpoint_abs = AbsoluteCP()
            checkpoint_abs.abs_state.i = target_mean.i
            checkpoint_abs.abs_state.O = target_mean.O

            orbit_adj = PlaneOrientation()
            orbit_adj.evaluate_manoeuvre(self.chaser, checkpoint_abs, self.target)

        if checkpoint.manoeuvre_type == 'standard':
            orbit_adj = MultiLambert()
            self.manoeuvre_plan += orbit_adj.evaluate_manoeuvre(self.chaser, checkpoint, self.target, approach_ellipsoid, True)
            # self.linearized_including_J2(checkpoint, approach_ellipsoid)

        elif checkpoint.manoeuvre_type == 'radial':
            # Manoeuvre type is radial -> deltaT is calculated from CW-equations -> solved with multi-lambert
            dt = np.pi / np.sqrt(mu_earth / target_mean.a ** 3.0)

            checkpoint.t_min = dt
            checkpoint.t_max = dt + 1.0

            orbit_adj = MultiLambert()
            self.manoeuvre_plan += orbit_adj.evaluate_manoeuvre(self.chaser, checkpoint, self.target, approach_ellipsoid, True)
            # self.linearized_including_J2(checkpoint, approach_ellipsoid)

        elif checkpoint.manoeuvre_type == 'drift':
            orbit_adj = Drift()
            orbit_adj.evaluate_manoeuvre(self.chaser, checkpoint, self.target, self.manoeuvre_plan)

    def _print_state(self, satellite):
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

    def _print_checkpoint(self, checkpoint):
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

    def _print_result(self):
        """
            Print out results of the simulation and all the manoeuvres.
        """
        tot_dv = 0

        for it, man in enumerate(self.manoeuvre_plan):
            print '\n Manoeuvre nr. ' + str(it) + ':'
            print '>>>   DeltaV:            ' + str(man.deltaV)
            print '>>>   Normalized DeltaV: ' + str(np.linalg.norm(man.deltaV))
            print '>>>   Execution Epoch:   ' + str(man.execution_epoch)
            tot_dv += np.linalg.norm(man.deltaV)

        return tot_dv, (man.execution_epoch - self.scenario.date).total_seconds()
