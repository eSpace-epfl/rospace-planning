# @copyright Copyright (c) 2017, Davide Frey (frey.davide.ae@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

"""Main file to plot a trajectory given the manoeuvre plan."""

import numpy as np
import pykep as pk
import scipy.io as sio
import os

from scenario import Scenario
from state import Chaser, Satellite
from space_tf import Cartesian, mu_earth, KepOrbElem
from datetime import timedelta

from org.orekit.propagation import SpacecraftState
from org.orekit.frames import FramesFactory
from org.orekit.orbits import CartesianOrbit
from org.orekit.utils import PVCoordinates
from org.orekit.utils import Constants as Cst
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.time import AbsoluteDate, TimeScalesFactory


def _change_propagator_ic(propagator, initial_state, epoch, mass):
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

def _print_state(satellite):
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

    kep_mean = satellite.get_mean_oe('real-world')

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

def plot_result(manoeuvre_plan, scenario, save_path):

    dir_list = os.listdir(save_path)
    last = -1
    for el in dir_list:
        if 'test_' in el and os.path.isdir(save_path + '/' + el):
            last = int(el[5:]) if int(el[5:]) > last else last

    os.makedirs(save_path + '/test_' + str(last + 1))

    print "Simulating manoeuvre plan in folder /test_" + str(last + 1) + " ..."

    # Simulating the whole manoeuvre and store the result
    chaser = Chaser()
    target = Satellite()
    chaser_extra = Chaser()

    chaser.set_from_satellite(scenario.chaser_ic)
    target.set_from_satellite(scenario.target_ic)

    chaser_cart_extra = Cartesian()
    target_cart_extra = Cartesian()

    epoch = scenario.date

    extra_propagation = 0

    print "--------------------Chaser initial state-------------------"
    _print_state(chaser)
    print "\n--------------------Target initial state-------------------"
    _print_state(target)
    print "------------------------------------------------------------\n"

    L = len(manoeuvre_plan)
    for i in xrange(0, L):
        print " --> Simulating manoeuvre " + str(i)
        print "     Start pos: " + str(chaser.rel_state.R)

        # Creating list of radius of target and chaser
        R_target = []
        R_chaser = []
        R_chaser_lvlh = []

        man = manoeuvre_plan[i]

        dt = 100.0
        step = man.duration / dt
        t_d = man.duration - np.floor(step) * dt
        dt_tot = 0

        for j in xrange(0, int(np.floor(step))):
            chaser_prop = scenario.prop_chaser.propagate(epoch)
            target_prop = scenario.prop_target.propagate(epoch)

            epoch += timedelta(seconds=100.0)
            dt_tot += 100.0

            chaser.abs_state.R = chaser_prop[0].R
            chaser.abs_state.V = chaser_prop[0].V

            target.abs_state.R = target_prop[0].R
            target.abs_state.V = target_prop[0].V

            chaser.rel_state.from_cartesian_pair(chaser.abs_state, target.abs_state)

            R_chaser.append(chaser.abs_state.R)
            R_target.append(target.abs_state.R)
            R_chaser_lvlh.append(chaser.rel_state.R)

        # Re-initialize propagators and update epoch
        epoch += timedelta(seconds=t_d)

        chaser_prop = scenario.prop_chaser.propagate(epoch)
        target_prop = scenario.prop_target.propagate(epoch)

        chaser.abs_state.R = chaser_prop[0].R
        chaser.abs_state.V = chaser_prop[0].V

        target.abs_state.R = target_prop[0].R
        target.abs_state.V = target_prop[0].V
        chaser.abs_state.V += man.dV

        chaser.rel_state.from_cartesian_pair(chaser.abs_state, target.abs_state)

        _change_propagator_ic(scenario.prop_chaser, chaser.abs_state, epoch, chaser.mass)
        _change_propagator_ic(scenario.prop_target, target.abs_state, epoch, target.mass)

        print "     End pos: " + str(chaser.rel_state.R)

        # EXTRA PROPAGATION TO CHECK TRAJECTORY SAFETY

        # r_C_extra = chaser_cart.R
        # v_C_extra = chaser_cart.V
        # r_T_extra = target_cart.R
        # v_T_extra = target_cart.V
        #
        # chaser_cart_extra.R = r_C_extra
        # chaser_cart_extra.V = v_C_extra
        #
        # target_cart_extra.R = r_T_extra
        # target_cart_extra.V = v_T_extra
        #
        # chaser_extra.rel_state.from_cartesian_pair(chaser_cart_extra, target_cart_extra)
        #
        # R_chaser_lvlh_extra = [chaser_extra.rel_state.R]
        #
        # for j in xrange(0, extra_propagation):
        #     r_C_extra, v_C_extra = pk.propagate_lagrangian(r_C_extra, v_C_extra, 1.0, mu_earth)
        #     chaser_cart_extra.R = np.array(r_C_extra)
        #     chaser_cart_extra.V = np.array(v_C_extra)
        #
        #     r_T_extra, v_T_extra = pk.propagate_lagrangian(r_T_extra, v_T_extra, 1.0, mu_earth)
        #     target_cart_extra.R = np.array(r_T_extra)
        #     target_cart_extra.V = np.array(v_T_extra)
        #
        #     chaser_extra.rel_state.from_cartesian_pair(chaser_cart_extra, target_cart_extra)
        #
        #     R_chaser_lvlh_extra.append(chaser_extra.rel_state.R)
        #
        #     r_C_extra = np.array(r_C_extra)
        #     v_C_extra = np.array(v_C_extra)
        #     r_T_extra = np.array(r_T_extra)
        #     v_T_extra = np.array(v_T_extra)

        # Saving in .mat file
        sio.savemat(save_path + '/test_' + str(last + 1) + '/manoeuvre_' + str(i) + '.mat',
                    mdict={'abs_state_c': R_chaser, 'rel_state_c': R_chaser_lvlh, 'abs_state_t': R_target,
                           'deltaV': man.dV, 'duration': man.duration}) #, 'rel_state_c_extra': R_chaser_lvlh_extra})

    print "\nManoeuvre saved."

def main():
    # Create scenario
    scenario = Scenario()
    scenario.import_yaml_scenario()

    # Import scenario solution
    manoeuvre_plan = scenario.import_solved_scenario()

    # Path where the files should be stored
    save_path = '/home/dfrey/polybox/manoeuvre'

    plot_result(manoeuvre_plan, scenario, save_path)

if __name__ == "__main__":
    main()
