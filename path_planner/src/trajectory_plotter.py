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

from rospace_lib import CartesianTEME
from scenario import Scenario
from state import Chaser, Satellite
from datetime import timedelta
from path_planning_propagator import Propagator

from org.orekit.propagation import SpacecraftState
from org.orekit.frames import FramesFactory
from org.orekit.orbits import CartesianOrbit
from org.orekit.utils import PVCoordinates
from org.orekit.utils import Constants as Cst
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.time import AbsoluteDate, TimeScalesFactory

def print_state(kep):
    print "      a :     " + str(kep.a)
    print "      e :     " + str(kep.e)
    print "      i :     " + str(kep.i)
    print "      O :     " + str(kep.O)
    print "      w :     " + str(kep.w)
    print "      v :     " + str(kep.v)

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
    # chaser_extra = Chaser()

    chaser.set_from_satellite(scenario.chaser_ic)
    target.set_from_satellite(scenario.target_ic)

    chaser_cart_extra = CartesianTEME()
    target_cart_extra = CartesianTEME()

    epoch = scenario.date

    # extra_propagation = 0

    # Set propagators
    chaser.prop.initialize_propagator('chaser', chaser.get_osc_oe(), 'real-world', epoch)
    target.prop.initialize_propagator('target', target.get_osc_oe(), 'real-world', epoch)

    print "--------------------Chaser initial state-------------------"
    print " >> Osc Elements:"
    print_state(chaser.get_osc_oe())
    print "\n >> Mean Elements:"
    print_state(chaser.get_mean_oe())
    print "\n >> LVLH:"
    print "     R: " + str(chaser.rel_state.R)
    print "     V: " + str(chaser.rel_state.V)
    print "\n--------------------Target initial state-------------------"
    print " >> Osc Elements:"
    print_state(target.get_osc_oe())
    print "\n >> Mean Elements:"
    print_state(target.get_mean_oe())
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

        duration = (man.execution_epoch - epoch).total_seconds()

        for j in xrange(0, int(np.floor(duration)), 100):
            chaser_prop = chaser.prop.orekit_prop.propagate(epoch + timedelta(seconds=j))
            target_prop = target.prop.orekit_prop.propagate(epoch + timedelta(seconds=j))

            chaser.rel_state.from_cartesian_pair(chaser_prop[0], target_prop[0])

            R_chaser.append(chaser_prop[0].R)
            R_target.append(target_prop[0].R)
            R_chaser_lvlh.append(chaser.rel_state.R)

        # Re-initialize propagators and update epoch
        epoch += timedelta(seconds=duration)

        chaser_prop = chaser.prop.orekit_prop.propagate(epoch)
        target_prop = target.prop.orekit_prop.propagate(epoch)

        chaser.rel_state.from_cartesian_pair(chaser_prop[0], target_prop[0])

        chaser_prop[0].V += man.deltaV

        chaser.prop.change_initial_conditions(chaser_prop[0], epoch, chaser.mass)
        target.prop.change_initial_conditions(target_prop[0], epoch, target.mass)

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
                           'deltaV': man.deltaV, 'duration': duration}) #, 'rel_state_c_extra': R_chaser_lvlh_extra})

    print "\nManoeuvre saved."

def main(filename):
    # Create scenario
    scenario = Scenario()
    scenario.import_yaml_scenario(filename)

    # Import scenario solution
    manoeuvre_plan = scenario.import_solved_scenario()

    # scenario.prop_chaser._propagator_num.resetAtEnd = False
    # scenario.prop_target._propagator_num.resetAtEnd = False

    # Path where the files should be stored
    save_path = '/home/dfrey/polybox/manoeuvre'

    plot_result(manoeuvre_plan, scenario, save_path)

if __name__ == "__main__":
    main('scenario')
