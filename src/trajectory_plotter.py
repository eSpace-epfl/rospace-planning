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

    chaser.set_from_other_satellite(scenario.chaser_ic)
    target.set_from_other_satellite(scenario.target_ic)

    chaser_cart = Cartesian()
    target_cart = Cartesian()

    chaser_cart_extra = Cartesian()
    target_cart_extra = Cartesian()

    chaser_cart.from_keporb(chaser.abs_state)
    target_cart.from_keporb(target.abs_state)

    epoch = scenario.date

    extra_propagation = 0

    L = len(manoeuvre_plan)
    for i in xrange(0, L):
        print " --> Simulating manoeuvre " + str(i)

        # Creating list of radius of target and chaser
        R_target = [target_cart.R]
        R_chaser = [chaser_cart.R]
        R_chaser_lvlh = [chaser.rel_state.R]

        man = manoeuvre_plan[i]

        r_C = chaser_cart.R
        v_C = chaser_cart.V
        r_T = target_cart.R
        v_T = target_cart.V

        for j in xrange(0, int(np.floor(man.duration))):
            chaser_prop = scenario.prop_chaser.propagate(epoch + timedelta(seconds=j+1))
            target_prop = scenario.prop_target.propagate(epoch + timedelta(seconds=j+1))

            chaser_cart.R = chaser_prop[0].R
            chaser_cart.V = chaser_prop[0].V
            target_cart.R = target_prop[0].R
            target_cart.V = target_prop[0].V

            chaser.rel_state.from_cartesian_pair(chaser_cart, target_cart)

            R_chaser.append(chaser_cart.R)
            R_target.append(target_cart.R)
            R_chaser_lvlh.append(chaser.rel_state.R)

        chaser_prop = scenario.prop_chaser.propagate(epoch + timedelta(seconds=man.duration))
        target_prop = scenario.prop_target.propagate(epoch + timedelta(seconds=man.duration))

        chaser_cart.R = chaser_prop[0].R
        chaser_cart.V = chaser_prop[0].V
        target_cart.R = target_prop[0].R
        target_cart.V = target_prop[0].V

        chaser.rel_state.from_cartesian_pair(chaser_cart, target_cart)

        # Re-initialize propagators and update epoch
        epoch += timedelta(seconds=man.duration)
        chaser_cart.V += man.dV

        # Osculating orbital elements
        chaser_new_ic = KepOrbElem()
        target_new_ic = KepOrbElem()
        chaser_new_ic.from_cartesian(chaser_cart)
        target_new_ic.from_cartesian(target_cart)

        scenario.initialize_propagators(chaser_new_ic, target_new_ic, epoch)


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
        R_chaser_lvlh_extra = [chaser_extra.rel_state.R]
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
                           'deltaV': man.dV, 'duration': man.duration, 'rel_state_c_extra': R_chaser_lvlh_extra})

    print "\nManoeuvre saved."

def main(prop):
    # Import scenario and initial conditions
    scenario = Scenario()
    scenario.set_propagator_type(prop)
    scenario.import_yaml_scenario()

    # Import scenario solution
    manoeuvre_plan = scenario.import_solved_scenario()

    # Add lockers to manoeuvre plan
    for man in manoeuvre_plan:
        man.add_lock()

    # Path where the files should be stored
    save_path = '/home/dfrey/polybox/manoeuvre'

    plot_result(manoeuvre_plan, scenario, save_path)

if __name__ == "__main__":
    # prop = '2-body'
    prop = 'real-world'
    main(prop)
