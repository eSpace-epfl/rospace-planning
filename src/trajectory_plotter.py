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

from scenario import Scenario
from state import Chaser, Satellite
from space_tf import Cartesian, mu_earth

def main():
    # Import scenario and initial conditions
    scenario = Scenario()
    scenario.import_yaml_scenario()

    # Import scenario solution
    manoeuvre_plan = scenario.import_solved_scenario()

    plot_result(manoeuvre_plan, scenario)

def plot_result(manoeuvre_plan, scenario):
    print "Simulating manoeuvre plan... "

    # Simulating the whole manoeuvre and store the result
    chaser = Chaser()
    target = Satellite()

    chaser.set_from_other_satellite(scenario.chaser_ic)
    target.set_from_other_satellite(scenario.target_ic)

    chaser_cart = Cartesian()
    target_cart = Cartesian()

    chaser_cart.from_keporb(chaser.abs_state)
    target_cart.from_keporb(target.abs_state)

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
            r_C, v_C = pk.propagate_lagrangian(r_C, v_C, 1.0, mu_earth)
            chaser_cart.R = np.array(r_C)
            chaser_cart.V = np.array(v_C)

            r_T, v_T = pk.propagate_lagrangian(r_T, v_T, 1.0, mu_earth)
            target_cart.R = np.array(r_T)
            target_cart.V = np.array(v_T)

            chaser.rel_state.from_cartesian_pair(chaser_cart, target_cart)

            R_chaser.append(chaser_cart.R)
            R_target.append(target_cart.R)
            R_chaser_lvlh.append(chaser.rel_state.R)

        r_C, v_C = pk.propagate_lagrangian(r_C, v_C, man.duration - np.floor(man.duration), mu_earth)
        chaser_cart.R = np.array(r_C)
        chaser_cart.V = np.array(v_C)

        r_T, v_T = pk.propagate_lagrangian(r_T, v_T, man.duration - np.floor(man.duration), mu_earth)
        target_cart.R = np.array(r_T)
        target_cart.V = np.array(v_T)

        chaser.rel_state.from_cartesian_pair(chaser_cart, target_cart)

        # Apply dV
        r_C, v_C = pk.propagate_lagrangian(r_C, v_C + man.dV, 1e-3, mu_earth)
        chaser_cart.R = np.array(r_C)
        chaser_cart.V = np.array(v_C)

        r_T, v_T = pk.propagate_lagrangian(r_T, v_T, 1e-3, mu_earth)
        target_cart.R = np.array(r_T)
        target_cart.V = np.array(v_T)

        # Saving in .mat file
        sio.savemat('/home/dfrey/polybox/manoeuvre/manoeuvre_' + str(i) + '.mat',
                    mdict={'abs_state_c': R_chaser, 'rel_state_c': R_chaser_lvlh, 'abs_state_t': R_target,
                           'deltaV': man.dV})

    print "\nManoeuvre saved."

if __name__ == "__main__":
    main()
