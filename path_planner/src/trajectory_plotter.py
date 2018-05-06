# @copyright Copyright (c) 2017, Davide Frey (frey.davide.ae@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

"""Main file to plot a trajectory given the manoeuvre plan."""

import numpy as np
import scipy.io as sio
import os

from rospace_lib import CartesianTEME
from scenario import Scenario
from state import Chaser, Satellite
from datetime import timedelta, datetime

def print_state(kep):
    print "      a :     " + str(kep.a)
    print "      e :     " + str(kep.e)
    print "      i :     " + str(kep.i)
    print "      O :     " + str(kep.O)
    print "      w :     " + str(kep.w)
    print "      v :     " + str(kep.v)

def plot_result(manoeuvre_plan, scenario, save_path, extra_dt=0.0):

    dir_list = os.listdir(save_path)
    last = -1
    for el in dir_list:
        if 'test_' in el and os.path.isdir(save_path + '/' + el):
            last = int(el[5:]) if int(el[5:]) > last else last

    os.makedirs(save_path + '/test_' + str(last + 1))

    print "Simulating manoeuvre plan in folder /test_" + str(last + 1) + " ..."

    # Simulating the whole manoeuvre and store the result
    chaser = Chaser(scenario.date)
    target = Satellite(scenario.date)

    chaser.set_from_satellite(scenario.chaser_ic)
    target.set_from_satellite(scenario.target_ic)

    epoch = scenario.date

    # Set propagators
    chaser.prop.initialize_propagator('chaser', chaser.get_osc_oe(), '2-body')
    target.prop.initialize_propagator('target', target.get_osc_oe(), '2-body')

    if extra_dt > 0.0:
        chaser_extra = Chaser(scenario.date)

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
        print duration

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

        R_chaser.append(chaser_prop[0].R)
        R_target.append(target_prop[0].R)
        R_chaser_lvlh.append(chaser.rel_state.R)

        print "     End pos: " + str(chaser.rel_state.R)

        # EXTRA PROPAGATION TO CHECK TRAJECTORY SAFETY
        if extra_dt > 0.0 and duration > 1.0:
            chaser_extra.rel_state.from_cartesian_pair(chaser_prop[0], target_prop[0])

            R_chaser_lvlh_extra = [chaser_extra.rel_state.R]

            for j in xrange(0, int(np.floor(extra_dt)), 100):
                chaser_prop_extra = chaser.prop.orekit_prop.propagate(epoch + timedelta(seconds=j))
                target_prop_extra = target.prop.orekit_prop.propagate(epoch + timedelta(seconds=j))

                chaser_extra.rel_state.from_cartesian_pair(chaser_prop_extra[0], target_prop_extra[0])
                R_chaser_lvlh_extra.append(chaser_extra.rel_state.R)

            chaser_prop_extra = chaser.prop.orekit_prop.propagate(epoch + timedelta(seconds=extra_dt))
            target_prop_extra = target.prop.orekit_prop.propagate(epoch + timedelta(seconds=extra_dt))

            chaser_extra.rel_state.from_cartesian_pair(chaser_prop_extra[0], target_prop_extra[0])
            R_chaser_lvlh_extra.append(chaser_extra.rel_state.R)

            chaser.prop.change_initial_conditions(chaser_prop[0], epoch, chaser.mass)
            target.prop.change_initial_conditions(target_prop[0], epoch, target.mass)
        else:
            R_chaser_lvlh_extra = []

        chaser_prop[0].V += man.deltaV

        chaser.prop.change_initial_conditions(chaser_prop[0], epoch, chaser.mass)
        target.prop.change_initial_conditions(target_prop[0], epoch, target.mass)

        # EXTRA PROPAGATION TO CHECK TRAJECTORY SAFETY AFTER LAST MANOEUVRE
        if extra_dt > 0.0 and i == L - 1:
            chaser_extra.rel_state.from_cartesian_pair(chaser_prop[0], target_prop[0])

            R_chaser_lvlh_extra = [chaser_extra.rel_state.R]

            for j in xrange(0, int(np.floor(extra_dt)), 100):
                chaser_prop_extra = chaser.prop.orekit_prop.propagate(epoch + timedelta(seconds=j))
                target_prop_extra = target.prop.orekit_prop.propagate(epoch + timedelta(seconds=j))

                chaser_extra.rel_state.from_cartesian_pair(chaser_prop_extra[0], target_prop_extra[0])
                R_chaser_lvlh_extra.append(chaser_extra.rel_state.R)

            chaser_prop_extra = chaser.prop.orekit_prop.propagate(epoch + timedelta(seconds=extra_dt))
            target_prop_extra = target.prop.orekit_prop.propagate(epoch + timedelta(seconds=extra_dt))

            chaser_extra.rel_state.from_cartesian_pair(chaser_prop_extra[0], target_prop_extra[0])
            R_chaser_lvlh_extra.append(chaser_extra.rel_state.R)

            chaser.prop.change_initial_conditions(chaser_prop[0], epoch, chaser.mass)
            target.prop.change_initial_conditions(target_prop[0], epoch, target.mass)

        # Saving in .mat file
        sio.savemat(save_path + '/test_' + str(last + 1) + '/manoeuvre_' + str(i) + '.mat',
                    mdict={'abs_state_c': R_chaser, 'rel_state_c': R_chaser_lvlh, 'abs_state_t': R_target,
                           'deltaV': man.deltaV, 'duration': duration, 'rel_state_c_extra': R_chaser_lvlh_extra})

    print "\nManoeuvre saved in folder " + str(last + 1) + "."

def main(filename, scenario, manoeuvre_plan):
    # Create scenario
    # scenario = Scenario(datetime(2014, 7, 16, 6, 58, 50, 646144))
    # scenario.import_yaml_scenario(filename)

    # Import scenario solution
    # manoeuvre_plan = scenario.import_solved_scenario()

    # Path where the files should be stored
    save_path = '/home/dfrey/polybox/manoeuvre'

    plot_result(manoeuvre_plan, scenario, save_path, 20000)

if __name__ == "__main__":
    main('scenario_camille')
