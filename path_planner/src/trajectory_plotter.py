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
import argparse

from scenario import Scenario
from state import Chaser, Satellite
from datetime import timedelta


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

    print "\n[INFO]: Simulating manoeuvre plan in folder /test_" + str(last + 1) + " ..."

    # Simulating the whole manoeuvre and store the result
    chaser = Chaser()
    target = Satellite()

    target.initialize_satellite('target', scenario.ic_name, scenario.prop_type)
    chaser.initialize_satellite('chaser', scenario.ic_name, scenario.prop_type, target)

    epoch = scenario.date

    if extra_dt > 0.0:
        chaser_extra = Chaser()

    print "\n------------------------Chaser initial state-----------------------"
    print " >> Osc Elements:"
    print_state(chaser.get_osc_oe())
    print "\n >> Mean Elements:"
    print_state(chaser.get_mean_oe())
    print "\n >> LVLH:"
    print "     R: " + str(chaser.rel_state.R)
    print "     V: " + str(chaser.rel_state.V)
    print "\n------------------------Target initial state-----------------------"
    print " >> Osc Elements:"
    print_state(target.get_osc_oe())
    print "\n >> Mean Elements:"
    print_state(target.get_mean_oe())
    print "--------------------------------------------------------------------\n"

    L = len(manoeuvre_plan)
    for i in xrange(0, L):
        # Creating list of radius of target and chaser
        R_target = []
        R_chaser = []
        R_chaser_lvlh = []

        man = manoeuvre_plan[i]

        duration = (man.execution_epoch - epoch).total_seconds()

        print "\n[INFO]: Simulating manoeuvre " + str(i)
        print "        Starting pos:   " + str(chaser.rel_state.R)
        print "        Starting epoch: " + str(man.execution_epoch)
        print "        Duration:       " + str(duration)

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

        print "        End pos: " + str(chaser.rel_state.R)

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

    print "\n[INFO]: Manoeuvres saved in folder nr. " + str(last + 1) + "."


def main(manoeuvre_plan=None, scenario=None, filename=None, save_path='/home/dfrey/polybox/manoeuvre', extra_dt=20000):

    if scenario is None and filename is not None:
        # Create scenario
        scenario = Scenario()

        # Import scenario solution
        manoeuvre_plan = scenario.import_solved_scenario(filename)
    elif scenario is None and filename is None:
        raise IOError('File name needed as input!')

    if manoeuvre_plan is not None:
        plot_result(manoeuvre_plan, scenario, save_path, extra_dt)
    else:
        raise IOError('Manoeuvre plan needed as input!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filename',
                        help='Name of the scenario pickle file which should be uploaded.')
    parser.add_argument('--extra_dt',
                        help='State how many seconds of extra propagation are needed (standard 20000 seconds).',
                        type=float)
    parser.add_argument('--save_path',
                        help='Specify a path where the manoeuvres should be saved.')
    args = parser.parse_args()

    if args.extra_dt:
        print "TYPE: " + str(type(args.extra_dt))
        extra_dt = args.extra_dt
    else:
        extra_dt = 0.0

    if args.save_path:
        save_path = args.save_path
    else:
        save_path = '/home/dfrey/polybox/manoeuvre'

    main(filename=args.filename, extra_dt=extra_dt, save_path=save_path)
