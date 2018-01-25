import yaml
import os
import datetime as dt
import numpy as np
import pickle
import sys

from state import State
from checkpoint import CheckPoint

class Scenario(object):

    def __init__(self):
        # Scenario information
        self.nr_checkpoints = 0
        self.checkpoints = []
        self.name = 'Standard'
        self.overview = ''

        # Chaser and target actual state
        self.chaser = State()
        self.target = State()

        # TDB
        self.mission_start = dt.datetime(2017, 9, 15, 13, 20, 0)
        self.keep_out_zone = 0.05

    def import_solved_scenario(self):
        """
            Import a solved scenario from pickle file 'scenario.p'
        """

        # Try to import the file
        try:
            with open('scenario.pickle', 'rb') as file:
                obj = pickle.load(file)
                if obj['scenario_name'] == self.name:
                    print "\n -----------------> Old manoeuvre elaborated <--------------------"
                    print "Old solution loaded!"
                    return obj['command_line']
                else:
                    print "Old scenario does not correspond to actual one."
                    sys.exit(1)
        except IOError:
            print "\nScenario file not found."
            sys.exit(1)

    def export_solved_scenario(self, command_line):
        """
            Export a solved scenario into pickle file 'scenario.p'
        """

        # TODO: Find a way to store the complete scenario and upload it afterwards
        # TODO: Remove the first command, s.t the scenario can be applied regardless of the initial position

        # Export the "self" into "scenario.p"
        with open('scenario.pickle', 'wb') as file:

            obj = {'scenario_name': self.name, 'command_line': command_line}

            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
            print "Command Line & Scenario saved..."

    def import_yaml_scenario(self):
        """
            Parse scenario from .yaml file.
        """

        # Opening scenario file
        scenario_path = os.getcwd()[:-3] + 'cfg/scenario.yaml'
        scenario_file = file(scenario_path, 'r')
        scenario = yaml.load(scenario_file)
        scenario = scenario['scenario']
        checkpoints = scenario['CheckPoints']

        # Opening initial conditions file
        initial_conditions_path = os.getcwd()[:-3] + 'cfg/initial_conditions.yaml'
        initial_conditions_file = file(initial_conditions_path, 'r')
        initial_conditions = yaml.load(initial_conditions_file)
        chaser_ic = initial_conditions['chaser']
        target_ic = initial_conditions['target']

        # Assign variables
        self.nr_checkpoints = len(checkpoints)
        self.name = scenario['name']
        self.overview = scenario['overview']

        # Assign initial conditions, assuming target in tle and chaser in keplerian
        tle_target = target_ic['tle']
        self.target.kep.from_tle(eval(str(tle_target['i'])),
                                 eval(str(tle_target['O'])),
                                 eval(str(tle_target['e'])),
                                 eval(str(tle_target['m'])),
                                 eval(str(tle_target['w'])),
                                 eval(str(tle_target['n'])))

        kep_chaser = chaser_ic['kep']
        for k in kep_chaser:
            exec('self.chaser.kep.' + k + '=' + str(kep_chaser[k]))

        # Extract CheckPoints
        for i in xrange(0, self.nr_checkpoints):
            checkpoint = CheckPoint()
            checkpoint.id = checkpoints['S' + str(i)]['id']

            try:
                checkpoint.state.from_other_state(self.checkpoints[-1].state)
            except:
                pass

            try:
                checkpoint.error_ellipsoid = checkpoints['S' + str(i)]['error_ellipsoid']
            except:
                pass

            pos = checkpoints['S' + str(i)]['position']
            var_list = []
            for ref_frame in pos:
                for var in pos[ref_frame]:
                    exec('checkpoint.state.' + ref_frame + '.' + var + '= ' + str(pos[ref_frame][var]))
                    var_list.append(ref_frame + '.' + var)

                if ref_frame == 'lvlh':
                    checkpoint.time_dependancy = True

                self.checkpoints.append(checkpoint)

