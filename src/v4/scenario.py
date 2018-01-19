import yaml
import datetime as dt
import numpy as np
import pickle
import sys
import rospy
import scipy.io as sio

from state import State
from space_tf import Cartesian, CartesianLVLH, KepOrbElem, mu_earth, QNSRelOrbElements, CartesianLVC

class Scenario:

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

        scenario_file = file('scenario/scenario.yaml', 'r')
        scenario = yaml.load(scenario_file)
        scenario = scenario['scenario']

        # Extract from yaml
        checkpoints = scenario['CheckPoints']
        initial_conditions = scenario['InitialConditions']

        # Assign variables
        self.nr_checkpoints = len(checkpoints)
        self.name = scenario['name']
        self.overview = scenario['overview']

        # Assign initial conditions
        tle_target = initial_conditions['tle_target']
        self.target.kep.from_tle(tle_target['i'], tle_target['O'], tle_target['e'], tle_target['m'], tle_target['w'], tle_target['n'])

        kep_chaser = initial_conditions['kep_chaser']
        for k in kep_chaser:
            exec('self.chaser.kep.' + k + '=' + str(kep_chaser[k]))

        chaser_next = chaser

        # Extract CheckPoints
        for i in xrange(0, self.nr_checkpoints):
            S = CheckPoint()
            S.id = checkpoints['S' + str(i)]['id']

            try:
                S.error_ellipsoid = checkpoints['S' + str(i)]['error_ellipsoid']
            except:
                pass

            pos = checkpoints['S' + str(i)]['position']
            var_list = []
            for ref_frame in pos:
                for var in pos[ref_frame]:
                    exec('S.position.' + ref_frame + '.' + var +
                         '= ' + str(pos[ref_frame][var]))
                    var_list.append(ref_frame + '.' + var)
                self.checkpoints.append(S)
            S.generate_free_coordinates(var_list, chaser_next, target)
            chaser_next = S.position
