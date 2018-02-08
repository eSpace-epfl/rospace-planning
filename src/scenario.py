import yaml
import numpy as np
import pickle
import sys

from state import Chaser, Target
from checkpoint import RelativeCP, AbsoluteCP

class Scenario(object):

    def __init__(self):
        # Scenario information
        self.name = 'Standard'
        self.overview = ''

        # Checkpoint list
        self.checkpoints = []

        # Chaser and target actual state
        self.chaser = Chaser()
        self.target = Target()

    def import_solved_scenario(self):
        """
            Import a solved scenario, i.e the manoeuvre plan, from pickle file 'scenario.pickle'
        """

        # Actual path
        abs_path = sys.argv[0]
        path_idx = abs_path.find('cso_path_planner')
        abs_path = abs_path[0:path_idx + 16]

        scenario_path = abs_path + '/src/scenario.pickle'

        # Try to import the file
        try:
            with open(scenario_path, 'rb') as file:
                obj = pickle.load(file)
                if obj['scenario_name'] == self.name:
                    print "\n ----> Offline solution loaded! <---- \n"
                    return obj['manoeuvre_plan']
                else:
                    print "Old scenario does not correspond to actual one."
                    sys.exit(1)
        except IOError:
            print "\nScenario file not found."
            sys.exit(1)

    def export_solved_scenario(self, manoeuvre_plan):
        """
            Export a solved scenario into pickle file 'scenario.pickle'
        """

        # Export the "self" into "scenario.p"
        with open('scenario.pickle', 'wb') as file:

            obj = {'scenario_name': self.name, 'manoeuvre_plan': manoeuvre_plan}

            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
            print "Manoeuvre plan saved..."

    def import_yaml_scenario(self):
        """
            Parse scenario from .yaml files in the /cfg folder
        """

        # Actual path
        abs_path = sys.argv[0]
        path_idx = abs_path.find('cso_path_planner')
        abs_path = abs_path[0:path_idx + 16]

        scenario_path = abs_path + '/cfg/scenario.yaml'
        initial_conditions_path = abs_path + '/cfg/initial_conditions.yaml'

        # Opening scenario file
        scenario_file = file(scenario_path, 'r')
        scenario = yaml.load(scenario_file)
        scenario = scenario['scenario']
        checkpoints = scenario['CheckPoints']

        # Opening initial conditions file
        initial_conditions_file = file(initial_conditions_path, 'r')
        initial_conditions = yaml.load(initial_conditions_file)
        chaser_ic = initial_conditions['chaser']
        target_ic = initial_conditions['target']

        # Assign variables
        self.name = scenario['name']
        self.overview = scenario['overview']

        # Assign initial conditions, assuming target in tle and chaser in keplerian
        self.target.set_abs_state_from_tle(target_ic['tle'])
        self.chaser.set_abs_state_from_kep(chaser_ic['kep'], self.target)

        # Extract CheckPoints
        for i in xrange(0, len(checkpoints)):
            pos = checkpoints['S' + str(i)]['position']
            ref_frame = pos.keys()[0]

            prev = self.checkpoints[-1] if len(self.checkpoints) > 0 else None

            if ref_frame == 'lvlh':
                checkpoint = RelativeCP()
                checkpoint.set_rel_state_from_lvlh(pos[ref_frame], self.chaser, self.target)
                checkpoint.error_ellipsoid = checkpoints['S' + str(i)]['error_ellipsoid']
            elif ref_frame == 'kep':
                checkpoint = AbsoluteCP(prev)
                checkpoint.set_abs_state_from_kep(pos[ref_frame], self.chaser, self.target)

            checkpoint.id = checkpoints['S' + str(i)]['id']

            self.checkpoints.append(checkpoint)

