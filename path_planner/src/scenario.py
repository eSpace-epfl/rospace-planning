# @copyright Copyright (c) 2017, Davide Frey (frey.davide.ae@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

"""Class containing the scenario definition."""

import yaml
import numpy as np
import pickle
import sys

from state import Satellite
from checkpoint import CheckPoint
from datetime import datetime


class Scenario(object):
    """
        Base class defining a scenario.
        In there the chaser and target initial conditions are stored, as well as the checkpoints of the missions and
        the some missions constraints.

        Attributes:
            name (str): Scenario name.
            overview (str): Brief scenario overview, explaining all the steps.
            checkpoints (list): A list containing all the checkpoints that has to be executed in the right order.
            target_ic (Satellite): Target initial state.
            date (timedelta): Time at which the simulation start.
            prop_type (str): Propagator type that has to be used. It can be either 'real-world' - with all the
                disturbances, or '2-body' - considering only the exact solution to the 2-body equation.
            approach_ellipsoid (np.array): Axis of the approach ellipsoid around the target [km].
            koz_r (float64): Radius of Keep-Out Zone drawn around the target [km].
    """

    def __init__(self):
        # Scenario information
        self.name = 'Standard'
        self.overview = ''

        # Checkpoint list
        self.checkpoints = []

        # Satellite initial states
        self.target_ic = Satellite()

        # Target Keep-Out Zones
        self.approach_ellipsoid = np.array([0.0, 0.0, 0.0])
        self.koz_r = 0.0

    def import_solved_scenario(self):
        """
            Import a solved scenario, i.e the manoeuvre plan, from pickle file 'scenario.pickle'

            Return:
                Manoeuvre plan, the list containing the manoeuvres to perform this scenario.
        """

        # Actual path
        abs_path = sys.argv[0]
        path_idx = abs_path.find('cso_path_planner')
        abs_path = abs_path[0:path_idx + 16]

        scenario_path = abs_path + '/example/scenario.pickle'

        # Try to import the file
        try:
            with open(scenario_path, 'rb') as file:
                obj = pickle.load(file)
                if obj['scenario_name'] == self.name:
                    print "\n ----> Offline solution loaded! <---- \n"

                    self.checkpoints = obj['checkpoints']
                    self.name = obj['scenario_name']
                    self.target_ic = obj['target_ic']
                    self.date = obj['scenario_epoch']
                    self.prop_type = obj['prop_type']

                    return obj['manoeuvre_plan']
                else:
                    print "[WARNING]: Scenario in cfg folder does not correspond to actual one."
                    sys.exit(1)
        except IOError:
            print "\n[WARNING]: Scenario file not found."
            sys.exit(1)

    def export_solved_scenario(self, manoeuvre_plan):
        """
            Export a solved scenario into pickle file 'scenario.pickle' in the /example folder.
        """

        # Actual path
        abs_path = sys.argv[0]
        path_idx = abs_path.find('path_planner')
        abs_path = abs_path[0:path_idx]

        pickle_path = abs_path + 'path_planner/example/scenario.pickle'

        with open(pickle_path, 'wb') as file:

            obj = {'scenario_name': self.name, 'checkpoints': self.checkpoints, 'manoeuvre_plan': manoeuvre_plan,
                   'target_ic': self.target_ic, 'scenario_epoch': self.date,
                   'prop_type': self.prop_type}

            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

            print "Manoeuvre plan saved..."

    def import_yaml_scenario(self):
        """
            Parse scenario and import initial conditions from .yaml files in the /cfg folder.
        """

        # Actual path
        abs_path = sys.argv[0]
        path_idx = abs_path.find('path_planner')
        abs_path = abs_path[0:path_idx]

        # Opening scenario file
        scenario_path = abs_path + 'path_planner/cfg/scenario.yaml'
        scenario_file = file(scenario_path, 'r')
        scenario = yaml.load(scenario_file)
        scenario = scenario['scenario']
        checkpoints = scenario['CheckPoints']

        # Assign variables
        self.name = scenario['name']
        self.overview = scenario['overview']
        self.prop_type = scenario['prop_type']

        # Initialize satellites
        self.target_ic.initialize_satellite('target')

        # Extract CheckPoints
        for i in xrange(0, len(checkpoints)):
            pos = checkpoints['S' + str(i)]['position']
            prev = self.checkpoints[-1] if len(self.checkpoints) > 0 else None

            checkpoint = CheckPoint(prev)
            checkpoint.id = checkpoints['S' + str(i)]['id']

            checkpoint.set_state(pos)

            if 'error_ellipsoid' in checkpoints['S' + str(i)].keys():
                checkpoint.error_ellipsoid = checkpoints['S' + str(i)]['error_ellipsoid']

            if 'manoeuvre' in checkpoints['S' + str(i)].keys():
                checkpoint.manoeuvre_type = checkpoints['S' + str(i)]['manoeuvre']

            if 't_min' in checkpoints['S' + str(i)].keys():
                checkpoint.t_min = checkpoints['S' + str(i)]['t_min']

            if 't_max' in checkpoints['S' + str(i)].keys():
                checkpoint.t_max = checkpoints['S' + str(i)]['t_max']

            self.checkpoints.append(checkpoint)
