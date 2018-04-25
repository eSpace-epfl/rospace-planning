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

from state import Satellite, Chaser
from checkpoint import AbsoluteCP, RelativeCP
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
        self.chaser_ic = Chaser()
        self.target_ic = Satellite()

        # Target Keep-Out Zones
        self.approach_ellipsoid = np.array([0.0, 0.0, 0.0])
        self.koz_r = 0.0

        # Scenario starting date
        self.date = datetime.utcnow()

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
                    self.target_ic.set_from_satellite(obj['target_ic'])
                    self.chaser_ic.set_from_satellite(obj['chaser_ic'])
                    self.date = obj['scenario_epoch']

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

            # Delete propagator to be able to dump satellite in pickle file
            del self.target_ic.prop
            del self.chaser_ic.prop

            obj = {'scenario_name': self.name, 'checkpoints': self.checkpoints, 'manoeuvre_plan': manoeuvre_plan,
                   'target_ic': self.target_ic, 'chaser_ic': self.chaser_ic, 'scenario_epoch': self.date}

            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

            print "Manoeuvre plan saved..."

    def import_yaml_scenario(self, filename):
        """
            Parse scenario and import initial conditions from .yaml files in the /cfg folder.

        Args:
            filename (str): name of the scenario yaml configuration file.
        """

        # Actual path
        abs_path = sys.argv[0]
        path_idx = abs_path.find('path_planner')
        abs_path = abs_path[0:path_idx]

        if 'unittest' in abs_path:
            abs_path = '/home/dfrey/rospace_ws/src/planning/'

        # Opening scenario file
        scenario_path = abs_path + 'path_planner/cfg/' + filename + '.yaml'
        scenario_file = file(scenario_path, 'r')
        scenario = yaml.load(scenario_file)
        scenario = scenario['scenario']
        checkpoints = scenario['CheckPoints']

        # Assign variables
        self.name = scenario['name']
        self.overview = scenario['overview']

        # Initialize satellites
        self.target_ic.initialize_satellite('target', self.date, scenario['prop_type'])
        self.chaser_ic.initialize_satellite('chaser', self.date, scenario['prop_type'], self.target_ic)

        # Extract CheckPoints
        for i in xrange(0, len(checkpoints)):
            pos = checkpoints['S' + str(i)]['position']

            if 'kep' in pos.keys():
                checkpoint = AbsoluteCP()
                checkpoint.set_abs_state(pos['kep'])
            elif 'lvlh' in pos.keys():
                checkpoint = RelativeCP()
                checkpoint.set_rel_state(pos['lvlh'])
            else:
                raise TypeError()

            checkpoint.id = checkpoints['S' + str(i)]['id']

            if 'error_ellipsoid' in checkpoints['S' + str(i)].keys():
                checkpoint.error_ellipsoid = checkpoints['S' + str(i)]['error_ellipsoid']

            if 'manoeuvre' in checkpoints['S' + str(i)].keys():
                checkpoint.manoeuvre_type = checkpoints['S' + str(i)]['manoeuvre']

            if 't_min' in checkpoints['S' + str(i)].keys():
                checkpoint.t_min = checkpoints['S' + str(i)]['t_min']

            if 't_max' in checkpoints['S' + str(i)].keys():
                checkpoint.t_max = checkpoints['S' + str(i)]['t_max']

            self.checkpoints.append(checkpoint)
