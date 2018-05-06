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
import os

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
        self.date = None

    def import_solved_scenario(self, filename):
        """
            Import a solved scenario, i.e the manoeuvre plan, from pickle file 'filename.pickle'

            Return:
                Manoeuvre plan, the list containing the manoeuvres to perform this scenario.
        """

        # Actual path
        abs_path = os.path.dirname(os.path.abspath(__file__))
        scenario_path = os.path.join(abs_path, '../example/' + filename + '.pickle')

        # Try to import the file
        try:
            with open(scenario_path, 'rb') as file:
                obj = pickle.load(file)
                print "\n ----> Offline solution loaded! <---- \n"
                self.name = obj['scenario_name']
                self.overview = obj['scenario_overview']
                self.checkpoints = obj['checkpoints']
                self.target_ic.set_from_satellite(obj['target_ic'])
                self.chaser_ic.set_from_satellite(obj['chaser_ic'])
                self.date = obj['scenario_epoch']

                return obj['manoeuvre_plan']
        except IOError:
            raise IOError('Scenario file not found!')

    def export_solved_scenario(self, manoeuvre_plan):
        """
            Export a solved scenario into pickle file 'scenario_name_date.pickle' in the /example folder.
        """

        # Actual path
        abs_path = os.path.dirname(os.path.abspath(__file__))

        # Check if "/example" folder exists
        if not os.path.exists(os.path.join(abs_path, '../example')):
            os.makedirs(os.path.join(abs_path, '../example'))

        filename = self.name.replace(' ', '_') + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        pickle_path = os.path.join(abs_path, '../example/' + filename + '.pickle')

        with open(pickle_path, 'wb') as file:

            # Delete propagator to be able to dump satellite in pickle file
            del self.target_ic.prop
            del self.chaser_ic.prop

            obj = {'scenario_name': self.name,
                   'scenario_overview': self.overview,
                   'checkpoints': self.checkpoints,
                   'manoeuvre_plan': manoeuvre_plan,
                   'target_ic': self.target_ic,
                   'chaser_ic': self.chaser_ic,
                   'scenario_epoch': self.date}

            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

            print "\n[INFO]: Manoeuvre plan saved in " + filename

    def import_yaml_scenario(self, filename, ic_name='std_ic'):
        """
            Parse scenario and import initial conditions from .yaml files in the /cfg folder.

        Args:
            filename (str): name of the scenario yaml configuration file.
        """

        # Opening scenario file
        abs_path = os.path.dirname(os.path.abspath(__file__))
        scenario_path = os.path.join(abs_path, '../cfg/' + filename + '.yaml')
        scenario_file = file(scenario_path, 'r')
        scenario = yaml.load(scenario_file)

        if 'scenario' in scenario.keys():
            scenario = scenario['scenario']
        else:
            raise IOError('Missing "scenario" key in yaml file!')

        if 'checkpoints' in scenario.keys():
            checkpoints = scenario['checkpoints']
        else:
            raise IOError('Missing "checkpoints" key in yaml file!')

        # Assign variables
        if 'name' in scenario.keys():
            self.name = scenario['name']
        else:
            raise IOError('Missing "name" key in yaml file!')

        if 'overview' in scenario.keys():
            self.overview = scenario['overview']
        else:
            raise IOError('Missing "overview" key in yaml file!')

        # Initialize satellites
        self.target_ic.initialize_satellite(ic_name, 'target', scenario['prop_type'])
        self.chaser_ic.initialize_satellite(ic_name, 'chaser', scenario['prop_type'], self.target_ic)

        # Initialize date from satellites
        self.date = self.target_ic.prop.date

        # Extract CheckPoints
        for i in xrange(0, len(checkpoints)):
            pos = checkpoints['S' + str(i)]['state']

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
