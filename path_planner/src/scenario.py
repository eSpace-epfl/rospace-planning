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

from checkpoint import AbsoluteCP, RelativeCP
from datetime import datetime
from rospace_lib import OscKepOrbElem
from state import Satellite


class Scenario(object):
    """
        Base class defining a scenario.
        In there the chaser and target initial conditions are stored, as well as the checkpoints of the missions and
        the some missions constraints.

        Attributes:
            name (str): Scenario name.
            overview (str): Brief scenario overview, explaining all the steps.
            prop_type (str): Propagator type for this scenario.
            ic_name (str): Name of the initial conditions chosen.
            checkpoints (list): A list containing all the checkpoints that has to be executed in the right order.
            chaser_ic (OscKepOrbElem): Chaser initial state.
            target_ic (OscKepOrbElem): Target initial state.
            chaser_mass (float64): Chaser initial mass.
            target_mass (float64): Target initial mass.
            date (timedelta): Time at which the simulation start.
            approach_ellipsoid (np.array): Axis of the approach ellipsoid around the target [km].
            koz_r (float64): Radius of Keep-Out Zone drawn around the target [km].
    """

    def __init__(self):
        # Scenario information
        self.name = 'Standard'
        self.overview = ''
        self.prop_type = ''
        self.ic_name = ''

        # Checkpoint list
        self.checkpoints = []

        # Satellite initial states
        self.chaser_ic = OscKepOrbElem()
        self.target_ic = OscKepOrbElem()
        self.chaser_mass = 0.0
        self.target_mass = 0.0

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
                self.target_ic = obj['target_ic']
                self.chaser_ic = obj['chaser_ic']
                self.ic_name = obj['ic_name']
                self.target_mass = obj['target_mass']
                self.chaser_mass = obj['chaser_mass']
                self.date = obj['scenario_epoch']
                self.prop_type = obj['prop_type']

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

            obj = {'scenario_name': self.name,
                   'scenario_overview': self.overview,
                   'checkpoints': self.checkpoints,
                   'manoeuvre_plan': manoeuvre_plan,
                   'target_ic': self.target_ic,
                   'chaser_ic': self.chaser_ic,
                   'ic_name': self.ic_name,
                   'target_mass': self.target_mass,
                   'chaser_mass': self.chaser_mass,
                   'scenario_epoch': self.date,
                   'prop_type': self.prop_type}

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

        # Set propagator type and initial conditions name
        self.ic_name = ic_name
        self.prop_type = scenario['prop_type']

        # self.target_ic, self.target_mass, self.date = Satellite.export_initial_condition('target', ic_name)
        # self.chaser_ic, self.chaser_mass, date_tmp = Satellite.export_initial_condition('chaser', ic_name)

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
