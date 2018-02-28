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
from checkpoint import RelativeCP, AbsoluteCP
from propagator.OrekitPropagator import OrekitPropagator
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
            chaser_ic (Chaser): Chaser initial state.
            target_ic (Satellite): Target initial state.
            prop_chaser (OrekitPropagator): Chaser propagator.
            prop_target (OrekitPropagator): Target propagator.
            approach_ellipsoid (np.array): Axis of the approach ellipsoid around the target [km].
            koz_r (float64): Radius of Keep-Out Zone drawn around the target [km].
    """

    def __init__(self):
        # Scenario information
        self.name = 'Standard'
        self.overview = ''

        # Checkpoint list
        self.checkpoints = []

        # Chaser and target actual state
        self.chaser_ic = Chaser()
        self.target_ic = Satellite()

        # Propagators
        OrekitPropagator.init_jvm()
        self.prop_chaser = OrekitPropagator()
        self.prop_target = OrekitPropagator()
        self.date = datetime.utcnow()

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

                    self.checkpoints = obj['checkpoints']

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
            Export a solved scenario into pickle file 'scenario.pickle' in the /example folder.
        """

        # Actual path
        abs_path = sys.argv[0]
        path_idx = abs_path.find('cso_path_planner')
        abs_path = abs_path[0:path_idx + 16]

        pickle_path = abs_path + '/example/scenario.pickle'

        with open(pickle_path, 'wb') as file:

            # Remove locks
            for man in manoeuvre_plan:
                man.remove_lock()

            for chkp in self.checkpoints:
                chkp.remove_lock()

            self.chaser_ic.remove_lock()
            self.target_ic.remove_lock()

            obj = {'scenario_name': self.name, 'checkpoints': self.checkpoints, 'manoeuvre_plan': manoeuvre_plan,
                   'chaser_ic': self.chaser_ic, 'target_ic': self.target_ic}

            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

            print "Manoeuvre plan saved..."

    def import_yaml_scenario(self):
        """
            Parse scenario and import initial conditions from .yaml files in the /cfg folder.
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
        self.koz_r = scenario['keep_out_zone']
        self.approach_ellipsoid = scenario['approach_ellipsoid']

        # Assign initial conditions, assuming target in tle and chaser in keplerian
        self.target_ic.set_abs_state_from_tle(target_ic['tle'])
        self.chaser_ic.set_abs_state(chaser_ic['kep'], self.target_ic)
        self.chaser_ic.set_rel_state_from_abs_state(self.target_ic)

        # Initialize propagators
        self.initialize_propagators(self.chaser_ic.abs_state, self.target_ic.abs_state, self.date)

        # Extract CheckPoints
        for i in xrange(0, len(checkpoints)):
            pos = checkpoints['S' + str(i)]['position']
            ref_frame = pos.keys()[0]

            prev = self.checkpoints[-1] if len(self.checkpoints) > 0 else None

            if ref_frame == 'lvlh':
                checkpoint = RelativeCP()
                checkpoint.set_rel_state(pos[ref_frame], self.chaser_ic, self.target_ic)
                checkpoint.error_ellipsoid = checkpoints['S' + str(i)]['error_ellipsoid']
                if 'manoeuvre' in checkpoints['S' + str(i)].keys():
                    checkpoint.manoeuvre_type = checkpoints['S' + str(i)]['manoeuvre']
                if 't_min' in checkpoints['S' + str(i)].keys():
                    checkpoint.t_min = checkpoints['S' + str(i)]['t_min']
                if 't_max' in checkpoints['S' + str(i)].keys():
                    checkpoint.t_max = checkpoints['S' + str(i)]['t_max']
            elif ref_frame == 'kep':
                checkpoint = AbsoluteCP(prev)
                checkpoint.set_abs_state(pos[ref_frame], self.chaser_ic, self.target_ic)

            checkpoint.id = checkpoints['S' + str(i)]['id']

            self.checkpoints.append(checkpoint)

    def initialize_propagators(self, chaser_ic, target_ic, date):
        """
            Initialize orekit propagators given initial conditions.

        Args:
            chaser_ic (KepOrbElem): Initial conditions of chaser given in KepOrbElem.
            target_ic (KepOrbElem): Initial conditions of target given in KepOrbElem.
            epoch (datetime): Initialization date.
        """

        # Actual path
        abs_path = sys.argv[0]
        path_idx = abs_path.find('nodes')
        abs_path = abs_path[0:path_idx]

        chaser_settings_path = abs_path + 'simulator/cso_gnc_sim/cfg/chaser.yaml'
        target_settings_path = abs_path + 'simulator/cso_gnc_sim/cfg/target.yaml'

        chaser_settings = file(chaser_settings_path, 'r')
        target_settings = file(target_settings_path, 'r')

        propSettings = yaml.load(chaser_settings)
        propSettings = propSettings['propagator_settings']

        self.prop_chaser.initialize(propSettings,
                               chaser_ic,
                               date)

        propSettings = yaml.load(target_settings)
        propSettings = propSettings['propagator_settings']

        self.prop_target.initialize(propSettings,
                               target_ic,
                               date)