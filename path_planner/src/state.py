# @copyright Copyright (c) 2017, Davide Frey (frey.davide.ae@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

"""Class defining the state of a satellite."""

import yaml
import os

from rospace_lib import KepOrbElem, CartesianTEME, OscKepOrbElem, CartesianLVLH, Cartesian
from rospace_lib.misc import QuickPropagator
from copy import deepcopy
from datetime import datetime


class Satellite(object):
    """
        Class that holds the basic information of a Satellite, its cartesian absolute position, its mass and its
        identification name.

    Attributes:
        mass (float64): Mass of the satellite in [kg].
        abs_state (CartesianTEME): Cartesian absolute position of the satellite with respect to Earth Inertial frame.
        name (str): Name of the satellite.
        prop (QuickPropagator): QuickPropagator of this satellite.
    """

    def __init__(self):
        self.mass = 0.0
        self.abs_state = CartesianTEME()
        self.name = ''
        self.prop = None

    @staticmethod
    def export_initial_condition(sat_name, ic_name):
        """
            Function to export some initial conditions from the yaml file, given the name of them.

        Args:
            sat_name (str): Satellite name
            ic_name (str): Name of the initial conditions that has to be exported.

        Return:
            state (OscKepOrbElem): State exported from the initial conditions in osculating KepOrbElem.
            mass (float64): Mass of the satellite.
            date (float64): Initial propagation date.
        """

        # Opening initial conditions file
        abs_path = os.path.dirname(os.path.abspath(__file__))
        initial_conditions_path = os.path.join(abs_path, '../cfg/initial_conditions.yaml')
        initial_conditions_file = file(initial_conditions_path, 'r')
        initial_conditions = yaml.load(initial_conditions_file)

        # Export initial state
        state = OscKepOrbElem()
        state.a = initial_conditions[ic_name][sat_name]['kep']['a']
        state.e = initial_conditions[ic_name][sat_name]['kep']['e']
        state.i = initial_conditions[ic_name][sat_name]['kep']['i']
        state.O = initial_conditions[ic_name][sat_name]['kep']['O']
        state.w = initial_conditions[ic_name][sat_name]['kep']['w']

        if 'v' in initial_conditions[ic_name][sat_name]['kep'].keys():
            state.v = initial_conditions[ic_name][sat_name]['kep']['v']
        elif 'E' in initial_conditions[ic_name][sat_name]['kep'].keys():
            state.E = initial_conditions[ic_name][sat_name]['kep']['E']
        elif 'm' in initial_conditions[ic_name][sat_name]['kep'].keys():
            state.m = initial_conditions[ic_name][sat_name]['kep']['m']
        else:
            raise AttributeError('No anomaly defined!')

        init_state = Cartesian()
        init_state.from_keporb(state)

        # Export mass
        mass = initial_conditions[ic_name][sat_name]['mass']

        # Export date
        date = eval(initial_conditions[ic_name]['date'])

        return init_state, mass, date

    def initialize_satellite(self, name, ic_name, prop_type, target=None):
        """
            Initialize the satellite taking as input the initial conditions name, and the propagator informations.

        Args:
            name (str): Satellite name (has to match the name of the propagator's settings).
            ic_name (str): Initial condition name state in the yaml file.
            prop_type (str): Propagator type.
            target (Satellite): If the initialized satellite is of type Chaser it needs a reference spacecraft,
                the target.
        """


        state, mass, date = self.export_initial_condition(name, ic_name)

        self.abs_state = state
        self.mass = mass
        self.name = name

        # Set relative state in case self is Chaser
        if type(self) == Chaser:
            if target is not None:
                self.rel_state.from_cartesian_pair(self.abs_state, target.abs_state)
            else:
                raise IOError('Missing target input to initialize relative state!')

        # Create and initialize propagator
        self.prop = QuickPropagator()

        init_state = dict()

        init_state["position"] = state
        init_state["spin"] = [0.0, 0.0, 0.0]
        init_state["rotation_acceleration"] = [0.0, 0.0, 0.0]
        init_state["attitude"] = [0.0, 0.0, 0.0, 1.0]
        self.prop.initialize_propagator(name, init_state, prop_type)

    def set_abs_state_from_cartesian(self, cartesian):
        """
            Given some cartesian coordinates set the absolute state of the satellite.

        Args:
            cartesian (Cartesian)
        """

        if cartesian.frame == self.abs_state.frame:
            self.abs_state = deepcopy(cartesian)
        else:
            raise TypeError()

    def get_osc_oe(self):
        """
            Return the osculating orbital elements of the satellite.

        Return:
              kep_osc (OscKepOrbElem): Osculating orbital elements.
        """

        kep_osc = OscKepOrbElem()
        kep_osc.from_cartesian(self.abs_state)

        return kep_osc

    def get_mean_oe(self):
        """
            Return mean orbital elements of the satellite.

        Return:
            kep_mean (KepOrbElem): Mean orbital elements.
        """

        kep_osc = self.get_osc_oe()

        kep_mean = KepOrbElem()

        if self.prop.prop_type == 'real-world':
            kep_mean.from_osc_elems(kep_osc)
        elif self.prop.prop_type == '2-body':
            kep_mean.from_osc_elems(kep_osc, 'null')
        else:
            raise TypeError('Propagator type not recognized!')

        return kep_mean


class Chaser(Satellite):
    """
        Class that holds the information for a chaser. In addition to the Satellite information, also the relative
        position with respect to another satellite (target) is needed.

        Attributes:
            rel_state (CartesianLVLH): Holds the relative coordinates with respect to another satellite.
    """

    def __init__(self):
        super(Chaser, self).__init__()

        self.rel_state = CartesianLVLH()

    def set_abs_state_from_target(self, target):
        """
            Set absolute state given target absolute state and chaser relative state.

        Args:
             target (Satellite): State of the target.
        """

        self.abs_state.from_lvlh_frame(target.abs_state, self.rel_state)
