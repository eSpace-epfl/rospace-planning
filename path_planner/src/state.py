# @copyright Copyright (c) 2017, Davide Frey (frey.davide.ae@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

"""Class defining the state of a satellite."""

import sys
import yaml
import numpy as np

from rospace_lib import KepOrbElem, CartesianTEME, OscKepOrbElem
from path_planning_propagator import Propagator


class Satellite(object):
    """
        Class that holds the basic information of a Satellite, its cartesian absolute position, its mass and its
        identification name.

    Attributes:
        mass (float64): Mass of the satellite in [kg].
        abs_state (Cartesian): Cartesian absolute position of the satellite with respect to Earth Inertial frame.
        name (str): Name of the satellite.
    """

    def __init__(self):
        self.mass = 0.0
        self.abs_state = CartesianTEME()
        self.name = ''
        self.prop = Propagator()

    def initialize_satellite(self, name, date, prop_type='real-world'):
        """
            Initialize satellite attributes from the configuration files.

        Args:
            name (str): Name of the satellite, which should be stated as well in initial_conditions.yaml file.
        """

        # Actual path
        abs_path = sys.argv[0]
        path_idx = abs_path.find('path_planner')
        abs_path = abs_path[0:path_idx]

        # Opening initial conditions file
        initial_conditions_path = abs_path + 'path_planner/cfg/initial_conditions.yaml'
        initial_conditions_file = file(initial_conditions_path, 'r')
        initial_conditions = yaml.load(initial_conditions_file)

        # Check if the satellite initial conditions are stated in the configuration file
        if name in initial_conditions.keys():
            initial_conditions = initial_conditions[name]
        else:
            raise AttributeError('[ERROR]: Initial conditions for satellite ' + name +
                                 ' not stated in initial_conditions.yaml!')

        # Create a KepOrbElem to contain the initial conditions
        kep_ic = initial_conditions['kep']
        satellite_ic = KepOrbElem()
        satellite_ic.a = eval(str(kep_ic['a']))
        satellite_ic.e = eval(str(kep_ic['e']))
        satellite_ic.i = eval(str(kep_ic['i']))
        satellite_ic.O = eval(str(kep_ic['O']))
        satellite_ic.w = eval(str(kep_ic['w']))
        if 'v' in kep_ic.keys():
            satellite_ic.v = eval(str(kep_ic['v']))
        elif 'm' in kep_ic.keys():
            satellite_ic.m = eval(str(kep_ic['m']))
        elif 'E' in kep_ic.keys():
            satellite_ic.E = eval(str(kep_ic['E']))
        else:
            raise AttributeError('[ERROR]: Anomaly initial condition for satellite ' + name + ' not defined properly!')

        # Assign absolute state
        self.abs_state.from_keporb(satellite_ic)

        # Assign mass
        self.mass = eval(str(initial_conditions['mass']))

        # Assign propagator
        self.prop.initialize_propagator(name, satellite_ic, prop_type, date)

    def set_from_satellite(self, satellite):
        """
            Set attributes of the satellite using as reference another satellite.

        Args:
            satellite (Satellite)
        """

        if type(self) != type(satellite):
            raise TypeError()

        self.abs_state.R = satellite.abs_state.R
        self.abs_state.V = satellite.abs_state.V
        self.mass = satellite.mass
        self.prop = satellite.prop


    def set_abs_state_from_cartesian(self, cartesian):
        """
            Given some cartesian coordinates set the absolute state of the satellite.

        Args:
            cartesian (Cartesian)
        """

        self.abs_state.R = cartesian.R
        self.abs_state.V = cartesian.V

    def get_osc_oe(self):
        """
            Return the osculating orbital elements of the satellite.

        Return:
              kep_osc (KepOrbElem): Osculating orbital elements.
        """

        kep_osc = OscKepOrbElem()
        kep_osc.from_cartesian(self.abs_state)

        return kep_osc

    def get_mean_oe(self, prop_type='real-world'):
        """
            Return mean orbital elements of the satellite.

        Args:
            prop_type (str): Propagator type that has to be used, can be either a real-world propagator (standard) or
                a 2-body propagator.

        Return:
            kep_mean (KepOrbElem): Mean orbital elements.
        """

        kep_osc = self.get_osc_oe()

        kep_mean = KepOrbElem()

        if prop_type == 'real-world':
            kep_mean.from_osc_elems(kep_osc)
        elif prop_type == '2-body':
            kep_mean.from_osc_elems(kep_osc, 'null')
        else:
            raise TypeError('Propagator type not recognized!')

        return kep_mean


class Chaser(Satellite):

    def __init__(self):
        pass
