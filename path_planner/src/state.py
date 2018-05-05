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

from rospace_lib import KepOrbElem, CartesianTEME, OscKepOrbElem, CartesianLVLH
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

    def initialize_satellite(self, ic_name, name, prop_type, target=None):
        """
            Initialize satellite attributes from the configuration files.

        Args:
            ic_name (str): Name of the initial conditions that has to be imported.
            name (str): Name of the satellite, which should be stated as well in initial_conditions.yaml file.
            prop_type (str): Define the type of propagator to be used (2-body or real-world).
            target (Satellite): If self is a chaser type satellite, the reference target is needed to define the
                relative state with respect to it.
        """

        # Opening initial conditions file
        abs_path = os.path.dirname(os.path.abspath(__file__))
        initial_conditions_path = os.path.join(abs_path, '../cfg/initial_conditions.yaml')
        initial_conditions_file = file(initial_conditions_path, 'r')
        initial_conditions = yaml.load(initial_conditions_file)

        # Check if the initial conditions name is stated in the cfg file
        if ic_name in initial_conditions.keys():
            initial_conditions = initial_conditions[ic_name]
        else:
            raise IOError('Initial condition named ' + ic_name + ' not defined in .yaml file!')

        # Check if the satellite initial date is stated in the cfg file
        if 'date' in initial_conditions.keys():
            date = eval(initial_conditions['date'])
        else:
            raise IOError('Initial date for satellite ' + name + ' not stated in .yaml file!')

        # Check if the satellite initial conditions are stated in the cfg file
        if name in initial_conditions.keys():
            initial_conditions = initial_conditions[name]
        else:
            raise IOError('Initial conditions for satellite ' + name + ' not stated in .yaml file!')

        # Check if initial conditions are stated in term on KepOrbElem
        if 'kep' in initial_conditions.keys():
            kep_ic = initial_conditions['kep']
        else:
            raise IOError('Initial conditions for satellite ' + name + ' not stated as KepOrbElem in .yaml file!')

        # Create a KepOrbElem to contain the initial conditions
        sat_ic = KepOrbElem()
        sat_ic.a = eval(str(kep_ic['a']))
        sat_ic.e = eval(str(kep_ic['e']))
        sat_ic.i = eval(str(kep_ic['i']))
        sat_ic.O = eval(str(kep_ic['O']))
        sat_ic.w = eval(str(kep_ic['w']))

        if 'v' in kep_ic.keys():
            sat_ic.v = eval(str(kep_ic['v']))
        elif 'm' in kep_ic.keys():
            sat_ic.m = eval(str(kep_ic['m']))
        elif 'E' in kep_ic.keys():
            sat_ic.E = eval(str(kep_ic['E']))
        else:
            raise AttributeError('Anomaly initial condition for satellite ' + name + ' not defined properly!')

        # Assign absolute state
        self.abs_state.from_keporb(sat_ic)

        # Assign relative state
        if type(self) == Chaser:
            if target is not None:
                self.rel_state.from_cartesian_pair(self.abs_state, target.abs_state)
            else:
                raise IOError('Missing target input to initialize chaser!')

        # Assign mass
        self.mass = initial_conditions['mass']

        # Create and initialize propagator
        self.prop = QuickPropagator(date)
        self.prop.initialize_propagator(name, sat_ic, prop_type)

    def set_from_satellite(self, satellite):
        """
            Set attributes of the satellite using as reference another satellite.

        Args:
            satellite (Satellite or Chaser)
        """

        if type(self) != type(satellite):
            raise TypeError()

        self.abs_state.R = satellite.abs_state.R
        self.abs_state.V = satellite.abs_state.V
        self.mass = satellite.mass

        if hasattr(satellite, 'prop'):
            self.prop = satellite.prop
        else:
            print '[WARNING]: Propagator not setted!!'

        if hasattr(self, 'rel_state'):
            self.rel_state.R = satellite.rel_state.R
            self.rel_state.V = satellite.rel_state.V

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
