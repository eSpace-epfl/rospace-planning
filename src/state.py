# @copyright Copyright (c) 2017, Davide Frey (frey.davide.ae@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

"""Class defining the state of a satellite."""

import numpy as np

from space_tf import KepOrbElem, CartesianLVLH, Cartesian
from threading import RLock


class Satellite(object):
    """
        Class that holds the basic information of a Satellite.

        Attributes:
            mass (float64): Mass of the satellite in [kg].
            abs_state (Cartesian): Cartesian absolute position of the satellite with respect to Earth Inertial frame.
    """

    def __init__(self):
        self.mass = 0.0
        self.abs_state = Cartesian()

    def set_from_satellite(self, satellite):
        """
            Set attributes of the satellite using as reference another satellite.

        Args:
            satellite (Satellite, Chaser): State of a satellite.
        """

        if type(self) != type(satellite):
            raise TypeError

        self.abs_state.R = satellite.abs_state.R
        self.abs_state.V = satellite.abs_state.V
        self.mass = satellite.mass

        if hasattr(satellite, 'rel_state'):
            self.rel_state.R = satellite.rel_state.R
            self.rel_state.V = satellite.rel_state.V

    def set_abs_state_from_tle(self, tle):
        """
            Given TLE coordinates set the absolute state.

        Args:
            tle (Dictionary): TLE coordinates.
        """

        if type(tle) == dict:
            kep = KepOrbElem()
            kep.from_tle(eval(str(tle['i'])),
                         eval(str(tle['O'])),
                         eval(str(tle['e'])),
                         eval(str(tle['m'])),
                         eval(str(tle['w'])),
                         eval(str(tle['n'])))

            self.abs_state.from_keporb(kep)
        else:
            raise TypeError

    def set_abs_state_from_kep(self, kep):
        """
            Given keplerian orbital elements set the absolute state.

        Args:
            kep (Dictionary or KepOrbElem): Keplerian orbital elements stored either in a dictionary or in KepOrbElem.
        """

        if type(kep) == dict:
            kep_state = KepOrbElem()
            kep_state.a = eval(str(kep['a']))
            kep_state.e = eval(str(kep['e']))
            kep_state.i = eval(str(kep['i']))
            kep_state.O = eval(str(kep['O']))
            kep_state.w = eval(str(kep['w']))
            kep_state.v = eval(str(kep['v']))
            self.abs_state.from_keporb(kep_state)
        elif type(kep) == KepOrbElem:
            self.abs_state.from_keporb(kep)
        else:
            raise TypeError

    def get_osc_oe(self):
        """Return the osculating orbital elements of the satellite."""
        kep_osc = KepOrbElem()
        kep_osc.from_cartesian(self.abs_state)

        return kep_osc

    def get_mean_oe(self, prop_type='real-world'):
        """Return mean orbital elements of the satellite."""
        kep_osc = self.get_osc_oe()

        if prop_type == 'real-world':
            kep_mean = KepOrbElem()
            kep_mean.from_osc_elems(kep_osc)
            return kep_mean
        elif prop_type == '2-body':
            return kep_osc
        else:
            raise TypeError('Propagator type not recognized!')


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

    def set_abs_state_from_rel_state(self, target):
        """
            Set keplerian elements given target absolute state and chaser relative state.

        Args:
             target (Satellite): State of the target.
        """

        target_cart = Cartesian()
        chaser_cart = Cartesian()

        target_cart.from_keporb(target.abs_state)
        chaser_cart.from_lvlh_frame(target_cart, self.rel_state)

        self.abs_state.from_cartesian(chaser_cart)
