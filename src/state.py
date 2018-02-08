# @copyright Copyright (c) 2017, Davide Frey (frey.davide.ae@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

"""
    Class defining the state of a satellite.
"""

import numpy as np

from space_tf import KepOrbElem, CartesianLVLH, Cartesian


class Satellite(object):
    """
        Class that holds the basic information of a Satellite.

        Attributes:
            abs_state (KepOrbElem): Keplerian orbital element of the satellite.
    """

    def __init__(self):
        self.abs_state = KepOrbElem()

    def set_from_other_satellite(self, satellite):
        """
            Create a copy of the given satellite.

        Args:
            satellite (Chaser, Target, Satellite): State of a satellite
        """

        if type(self) != type(satellite):
            raise TypeError

        self.abs_state.a = satellite.abs_state.a
        self.abs_state.e = satellite.abs_state.e
        self.abs_state.i = satellite.abs_state.i
        self.abs_state.O = satellite.abs_state.O
        self.abs_state.w = satellite.abs_state.w
        self.abs_state.v = satellite.abs_state.v

        if hasattr(satellite, 'rel_state'):
            self.rel_state.R = satellite.rel_state.R
            self.rel_state.V = satellite.rel_state.V

    def set_abs_state_from_tle(self, tle):
        """
            Given TLE coordinates set the absolute state.

        Args:
            tle (Dictionary): TLE coordinates.
        """

        self.abs_state.from_tle(eval(str(tle['i'])),
                                eval(str(tle['O'])),
                                eval(str(tle['e'])),
                                eval(str(tle['m'])),
                                eval(str(tle['w'])),
                                eval(str(tle['n'])))


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

    def set_rel_state_from_abs_state(self, target):
        """
            Set relative state given target and chaser absolute state.

        Args:
            target (Satellite): State of the target.
        """

        target_cart = Cartesian()
        chaser_cart = Cartesian()

        target_cart.from_keporb(target.abs_state)
        chaser_cart.from_keporb(self.abs_state)

        self.rel_state.from_cartesian_pair(chaser_cart, target_cart)

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

    def set_abs_state_from_kep(self, kep, target=None):
        """
            Given keplerian orbital elements set the absolute state.

        Args:
            kep (Dictionary or KepOrbElem): Keplerian orbital elements stored either in a dictionary or in KepOrbElem.
            target (Satellite): State of the target.
        """

        if type(kep) == dict:
            # Note: "target" is needed because the initial conditions may be defined with respect to the target state.
            # Therefore, when the string is evaluated you need to give also the target and to have the same name both
            # in this function and in the initial_conditions.yaml file.
            self.abs_state.a = eval(str(kep['a']))
            self.abs_state.e = eval(str(kep['e']))
            self.abs_state.i = eval(str(kep['i']))
            self.abs_state.O = eval(str(kep['O']))
            self.abs_state.w = eval(str(kep['w']))
            self.abs_state.v = eval(str(kep['v']))

        elif type(kep) == KepOrbElem:
            self.abs_state.a = kep.a
            self.abs_state.e = kep.e
            self.abs_state.i = kep.i
            self.abs_state.O = kep.O
            self.abs_state.w = kep.w
            self.abs_state.v = kep.v
