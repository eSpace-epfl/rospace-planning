# @copyright Copyright (c) 2017, Davide Frey (frey.davide.ae@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

"""Class holding the definitions of Manoeuvres."""

import numpy as np

from space_tf import KepOrbElem, CartesianLVLH
from threading import RLock


class Manoeuvre(object):
    """
        Base class that contains the definition of a manoeuvre.

    Attributes:
        dV (array): Amount of delta-v needed to execute the burn. Given in kilometers per second with respect
            to TEME reference frame.
        duration (float64): Waiting time from last manoeuvre to the next one, given in seconds. (TO BE REVIEWED - Maybe useless)
        initial_state (KepOrbElem or CartesianLVLH): State at which the manoeuvre has to be executed, can be either
            defined in mean orbital elements or in cartesian LVLH reference frame.
        description (str): Brief description of the manoeuvre.
    """

    def __init__(self):
        self.dV = np.array([0, 0, 0])
        self.duration = 0
        self.initial_state = None
        self.description = None

    def set_initial_state(self, state):
        """Set manoeuvre initial state in KepOrbElem or CartesianLVLH.

        Args:
            abs_state (KepOrbElem): Absolute state in keplerian orbital elements.
        """

        state_type = type(state)

        if self.initial_state == None:
            if state_type == KepOrbElem:
                self.initial_state = KepOrbElem()
                self.initial_state.a = state.a
                self.initial_state.e = state.e
                self.initial_state.i = state.i
                self.initial_state.O = state.O
                self.initial_state.w = state.w
                self.initial_state.v = state.v
            elif state_type == CartesianLVLH:
                self.initial_state = CartesianLVLH()
                self.initial_state.R = state.R
                self.initial_state.V = state.V
            else:
                raise TypeError('State type not allowed!')
        else:
            raise AttributeError('Manoeuvre initial state has already been defined!')


class RelativeMan(Manoeuvre):
    """
        Extended class for manoeuvres in relative navigation.

    Attributes:
        rel_state (CartesianLVLH): Relative state at which manoeuvre should occur given in LVLH frame.
    """

    def __init__(self):
        super(RelativeMan, self).__init__()

        self.rel_state = CartesianLVLH()

    def set_rel_state(self, rel_state):
        """
            Define the starting relative state of the manoeuvre.

        Args:
            rel_state (CartesianLVLH): Relative state given in LVLH frame.
        """

        self.rel_state.R = rel_state.R
        self.rel_state.V = rel_state.V
