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


class Manoeuvre(object):
    """
        Base class that contains the definition of a manoeuvre.

    Attributes:
        dV (np.array): Amount of delta-v needed to execute the burn. Given in kilometers per second with respect
            to TEME reference frame.
        duration (float64): Waiting time from last manoeuvre to the next one, given in seconds. (TO BE REVIEWED - Maybe useless)
        abs_state (KepOrbElem): Absolute state at which the manoeuvre has to be executed.
        description (string): Brief description of the manoeuvre.
    """

    def __init__(self):
        self.dV = np.array([0, 0, 0])
        self.duration = 0

        # The state in which the manoeuvre should be executed
        self.abs_state = KepOrbElem()

        # Description on the manoeuvre purpose
        self.description = None

    def set_abs_state(self, abs_state):
        """
            Given absolute state in keplerian orbital elements, set the absolute state.

        Args:
            abs_state (KepOrbElem): Absolute state in keplerian orbital elements.
        """

        self.abs_state.a = abs_state.a
        self.abs_state.e = abs_state.e
        self.abs_state.i = abs_state.i
        self.abs_state.O = abs_state.O
        self.abs_state.w = abs_state.w
        self.abs_state.v = abs_state.v

    def remove_lock(self):
        """
            Remove thread lock from the KepOrbElem to be able to save it in a pickle file.
        """
        # TODO
        del self.abs_state._lock


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
