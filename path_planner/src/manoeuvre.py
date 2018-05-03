# @copyright Copyright (c) 2017, Davide Frey (frey.davide.ae@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

"""Class holding the definitions of Manoeuvres."""

import numpy as np

from rospace_lib import CartesianLVLH
from copy import deepcopy


class Manoeuvre(object):
    """
        Base class that contains the definition of a manoeuvre for absolute navigation.

    Attributes:
        deltaV (array): Amount of delta-v needed to execute the burn. Given in kilometers per second with respect
            to TEME reference frame.
        execution_epoch (datetime): Epoch at which the manoeuvre should be executed.
    """

    def __init__(self):
        self.deltaV = np.array([0, 0, 0])
        self.execution_epoch = None


class RelativeMan(Manoeuvre):
    """
        Extended class for manoeuvres in relative navigation.

    Attributes:
        initial_rel_state (CartesianLVLH): Relative state at which manoeuvre should occur given in LVLH frame.
    """

    def __init__(self):
        super(RelativeMan, self).__init__()

        self.initial_rel_state = CartesianLVLH()

    def set_initial_rel_state(self, rel_state):
        """
            Define the starting relative state of the manoeuvre.

        Args:
            rel_state (CartesianLVLH): Relative state given in LVLH frame.
        """

        self.initial_rel_state = deepcopy(rel_state)
