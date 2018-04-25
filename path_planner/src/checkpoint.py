# @copyright Copyright (c) 2017, Davide Frey (frey.davide.ae@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

"""Class holding the definitions of Check Points."""

import numpy as np

from rospace_lib import Cartesian, KepOrbElem, CartesianLVLH


class CheckPoint(object):
    """
        Base class that contains the definition of a checkpoint.

        Attributes:
            id (int): ID number which identify the checkpoint.
            manoeuvre_type (str): Force a specific manoeuvre type, can be either "standard" - uses only the basic
                functions, "radial" - doing a radial manoeuvre with a specific flight time or "drift" - where the
                spacecraft tries to drift to the checkpoint and if it cannot it apply a apogee/perigee change manoeuvre
                to be able to drift to the wanted position.
    """

    def __init__(self, id=0):
        self.id = id
        self.manoeuvre_type = 'standard'

    def set_from_checkpoint(self, chkp):
        """
            Set checkpoint parameters given another checkpoint, which can be either absolute or relative.

        Args:
            chkp (AbsoluteCP or RelativeCP)
        """

        if type(chkp) == type(self):
            self.id = chkp.id
            self.manoeuvre_type = chkp.manoeuvre_type

            if hasattr(chkp, 'rel_state'):
                self.rel_state = chkp.rel_state
                self.error_ellipsoid = chkp.error_ellipsoid
                self.t_min = chkp.t_min
                self.t_max = chkp.t_max
            elif hasattr(chkp, 'abs_state'):
                self.abs_state = chkp.abs_state
            else:
                raise Warning('Working with checkpoints instead of relativeCP or absoluteCP')
        else:
            raise TypeError()


class RelativeCP(CheckPoint):
    """
        Relative checkpoint class, based on Checkpoint.
        Holds the information for relative navigation.

        Attributes:
            rel_state (CartesianLVLH): Holds the relative coordinates with respect to a satellite.
            error_ellipsoid (np.array): Define an imaginary ellipsoid around the checkpoint, in which the manoeuvre is
                still performable. Measures are in km and goes according to the LVLH frame, i.e:
                [error(R-bar), error(V-bar), error(H-bar)]
            t_min (float64): Minimum time allowed to execute the manoeuvre [s], standard is 1.0 second.
            t_max (float64): Maximum time allowed to execute the manoeuvre [s], standard is 10 hours.
    """

    def __init__(self):
        super(RelativeCP, self).__init__()

        self.rel_state = CartesianLVLH()
        self.error_ellipsoid = [0.0, 0.0, 0.0]
        self.t_min = 1.0
        self.t_max = 36000.0

    def set_rel_state(self, rel_state):
        """
            Set relative state given relative state of the checkpoint in dictionary format.

        Args:
            rel_state (Dictionary or CartesianLVLH): Relative state of the checkpoint given either as a dictionary or
                as a CartesianLVLH object.
        """

        if type(rel_state) == dict:
            self.rel_state.R = np.array(rel_state['R']) if 'R' in rel_state.keys() else np.array([0.0, 0.0, 0.0])
            self.rel_state.V = np.array(rel_state['V']) if 'V' in rel_state.keys() else np.array([0.0, 0.0, 0.0])
        elif type(rel_state) == CartesianLVLH:
            self.rel_state.R = rel_state.R
            self.rel_state.V = rel_state.V
        else:
            raise TypeError


class AbsoluteCP(CheckPoint):
    """
        Absolute checkpoint class, based on Checkpoint.
        Holds the informations for absolute navigation.

    Args:
        abs_state (KepOrbElem): Absolute state of the checkpoint in keplerian orbital elements.
    """

    def __init__(self):
        super(AbsoluteCP, self).__init__()

        self.abs_state = KepOrbElem()

    def set_abs_state(self, abs_state):
        """
            Set absolute state given absolute state either in a dictionary.

        Args:
            abs_state (Dictionary or KepOrbElem): absolute state of checkpoint given either in a dictionary or in
                KepOrbElem.
        """

        if type(abs_state) == dict:
            self.abs_state.a = eval(str(abs_state['a'])) if 'a' in abs_state.keys() else 0.0
            self.abs_state.e = eval(str(abs_state['e'])) if 'e' in abs_state.keys() else 0.0
            self.abs_state.i = eval(str(abs_state['i'])) if 'i' in abs_state.keys() else 0.0
            self.abs_state.O = eval(str(abs_state['O'])) if 'O' in abs_state.keys() else 0.0
            self.abs_state.w = eval(str(abs_state['w'])) if 'w' in abs_state.keys() else 0.0
            self.abs_state.v = eval(str(abs_state['v'])) if 'v' in abs_state.keys() else 0.0
        elif type(abs_state) == KepOrbElem:
            self.abs_state.a = abs_state.a
            self.abs_state.e = abs_state.e
            self.abs_state.i = abs_state.i
            self.abs_state.O = abs_state.O
            self.abs_state.w = abs_state.w
            self.abs_state.v = abs_state.v
        else:
            raise TypeError
