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
            state (Cartesian, KepOrbElem, CartesianLVLH): The state which is represented by this checkpoint.
            error_ellipsoid (array): Error allowed around a checkpoint, out of which the manoeuvre is considered failed.
                Defined as an array in the LVLH frame in which the components are the allowed error in each direction in
                [km].
            manoeuvre_type (str): Force a specific manoeuvre type, can be either "standard" - uses only the basic
                functions, "radial" - doing a radial manoeuvre with a specific flight time or "drift" - where the
                spacecraft tries to drift to the checkpoint and if it cannot it apply a apogee/perigee change manoeuvre
                to be able to drift to the wanted position.
            t_min (float64): Minimum allowed time for the manoeuvre in [s].
            t_max (float64): Maximum allowed time for the manoeuvre in [s].
    """

    def __init__(self, id=0):
        self.state = None
        self.id = id

        self.error_ellipsoid = [0.0, 0.0, 0.0]
        self.manoeuvre_type = 'standard'

        self.t_min = 1.0
        self.t_max = 36000.0

    def set_state(self, state):
        """Set the state of the checkpoint.

        Args:
            state (Cartesian, KepOrbElem, CartesianLVLH, Dictionary): The state that has to be reached, can be either
                defined in Cartesian coordinates, Keplerian orbital elements, Cartesian LVLH frame or in a dictionary
                containing the state in either one of those state.
                Note that if stated in KepOrbElem they are intended as Mean Orbital Elements and NOT osculating!
        """

        state_type = type(state)

        if self.state == None:
            if state_type == Cartesian:
                self.state = Cartesian()
                self.state.R = state.R
                self.state.V = state.V
            elif state_type == KepOrbElem:
                self.state = KepOrbElem()
                self.state.a = state.a
                self.state.e = state.e
                self.state.i = state.i
                self.state.O = state.O
                self.state.w = state.w
                self.state.v = state.v
            elif state_type == CartesianLVLH:
                self.state = CartesianLVLH()
                self.state.R = state.R
                self.state.V = state.V
            elif state_type == dict:
                if 'cart' in state.keys():
                    self.state = Cartesian()
                    self.state.R = eval(str(state['cart']['R']))
                    self.state.V = eval(str(state['cart']['V']))
                elif 'kep' in state.keys():
                    self.state = KepOrbElem()
                    self.state.a = eval(str(state['kep']['a']))
                    self.state.e = eval(str(state['kep']['e']))
                    self.state.i = eval(str(state['kep']['i']))
                    self.state.O = eval(str(state['kep']['O']))
                    self.state.w = eval(str(state['kep']['w']))
                    self.state.v = eval(str(state['kep']['v']))
                elif 'lvlh' in state.keys():
                    self.state = Cartesian()
                    self.state.R = eval(str(state['lvlh']['R']))
                    self.state.V = eval(str(state['lvlh']['V']))
                else:
                    raise TypeError('State dictionary reference frame not recognized!')
            else:
                raise TypeError('State type not allowed!')
        else:
            raise AttributeError('Checkpoint state has already been defined!')

    def set_from_checkpoint(self, chkp):
        """
            Set the attributes of a checkpoint given another checkpoint as reference.

        Args:
            chkp (CheckPoint)
        """

        self.set_state(chkp.state)
        self.id = chkp.id
        self.error_ellipsoid = chkp.error_ellipsoid
        self.manoeuvre_type = chkp.manoeuvre_type
        self.t_min = chkp.t_min
        self.t_max = chkp.t_max
