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
        self.set_state(chkp.state)
        self.id = chkp.id
        self.error_ellipsoid = chkp.error_ellipsoid
        self.manoeuvre_type = chkp.manoeuvre_type
        self.t_min = chkp.t_min
        self.t_max = chkp.t_max


class RelativeCP(CheckPoint):
    """
        Relative checkpoint class, based on Checkpoint.
        Holds the information for relative navigation.

        Attributes:
            rel_state (CartesianLVLH): Holds the relative coordinates with respect to a satellite.
            error_ellipsoid (np.array): Define an imaginary ellipsoid around the checkpoint, in which the manoeuvre is
                still performable. Measures are in km and goes according to the LVLH frame, i.e:
                [error(R-bar), error(V-bar), error(H-bar)]
            manoeuvre_type (str): Forced manoeuvre, defined on the database.
            t_min (float64): Minimum time allowed to execute the manoeuvre [s], standard is 1.0 second.
            t_max (float64): Maximum time allowed to execute the manoeuvre [s], standard is 10 hours.
    """

    def __init__(self):
        super(RelativeCP, self).__init__()

        self.rel_state = CartesianLVLH()
        self.error_ellipsoid = [0.0, 0.0, 0.0]
        self.manoeuvre_type = 'standard'

        # Allowed times to execute the manoeuvre
        self.t_min = 1.0
        self.t_max = 36000.0

    def set_rel_state(self, rel_state, chaser=None, target=None):
        """
            Set relative state given relative state of the checkpoint in dictionary format.

        Args:
            rel_state (Dictionary): relative state given in a dictionary.
            chaser (Chaser): Chaser state.
            target (Satellite): Target state.
        """

        # Note: "target" and "chaser" are needed because the checkpoints may be defined with respect to the chaser or
        # target state. Therefore, when the string is evaluated you need to give also the target and to have the same
        # name both in this function and in the scenario.yaml file.

        if type(rel_state) == dict:
            self.rel_state.R = np.array(rel_state['R']) if 'R' in rel_state.keys() else np.array([0.0, 0.0, 0.0])
            self.rel_state.V = np.array(rel_state['V']) if 'V' in rel_state.keys() else np.array([0.0, 0.0, 0.0])
        else:
            raise TypeError


class AbsoluteCP(CheckPoint):
    """
        Absolute checkpoint class, based on Checkpoint.
        Holds the informations for absolute navigation.
    """

    def __init__(self, checkpoint=None):
        super(AbsoluteCP, self).__init__()

        self.abs_state = KepOrbElem()

        if checkpoint is not None and checkpoint.abs_state is not None:
            self.set_abs_state(checkpoint.abs_state)

    def set_abs_state(self, abs_state, chaser=None, target=None):
        """
            Set absolute state given absolute state either in a dictionary.

        Args:
            abs_state (Dictionary or KepOrbElem): absolute state of checkpoint given either in a dictionary or in
                KepOrbElem.
            chaser (Chaser): Chaser state.
            target (Target): Target state.
        """

        # Note: "target" and "chaser" are needed because the checkpoints may be defined with respect to the chaser or
        # target state. Therefore, when the string is evaluated you need to give also the target and to have the same
        # name both in this function and in the scenario.yaml file.

        if type(abs_state) == dict:
            self.abs_state.a = eval(abs_state['a']) if 'a' in abs_state.keys() else self.abs_state.a
            self.abs_state.e = eval(abs_state['e']) if 'e' in abs_state.keys() else self.abs_state.e
            self.abs_state.i = eval(abs_state['i']) if 'i' in abs_state.keys() else self.abs_state.i
            self.abs_state.O = eval(abs_state['O']) if 'O' in abs_state.keys() else self.abs_state.O
            self.abs_state.w = eval(abs_state['w']) if 'w' in abs_state.keys() else self.abs_state.w
            self.abs_state.v = eval(abs_state['v']) if 'v' in abs_state.keys() else 0.0
        elif type(abs_state) == KepOrbElem:
            self.abs_state.a = abs_state.a
            self.abs_state.e = abs_state.e
            self.abs_state.i = abs_state.i
            self.abs_state.O = abs_state.O
            self.abs_state.w = abs_state.w
            self.abs_state.v = abs_state.v
        else:
            raise TypeError
