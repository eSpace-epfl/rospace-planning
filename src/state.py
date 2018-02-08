import numpy as np

from space_tf import KepOrbElem, CartesianLVLH, Cartesian


class Satellite(object):

    def __init__(self):
        self.abs_state = KepOrbElem()

    def set_from_other_satellite(self, satellite):
        """
            Create a copy of the given satellite.

        Args:
            satellite (Chaser, Target, Satellite)
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


class Chaser(Satellite):

    def __init__(self):
        super(Chaser, self).__init__()

        self.rel_state = CartesianLVLH()

    def set_rel_state_from_abs_state(self, target):
        """
            Set lvlh coordinates given target absolute state.

        Args:
            target (Target)
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
             target (Target)
        """

        target_cart = Cartesian()
        chaser_cart = Cartesian()

        target_cart.from_keporb(target.abs_state)
        chaser_cart.from_lvlh_frame(target_cart, self.rel_state)

        self.abs_state.from_cartesian(chaser_cart)

    def set_abs_state_from_kep(self, kep, target):
        """
            Given keplerian orbital elements set the absolute state.

        Args:
            kep (Dictionary): Keplerian orbital elements stored in a dictionary.
            target (Target): Target position may be needed for the eval() functions.
        """

        self.abs_state.a = eval(str(kep['a']))
        self.abs_state.e = eval(str(kep['e']))
        self.abs_state.i = eval(str(kep['i']))
        self.abs_state.O = eval(str(kep['O']))
        self.abs_state.w = eval(str(kep['w']))
        self.abs_state.v = eval(str(kep['v']))


class Target(Satellite):

    def __init__(self):
        super(Target, self).__init__()

    def set_abs_state_from_tle(self, tle):
        """
            Given tle dictionary set the absolute state.

        Args:
            tle (Dictionary): TLE stored in a dictionary.
        """

        self.abs_state.from_tle(eval(str(tle['i'])),
                                eval(str(tle['O'])),
                                eval(str(tle['e'])),
                                eval(str(tle['m'])),
                                eval(str(tle['w'])),
                                eval(str(tle['n'])))
