import numpy as np

from space_tf import KepOrbElem, CartesianLVLH


class CheckPoint(object):

    def __init__(self, id=0):
        self.id = id


class RelativeCP(CheckPoint):

    def __init__(self):
        super(RelativeCP, self).__init__()

        self.rel_state = CartesianLVLH()
        self.error_ellipsoid = [0.0, 0.0, 0.0]


    def set_rel_state_from_lvlh(self, lvlh, chaser, target):
        """
            Set relative state given lvlh coordinates in dictionary format.

        Args:
            lvlh (Dictionary)
            chaser (Chaser)
            target (Target)
        """

        self.rel_state.R = eval(lvlh['R']) if 'R' in lvlh.keys() else np.array([0.0, 0.0, 0.0])
        self.rel_state.V = eval(lvlh['V']) if 'V' in lvlh.keys() else np.array([0.0, 0.0, 0.0])


class AbsoluteCP(CheckPoint):

    def __init__(self, checkpoint=None):
        super(AbsoluteCP, self).__init__()

        self.abs_state = KepOrbElem()

        if checkpoint is not None and checkpoint.abs_state is not None:
            self.set_abs_state_from_kep(checkpoint.abs_state)

    def set_abs_state_from_kep(self, kep, chaser=None, target=None):
        """
            Set absolute state given keplerian coordinates dictionary.
            Note: chaser and target are there for the eval() function in which they may appear.

        Args:
            kep (Dictionary or KepOrbElem)
            chaser (Chaser)
            target (Target)
        """

        if type(kep) == dict:
            self.abs_state.a = eval(kep['a']) if 'a' in kep.keys() else self.abs_state.a
            self.abs_state.e = eval(kep['e']) if 'e' in kep.keys() else self.abs_state.e
            self.abs_state.i = eval(kep['i']) if 'i' in kep.keys() else self.abs_state.i
            self.abs_state.O = eval(kep['O']) if 'O' in kep.keys() else self.abs_state.O
            self.abs_state.w = eval(kep['w']) if 'w' in kep.keys() else self.abs_state.w
            self.abs_state.v = eval(kep['v']) if 'v' in kep.keys() else 0.0

        elif type(kep) == KepOrbElem:
            self.abs_state.a = kep.a
            self.abs_state.e = kep.e
            self.abs_state.i = kep.i
            self.abs_state.O = kep.O
            self.abs_state.w = kep.w
            self.abs_state.v = kep.v
