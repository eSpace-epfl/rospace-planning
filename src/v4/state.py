

class State:

    def __init__(self):
        self.kep = KepOrbElem()
        self.lvlh = CartesianLVLH()

    def from_other_state(self, state):

        self.lvlh.R = state.lvlh.R
        self.lvlh.V = state.lvlh.V

        self.kep.a = state.kep.a
        self.kep.e = state.kep.e
        self.kep.i = state.kep.i
        self.kep.O = state.kep.O
        self.kep.w = state.kep.w
        self.kep.v = state.kep.v