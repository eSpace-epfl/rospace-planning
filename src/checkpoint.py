from state import State

class CheckPoint(object):

    def __init__(self, id=0):
        self.id = id
        self.state = State()

        self.time_dependancy = False
        self.error_ellipsoid = [0, 0, 0]