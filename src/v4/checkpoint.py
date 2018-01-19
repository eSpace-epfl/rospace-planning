

class CheckPoint:

    def __init__(self, id=0):
        self.id = id
        self.position = Position()

        self.time_dependancy = False
        self.error_ellipsoid = [0, 0, 0]