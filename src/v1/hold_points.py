import datetime as dt

class HoldPoint:

    def __init__(self):
        self.relative_position = [0, 0, 0]
        self.absolute_position = [0, 0, 0]
        self.tolerance = [0, 0, 0]
        self.execution_time = dt.datetime(2017, 9, 15, 12, 20, 0)
        self.hold_position = True
        self.id = 0

        self.next_hold_points = []
        self.next_cost = 0
        self.next_time = 0


        # Understand hold point type



    def set_hold_point(self, rel_pos, abs_pos, tol, time, next, id):
        self.id = id

    def set_neighbour(self, next):
        self.next_hold_points.append(next)
