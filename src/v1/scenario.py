import datetime as dt
import pykep as pk
import numpy as np

from space_tf import Cartesian, CartesianLVLH, KepOrbElem, mu_earth, QNSRelOrbElements


class HoldPoint:

    def __init__(self, pos=[0, 0, 0], id=0, exec_time=0):
        self.tolerance = [0, 0, 0]

        self.cartesian = Cartesian()
        self.lvlh = CartesianLVLH()
        self.kep = KepOrbElem()

        self.lvlh.R = pos

        # Maximum allowed time for a manoeuvre from this hold point to the next
        self.execution_time = exec_time
        self.id = id

        self.next_hold_points = []
        self.next_cost = 0
        self.next_time = 0

        self.type = type

    def set_hold_point(self, rel_pos, abs_pos, tol, time, next, id):
        self.id = id

    def set_neighbour(self, next_hp):
        self.next_hold_points.append(next_hp)

    def set_execution_time(self, exec_time):
        self.execution_time = exec_time


class Scenario:

    def __init__(self):

        # Scenario information
        self.nr_hold_points = 0
        self.keep_out_zone = 0.05
        self.hold_points = []
        self.start_scenario = dt.datetime(2017, 9, 15, 12, 20, 0)

        self.positions = []

    def start_simple_scenario(self, scenario_start_time, actual_epoch, actual_position, actual_velocity):
        print "Creating and starting simple scenario..."

        self.start_scenario = scenario_start_time

        self.nr_hold_points = 7
        N = self.nr_hold_points

        # Propagate the actual position to the time the scenario will start
        prop_time = scenario_start_time - actual_epoch
        prop_time = prop_time.seconds

        p_next, v_next = pk.propagate_lagrangian(actual_position, actual_velocity, prop_time, mu_earth)

        # Set up a simple scenario with 6 hold points in km
        P0 = HoldPoint()
        P1 = HoldPoint([0, 0.060, 0], 1, 10000)
        P2 = HoldPoint([0, 0.1, 0], 2, 10000)
        P3 = HoldPoint([0, -0.1, 0], 3, 10000)
        P4 = HoldPoint([0, -5, 0], 4, 15000)
        P5 = HoldPoint([0.5, -8, 0], 5, 86000)
        P6 = HoldPoint(id=6, exec_time=86000)
        P6.cartesian.R = np.array(p_next)
        P6.cartesian.V = np.array(v_next)

        for i in xrange(0,N):
            self.hold_points.append(eval('P' + str(i)))
            if i > 0:
                eval('P' + str(i) + '.set_neighbour(P' + str(i-1) + ')')
                #eval('P' + str(i) + '.set_execution_time(' + str(max_time_per_manoeuvre) + ')')

    def start_complex_scenario(self, scenario_start_time, actual_epoch, actual_position, actual_velocity):
        print "Creating and starting complex scenario..."

        self.start_scenario = scenario_start_time
        self.nr_hold_points = 8

        # Propagate the actual position to the time the scenario will start
        prop_time = scenario_start_time - actual_epoch
        prop_time = prop_time.seconds

        p_next, v_next = pk.propagate_lagrangian(actual_position, actual_velocity, prop_time, mu_earth)

        N = self.nr_hold_points

        # Set up a more complex scenario with 8 hold points in km
        S0 = HoldPoint([-4, 10, 0], 0, 60000)
        S1 = HoldPoint([0, 0.060, 0], 1, 10000)
        S2 = HoldPoint([0, 0.1, 0], 2, 10000)
        P3 = HoldPoint([0, -0.1, 0], 3, 10000)
        P4 = HoldPoint([0, -5, 0], 4, 15000)
        P5 = HoldPoint([0.5, -8, 0], 5, 86400)
        P6 = HoldPoint([3, -10, 0], 6, 86400)
        P7 = HoldPoint(id=7, exec_time=86400)
        P7.cartesian.R = np.array(p_next)
        P7.cartesian.V = np.array(v_next)

        for i in xrange(0, N):
            self.hold_points.append(eval('P' + str(i)))
            if i > 0:
                eval('P' + str(i) + '.set_neighbour(P' + str(i-1) + ')')
                #eval('P' + str(i) + '.set_execution_time(' + str(max_time_per_manoeuvre) + ')')
