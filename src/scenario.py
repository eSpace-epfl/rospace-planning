import datetime as dt
import pykep as pk
import numpy as np

from space_tf import Cartesian, CartesianLVLH, KepOrbElem
from space_tf.Constants import Constants as const


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

    def start_simple_scenario(self, scenario_start_time, actual_relative_position):
        print "Creating and starting simple scenario..."

        self.start_scenario = scenario_start_time

        self.nr_hold_points = 7
        N = self.nr_hold_points

        # Set up a simple scenario with 6 hold points in km
        P0 = HoldPoint()
        P1 = HoldPoint([0, 0.060, 0], 1, 10000)
        P2 = HoldPoint([0, 0.1, 0], 2, 10000)
        P3 = HoldPoint([0, -0.1, 0], 3, 10000)
        P4 = HoldPoint([0, -5, 0], 4, 15000)
        P5 = HoldPoint([0.5, -8, 0], 5, 150000)
        P6 = HoldPoint(actual_relative_position, 6, 150000)

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

        p_next, v_next = pk.propagate_lagrangian(actual_position, actual_velocity, prop_time, const.mu_earth)

        N = self.nr_hold_points

        # Set up a more complex scenario with 8 hold points in km
        P0 = HoldPoint()
        P1 = HoldPoint([0, 0.060, 0], 1, 10000)
        P2 = HoldPoint([0, 0.1, 0], 2, 10000)
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

    def create_approach_graph(self):
        """
            Create an approach graph, depending on the number of hold lines we want to have.
            Assuming we are already in the same orbital plane.
        :param N:
        :return:
        """

        hold_lines_position = [50, 500, -500, -5000, -30000]
        hold_lines_cardinality = [1, 3, 3, 3, 5]
        hold_points = []
        id = 0
        N = len(hold_lines_cardinality)

        # Create target hold point
        P_T = HoldPoint()
        P_T.set_hold_point(0, 0, 0, 0, 0, id)
        hold_points.append(P_T)

        for n in xrange(0, N):
            hold_points_old = hold_points
            hold_points = []

            for it in xrange(0, hold_lines_cardinality[n]):
                id += 1

                HP = HoldPoint()
                HP.set_hold_point(0, 0, 0, 0, 0, id)

                for i in xrange(0, len(hold_points_old)):
                    HP.set_neighbour(hold_points_old[i])
                hold_points.append(HP)


        id += 1
        # Create chaser hold point
        P_C = HoldPoint()
        P_C.set_hold_point(0, 0, 0, 0, 0, 0, id)

        for i in xrange(0, len(hold_points)):
            P_C.set_neighbour(hold_points[i])

    def plan_front_approach(self):
        """
            In this type of scenario the target is fly-byed, to ultimately reach a position in front of
            it, where the chaser can afterwards start a slow deceleration towards the target.
        :return:
        """

        # Add last hold point before mating
        P1 = HoldPoint()
        P1.set_hold_point()
        self.hold_points.append(P1)

    def plan_scenario(self):
        # Starting from the target position and from the given keep_out_zone, do a backward planning of a certain
        # number of hold points to be reached, depending on the actual position.

        # Start by adding
        pass
