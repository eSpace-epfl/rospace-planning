import datetime as dt
import pykep as pk
import epoch_clock

from space_tf.Constants import Constants as const


class HoldPoint:

    def __init__(self, rel_pos=[0, 0, 0], id=0, exec_time=0):
        self.relative_position = rel_pos
        self.absolute_position = [0, 0, 0]
        self.relative_velocity = [0, 0, 0]

        self.tolerance = [0, 0, 0]
        self.R_abs = [0, 0, 0]
        self.V_abs = [0, 0, 0]

        # Maximum allowed time for a manoeuvre from this hold point to the next
        self.execution_time = exec_time
        self.hold_position = True
        self.id = id
        self.relative_semimajor_axis = 0

        self.next_hold_points = []
        self.next_cost = 0
        self.next_time = 0

    def set_hold_point(self, rel_pos, abs_pos, tol, time, next, id):
        self.id = id

    def set_neighbour(self, next_hp):
        self.next_hold_points.append(next_hp)

    def set_execution_time(self, exec_time):
        self.execution_time = exec_time

    def update_hold_point(self):
        # When we are in an hold point, update it with the "real" position
        pass


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

        max_time_per_manoeuvre = 30000

        self.nr_hold_points = 7
        N = self.nr_hold_points

        # Set up a simple scenario with 6 hold points in km
        P0 = HoldPoint()
        P1 = HoldPoint([0, 0.060, 0], 1, 6000)
        P2 = HoldPoint([0, 0.1, 0], 2, 6000)
        P3 = HoldPoint([0, -0.1, 0], 3, 6000)
        P4 = HoldPoint([0, -5, 0], 4, 15000)
        P5 = HoldPoint([0.5, -8, 0], 5, 30000)
        P6 = HoldPoint(actual_relative_position, 6, 30000)

        for i in xrange(0,N):
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
