import pykep as pk
import numpy as np
import rospy
import datetime as dt
import epoch_clock

from scenario import Scenario
from solver import Solver
from space_tf import *
from space_tf.Constants import Constants as const

epoch = epoch_clock.Epoch()

class Command:

    def __init__(self):
        self.deltaV_TEM = [0, 0, 0]
        self.execution_time = dt.datetime(2017, 9, 15, 12, 20, 0)

    def set_deltaV(self, deltaV):
        self.deltaV_TEM = deltaV

    def set_execution_time(self, datetime):
        self.execution_time = datetime

class PathOptimizer:

    def __init__(self):
        # Initialize the optimizer
        # Create keplerian elements for target and chaser
        self.kep_target = KepOrbElem()
        self.kep_chaser = KepOrbElem()

        # Create cartesian element for target and chaser
        self.cart_target = Cartesian()
        self.cart_chaser = Cartesian()

        # Create propagated cartesian elements for target and chaser
        self.cart_target_prop = Cartesian()
        self.cart_chaser_prop = Cartesian()

        self.deltaV = np.array([0, 0, 0])
        self.estimated_distance = 0
        self.impulse_start_time = rospy.Time.now()

        # Timing variables
        self.sleep_flag = False
        self.rate = 0.1                 # Rate of simulation increases as we approach the target
        self.manoeuvre_start = dt.datetime(2017, 9, 15, 12, 20, 0)
        self._manoeuvre_idle_seconds = 120

        # Scenario variables
        self.scenario = None
        self.scenario_flag = False

        # Solver variables
        self.solver = Solver()

        # Define a list of "Commands"
        self.commands = []
        self.active_command = Command()

    def set_manoeuvre_start(self, year, month, day, hour, minute, seconds):
        self.manoeuvre_start = dt.datetime(year, month, day, hour, minute, seconds)

    def callback(self, target_oe, chaser_oe):
        # Update orbital elements value
        self.kep_target.from_message(target_oe.position)
        self.cart_target.from_keporb(self.kep_target)

        self.kep_chaser.from_message(chaser_oe.position)
        self.cart_chaser.from_keporb(self.kep_chaser)

        # If scenario has not been set up yet
        if not self.scenario_flag:
            self.scenario = Scenario()
            actual_rel_pos = self.cart_chaser.R - self.cart_target.R
            self.scenario.start_simple_scenario(self.manoeuvre_start, actual_rel_pos)
            self.scenario_flag = True

            # Solve right now lambert problem for this scenario
            self.solve_scenario()
            self.active_command = self.commands.pop(0)
        else:
            # Scenario planned and solved, now give outputs at the right time
            if self.active_command.execution_time <= epoch.now() and len(self.commands) > 0:
                self.sleep_flag = True
                self.active_command = self.commands.pop(0)

        # if self.manoeuvre_start <= epoch.now() and not self.sleep_flag:
        #     print "Solving the optimization problem. Thrusters will start at: " + \
        #           str(dt.timedelta(0, 3600*self.manoeuvre_start.time().hour + 60*self.manoeuvre_start.time().minute +
        #           self.manoeuvre_start.time().second) + dt.timedelta(0, self._manoeuvre_idle_seconds))
        #
        #     sol = self.simple_pykep_solution(self.cart_target.R, self.cart_target.V,
        #                                      self.cart_chaser.R, self.cart_chaser.V,
        #                                      self._manoeuvre_idle_seconds, 100000)
        #
        #     self.print_possible_solutions(sol)
        #
        #     self.deltaV = sol.get_v1()[0]
        #     self.sleep_flag = True
        #
        #     print "Setting next manoeuvre start..."
        #     # Theoritically set the next maneuvre to the time the previous one is completed + idle time
        #     self.set_manoeuvre_start(2017, 9, 15, 13, 40, 00)

    def solve_scenario(self):
        print "Solving the scenario starting at: " + str(self.manoeuvre_start)

        # Solve the Lamber Problem between points of a scenario
        # Extract scenario
        s = self.scenario

        # Extract needed constants
        mu = const.mu_earth

        # TODO: Eventually update again the position of the chaser exactly before performing the propagation

        # Define the first point when the manoeuvre start
        t_epoch = epoch.now()       # Epoch clock

        # Propagation time is the difference between epoch clock and manoeuvre start time
        t_prop = self.manoeuvre_start - t_epoch
        t_prop = t_prop.total_seconds()

        # TODO: Review time integration to reduce as more as possible uncertainty due to calculation time

        # Extract first hold point, a (possible) future point that will be reached when the manoeuvre is set to start
        chaser = s.hold_points.pop()

        # Extract target
        target = s.hold_points.pop(0)

        # Propagate actual position in TEME to the moment in time when the maneouvre will start, and set chaser according to that
        r_c_prop, v_c_prop = \
            pk.propagate_lagrangian(self.cart_chaser.R, self.cart_chaser.V, t_prop, mu)
        r_t_prop, v_t_prop = \
            pk.propagate_lagrangian(self.cart_target.R, self.cart_target.V, t_prop, mu)

        # Set absolute position at t = t_start
        chaser.cartesian.R = np.array(r_c_prop)
        chaser.cartesian.V = np.array(v_c_prop)
        target.cartesian.R = np.array(r_t_prop)
        target.cartesian.V = np.array(v_t_prop)

        # Set relative position at t = t_start
        chaser.lvlh.from_cartesian_pair(chaser.cartesian, target.cartesian)

        # TODO: Consider that when we arrive at the node we will have for sure a different position / velocity.
        # -> Think about what to do to correct for that

        # TODO: Consider that in each HP we may end up waiting for a certain idle time
        tot_time = 0
        tot_dV = 0

        old_delta_V2 = np.array([0, 0, 0])

        while len(s.hold_points) > 0:
            # Calculate R & V for the next hold point

            # Calculate a for the chaser orbit in the actual HP
            # h = np.cross(chaser.R_abs, chaser.V_abs)
            # h_norm = np.dot(h, h)
            # e = 1.0/mu * (np.cross(chaser.V_abs, h) - mu * chaser.R_abs / np.linalg.norm(chaser.R_abs))
            # a_c = (h_norm/mu) / (1 - np.linalg.norm(e)**2)

            # Extract next hold point and assign to it it's absolute speed and position at t = t0
            chaser_next = s.hold_points[-1]

            # TODO: Note that now the implementation assume that after every step we want to end with 0 relative velocity => we may be doing some useless burns

            # self.solver.clohessy_wiltshire_solver(a_c, chaser.relative_position, chaser.relative_velocity,
            #                                       chaser.execution_time, chaser_next.relative_position, chaser_next.relative_velocity,
            #                                       chaser.id, s.keep_out_zone)

            self.solver.multi_lambert_solver(chaser, chaser_next, target)

            # r_c_prop, v_c_prop = pk.propagate_lagrangian(chaser.R_abs, chaser.V_abs + self.solver.sol['deltaV_1'],
            #                                              self.solver.sol['deltaT'], mu)
            #
            # # Propagate actual position in TEME to the moment in time when the maneouvre will start
            # r_c_prop, v_c_prop = \
            #     pk.propagate_lagrangian(self.cart_chaser_prop.R, self.cart_chaser_prop.V + self.solver.sol['deltaV_1'], self.solver.cw_sol['deltaT'], mu)
            # r_t_prop, v_t_prop = \
            #     pk.propagate_lagrangian(self.cart_target_prop.R, self.cart_target_prop.V, self.solver.sol['deltaT'], mu)
            #
            # self.cart_chaser_prop.R = np.array(r_c_prop)
            # self.cart_chaser_prop.V = np.array(v_c_prop)
            # self.cart_target_prop.R = np.array(r_t_prop)
            # self.cart_target_prop.V = np.array(v_t_prop)
            #
            # chaser_next.R_abs = self.cart_chaser_prop.R
            # chaser_next.V_abs = self.cart_chaser_prop.V

            c = Command()
            c.set_deltaV(self.solver.sol['deltaV_1'] + old_delta_V2)
            c.set_execution_time(self.manoeuvre_start + dt.timedelta(0, tot_time))
            self.commands.append(c)

            old_delta_V2 = self.solver.sol['deltaV_2']
            tot_time += self.solver.sol['deltaT']
            tot_dV += self.solver.sol['deltaV']
            print "Needed time to perform the manoeuvre:          " + str(self.solver.sol['deltaT'])
            print "Needed deltaV to perform the manoeuvre:        " + str(self.solver.sol['deltaV_1'])
            print "Needed deltaV to brake in the wanted position: " + str(self.solver.sol['deltaV_2'])
            print ""

            chaser = s.hold_points.pop()

        print "----------------- Manoeuvre elaborated --------------------"
        print "--> Start time:            " + str(self.manoeuvre_start)
        print "--> Manoeuvre duration:    " + str(tot_time) + " seconds"
        print "--> Total deltaV:          " + str(tot_dV) + " km/s"


    # OLD
    def solve_optimization(self, r_c1, v_c1, t_rdv, t_start):
        mu = Constants.mu_earth
        opt = OptimizationProblem()

        obj = lambda x: [
            x[0].get_v1()[i] + x[0].get_v2()[i] + x[1].get_v1()[i] + x[1].get_v2()[i] + x[2].get_v1()[i] +
            x[2].get_v2()[i] for i in range(1, 3)]
        opt.init_objective_function(lambda x: x[0].get_v1()[0] + x[1] + x[2])

        r_c_start, v_c_start = pk.propagate_lagrangian(r_c1, v_c1, t_start, mu)
