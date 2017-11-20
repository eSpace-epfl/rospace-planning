import PyKEP as pk
import numpy as np
import rospy
import datetime as dt
import epoch_clock

from scenario import Scenario
from space_tf import *
from space_tf.Constants import Constants as const

epoch = epoch_clock.Epoch()

class PathOptimizer:

    def __init__(self):
        # Initialize the optimizer
        self.kep_target = KepOrbElem()
        self.kep_chaser = KepOrbElem()

        self.cart_target = Cartesian()
        self.cart_chaser = Cartesian()

        self.deltaV = np.array([0, 0, 0])
        self.estimated_distance = 0
        self.impulse_start_time = rospy.Time.now()

        # Timing variables
        self.sleep_flag = False
        self.rate = 0.1                 # Rate of simulation increases as we approach the target
        self.manoeuvre_start = dt.datetime(2017, 9, 15, 12, 20, 0)
        self._manoeuvre_idle_seconds = 120

        # Scenario variable
        self.scenario = None
        self.scenario_flag = False

    def set_manoeuvre_start(self, year, month, day, hour, minute, seconds):
        self.manoeuvre_start = dt.datetime(year,month,day,hour,minute,seconds)

    def find_optimal_path(self, msg):
        # Find optimal path
        print "Searching for optimal path"

    def callback(self, target_oe, chaser_oe):
        # Update orbital elements value
        self.kep_target.from_message(target_oe.position)
        self.cart_target.from_keporb(self.kep_target)

        self.kep_chaser.from_message(chaser_oe.position)
        self.cart_chaser.from_keporb(self.kep_chaser)

        self.evaluate_estimated_distance(self.cart_target.R, self.cart_chaser.R)

        # If scenario has not been set up yet
        if not self.scenario_flag:
            self.scenario = Scenario()
            self.scenario.start_simple_scenario(self.manoeuvre_start)
            self.scenario_flag = True

            # Solve right now lambert problem for this scenario
        else:
            if self.manoeuvre_start <= epoch.now() and not self.sleep_flag:
                pass




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

    def print_possible_solutions(self, sol, N=5):
        print sol
        for i in xrange(0, N+1):
            print "Solution allowing " + str(i) + " waiting orbits."
            dV_tot = np.array(sol.get_v1()[i]) + np.array(sol.get_v2()[i])
            print "---> Total deltaV needed (norm):   " + str(dV_tot)
            print "---> First impulse deltaV (norm):  " + str(np.array(sol.get_v1()[i]))
            print "---> Second impulse deltaV (norm): " + str(np.array(sol.get_v2()[i]))
        rospy.sleep(300)

    def simple_pykep_solution(self, r_t1, v_t1, r_c1, v_c1, t_start, t_rdv):
        # t_start is the "distance" from the callback at which the problem has to be solved.
        # t_rdv is the actual maximum allowed rendezvous duration (should compute the maximum possible for the minimum
        # consumption)

        mu = const.mu_earth

        r_c_start, v_c_start = pk.propagate_lagrangian(r_c1, v_c1, t_start, mu)
        r_t2, v_t2 = pk.propagate_lagrangian(r_t1, v_t1, t_rdv + t_start, mu)

        return pk.lambert_problem(r_c_start, r_t2, t_rdv, mu, False, 5)

    def solve_optimization(self, r_c1, v_c1, t_rdv, t_start):
        mu = Constants.mu_earth
        opt = OptimizationProblem()

        obj = lambda x: [x[0].get_v1()[i] + x[0].get_v2()[i] + x[1].get_v1()[i] + x[1].get_v2()[i] + x[2].get_v1()[i] + x[2].get_v2()[i] for i in range(1,3)]
        opt.init_objective_function(lambda x: x[0].get_v1()[0] + x[1] + x[2])

        r_c_start, v_c_start = pk.propagate_lagrangian(r_c1, v_c1, t_start, mu)

    def evaluate_estimated_distance(self, r_target, r_chaser):
        self.estimated_distance = np.linalg.norm(r_target - r_chaser)

    def propagate_estimated_distance(self, r_target, r_chaser, propagation_time):
        pass

    def multi_lambert_for_scenario(self):
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

        r_c_start, v_c_start = pk.propagate_lagrangian(self.cart_chaser.R, self.cart_chaser.V, t_prop, mu)
        s.hold_points[-1].R_abs = r_c_start
        s.hold_points[-1].V_abs = v_c_start

        # Extract first hold point, a (possible) future point that will be reached when the manoeuvre is set to start
        actual_HP = s.hold_points.pop()
        while len(s.hold_points) > 0:
            # Calculate R & V for the next hold point


            sol = pk.lambert_problem()



            actual_HP = s.hold_points.pop()
            pass