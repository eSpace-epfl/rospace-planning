from math import sqrt,pi,atan,tan,sin,cos
from space_tf import *
import PyKEP as pk
import numpy as np
import rospy
# from optimization_problem import OptimizationProblem
import datetime as dt
import epoch_clock

epoch = epoch_clock.Epoch()

class PathOptimizer:

    def __init__(self):
        # Initialize the optimizer
        self.kep_target = KepOrbElem()
        self.kep_chaser = KepOrbElem()

        self.cart_target = Cartesian()
        self.cart_chaser = Cartesian()

        self.deltaV = np.array([0, 0, 0])
        self.impulse_start_time = rospy.Time.now()

        # Timing variables
        self.sleep_flag = False
        self.rate = 0.1                 # Rate of simulation increases as we approach the target
        self.manoeuvre_start = dt.datetime(2017, 9, 15, 12, 20, 0)
        self._manoeuvre_idle_seconds = 120

    def set_manoeuvre_start(self, year, month, day, hour, minute, seconds):
        self.manoeuvre_start = dt.datetime(year,month,day,hour,minute,seconds)

    def find_optimal_path(self, msg):
        # Find optimal path
        print "Searching for optimal path"

    def callback(self, target_oe, chaser_oe):
        if self.manoeuvre_start <= epoch.now() and not self.sleep_flag:
            print "Solving the optimization problem. Thrusters will start at: " + \
                  str(dt.timedelta(0, 3600*self.manoeuvre_start.time().hour + 60*self.manoeuvre_start.time().minute +
                  self.manoeuvre_start.time().second) + dt.timedelta(0, self._manoeuvre_idle_seconds))

            # Update orbital elements value
            self.kep_target.from_message(target_oe.position)
            self.cart_target.from_keporb(self.kep_target)

            self.kep_chaser.from_message(chaser_oe.position)
            self.cart_chaser.from_keporb(self.kep_chaser)

            sol = self.simple_pykep_solution(self.cart_target.R, self.cart_target.V,
                                             self.cart_chaser.R, self.cart_chaser.V,
                                             self._manoeuvre_idle_seconds, 100000)

            self.print_possible_solutions(sol)

            self.deltaV = sol.get_v1()[0]
            self.sleep_flag = True

            print "Setting next manoeuvre start..."
            # Theoritically set the next maneuvre to the time the previous one is completed + idle time
            self.set_manoeuvre_start(2017, 9, 15, 13, 40, 00)


    def print_possible_solutions(self, sol, N=5):
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

        mu = Constants.mu_earth

        r_c_start, v_c_start = pk.propagate_lagrangian(r_c1, v_c1, t_start, mu)
        r_t2, v_t2 = pk.propagate_lagrangian(r_t1, v_t1, t_rdv + t_start, mu)

        return pk.lambert_problem(r_c_start, r_t2, t_rdv, mu, False, 5)

    def solve_optimization(self, r_c1, v_c1, t_rdv, t_start):
        mu = Constants.mu_earth
        opt = OptimizationProblem()

        obj = lambda x: [x[0].get_v1()[i] + x[0].get_v2()[i] + x[1].get_v1()[i] + x[1].get_v2()[i] + x[2].get_v1()[i] + x[2].get_v2()[i] for i in range(1,3)]
        opt.init_objective_function(lambda x: x[0].get_v1()[0] + x[1] + x[2])

        r_c_start, v_c_start = pk.propagate_lagrangian(r_c1, v_c1, t_start, mu)


