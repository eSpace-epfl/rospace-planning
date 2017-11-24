from numpy import pi, sqrt, sin, cos, arccos, tan, arctan, dot, array
from numpy.linalg import norm
from space_tf.Constants import Constants as const

import pykep as pk

mu = const.mu_earth


class OptimalTime:

    def __init__(self):
        self.t_opt = 0

    def eccentric_anomaly(self, e, theta):
        return 2.0 * arctan(sqrt((1-e)/(1+e)) * tan(theta/2.0))

    def mean_anomaly(self, e, E):
        return E - e * sin(E)

    def find_time_optimal_trajectory(self, r1, r2):
        r1_mag = norm(r1)
        r2_mag = norm(r2)

        r1_normed = r1 / r1_mag
        r2_normed = r2 / r2_mag

        alpha = arccos(dot(r1_normed, r2_normed))

        a_min = 1e12
        theta_min = 0
        e_min = 0
        for t in xrange(0, 1000):

            theta = 2*pi/1000 * t

            e = (r2_mag - r1_mag)/(r1_mag*cos(theta) - r2_mag*cos(theta+alpha))

            if 0 <= e < 1:
                p = r1_mag * (1 + e * cos(theta))
                a = p/(1 - e**2)

                if a < a_min:
                    a_min = a
                    theta_min = theta
                    e_min = e

        E1 = self.eccentric_anomaly(e_min, theta_min)
        E2 = self.eccentric_anomaly(e_min, theta_min + alpha)

        M1 = self.mean_anomaly(e_min, E1)
        M2 = self.mean_anomaly(e_min, E2)

        dt = abs(M1-M2) * sqrt(a_min**3/mu)

        return {'a': a_min, 'theta': theta_min, 'e': e_min, 'dt_opt': dt}

    def find_optimal_trajectory_time(self, r1, r2, v2, max_time):

        r2_t1 = r2
        v2_t1 = v2

        diff = 1e12
        t_opt = 0
        for t in xrange(1, max_time):
            r2_t2, v2_t2 = pk.propagate_lagrangian(r2_t1, v2_t1, t, mu)

            t_ = self.find_time_optimal_trajectory(r1, r2_t2)['dt_opt']

            err = abs(t_ - t)
            if err < diff:
                diff = err
                t_opt = t
                if err < 10:
                    break

        return t_opt




# r1 = array([7000, 0, 0])
# r2 = array([-4499.9999999999982, -7794.2286340599485, 0])
# v2 = array([7.50586*cos(pi/3), 6.8586*sin(pi/3), 0])
#
# t_opt = find_optimal_trajectory_time(r1, r2, v2, 60000)
# print t_opt
#
# r2_prop = pk.propagate_lagrangian(r2,v2,7104,mu)

# print pk.lambert_problem(r1,r2,7104,mu)
#
# r2_t1 = r2
# v2_t1 = v2
#
# diff = 1e12
# t_opt = 0
#
# for t in xrange(1, max_time):
#     r2_t2, v2_t2 = pk.propagate_lagrangian(r2_t1, v2_t1, t, mu)
#
#     t_ = find_time_optimal_trajectory(r1, r2_t2)['dt_opt']
#
#     err = abs(t_ - t)
#     if err < diff:
#         diff = err
#         t_opt = t

    def find_optimal_trajectory_time_for_scenario(self, target, chaser_next, chaser, max_time):

        p_C_TEM_t0 = chaser.cartesian.R

        p_T_TEM_t0 = target.cartesian.R
        v_T_TEM_t0 = target.cartesian.V

        diff = 1e12
        t_opt = 0
        for t in xrange(10, max_time):
            # Propagate target position at t = t0 + dt
            p_T_TEM_t1, v_T_TEM_t1 = pk.propagate_lagrangian(p_T_TEM_t0, v_T_TEM_t0, t, mu)
            target.cartesian.R = p_T_TEM_t1
            target.cartesian.V = v_T_TEM_t1

            # Now that the target is propagated, we can calculate absolute position of the chaser from its relative
            # This is the position he will have at time t = t0 + dt
            chaser_next.cartesian.from_lhlv_frame(target.cartesian, chaser_next.lvlh)

            p_C_TEM_t1 = chaser_next.cartesian.R
            v_C_TEM_t1 = chaser_next.cartesian.V

            t_ = self.find_time_optimal_trajectory(p_C_TEM_t0, p_C_TEM_t1)['dt_opt']

            err = abs(t_ - t)
            if err < diff:
                diff = err
                t_opt = t
                if err < 1:
                    break

        self.t_opt = t_opt