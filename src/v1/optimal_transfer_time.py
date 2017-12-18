from numpy import pi, sqrt, sin, cos, arccos, tan, arctan, dot, cross, array
from numpy.linalg import norm
# from space_tf import mu_earth

import pykep as pk


class OptimalTime:

    def __init__(self):
        self.t_opt = 0

    def eccentric_anomaly(self, e, theta):
        return 2.0 * arctan(sqrt((1-e)/(1+e)) * tan(theta/2.0))

    def mean_anomaly(self, e, E):
        return E - e * sin(E)

    def find_optimal_position(self, a1, e1, a2, e2):
        # Given data of the two orbits, find the optimal transfer position to minimize the delta-v

        mu = 398600.4

        p1 = a1 - (a1 * e1) * e1
        p2 = a2 - (a2 * e2) * e2

        h1 = sqrt(p1 * mu)
        h2 = sqrt(p2 * mu)

        deltav_min = 1e12
        a_min = 0
        e_min = 0
        theta1_min = 0
        theta2_min = 0

        angle_samples = 1e4
        for i in xrange(0, int(angle_samples)):
            theta1 = 2*pi/angle_samples * i
            for j in xrange(0, int(angle_samples)):
                theta2 = 2*pi/angle_samples * j

                r1 = p1/(1 + e1 * cos(theta1))
                r2 = p2/(1 + e2 * cos(theta2))

                # Evaluate the eccentricity of the trajectory that connects the two points
                # Assuming that also the new trajectory will have the periapsis aligned, as changing it
                # will lead into a bigger consume of delta-v.
                e = (r2 - r1) / (r1 * cos(theta2) - r2 * cos(theta1))

                # If the path is an ellipse, it may be optimal
                if 0 <= e < 1:
                    p_traj = r1 * (1 + e * cos(theta1))
                    a_traj = p_traj / (1 - e**2)
                    h_traj = sqrt(mu * p_traj)

                    deltav_1 = mu/h_traj * array([-sin(theta1), e + cos(theta1), 0]) - \
                               mu/h1 * array([-sin(theta1), e + cos(theta1), 0])

                    deltav_2 = mu/h_traj * array([-sin(theta2), e + cos(theta2), 0]) - \
                               mu/h2 * array([-sin(theta2), e + cos(theta2), 0])

                    deltav_tot = norm(deltav_1) + norm(deltav_2)

                    # Minize a to minimize delta-v
                    if deltav_min > deltav_tot:
                        a_min = a_traj
                        e_min = e
                        theta1_min = theta1
                        theta2_min = theta2
                        deltav_min = deltav_tot
                else:
                    # Move on
                    break

        return {'a': a_min, 'theta1': theta1_min, 'theta2': theta2_min, 'e': e_min, 'deltav': deltav_min}

    def find_time_optimal_trajectory(self, r1, v1, r2):
        h1 = cross(r1, v1)
        h1 = h1 / norm(h1)

        r1_mag = norm(r1)
        r2_mag = norm(r2)

        r1_normed = r1 / r1_mag
        r2_normed = r2 / r2_mag

        switch = cross(r1, r2)
        alpha = arccos(dot(r1_normed, r2_normed))

        if switch[2] * h1[2] <= 0:
            alpha = 2*pi - alpha

        a_min = 1e12
        theta_min = 0
        e_min = 0

        # TODO: Set accuracy to be dependant from the distance from the target
        # Angle samples define the distance of the point considered, it can be approximated as:
        # -->    2*pi/angle_samples * radius
        # Therefore, for a case of r = 7300km => d = 0.471km => ca. 0.06 of a second of travel
        # To achieve an occuracy in the order of meters, the samples should be around:
        # -->    angle_samples = 1e6
        # With a angle_samples of 1e4:
        # -->    d = 4.6km => ca. 1/2 second of travel (round orbit)

        # Solution considering that orbit 1 and 2 lies on the same periapsis

        angle_samples = 1e4
        for t in xrange(0, int(angle_samples)):
            theta = 2*pi/angle_samples * t
            e = (r2_mag - r1_mag)/(r1_mag*cos(theta) - r2_mag*cos(theta+alpha))

            if 0 <= e < 1:
                # Consider only elliptical trajectories
                p = r1_mag * (1 + e * cos(theta))
                a = p/(1.0 - e**2)

                if a < a_min:
                    a_min = a
                    theta_min = theta
                    e_min = e

        E1 = self.eccentric_anomaly(e_min, theta_min)
        E2 = self.eccentric_anomaly(e_min, theta_min + alpha)

        M1 = self.mean_anomaly(e_min, E1)
        M2 = self.mean_anomaly(e_min, E2)

        dt = abs(M1 - M2) * sqrt(a_min**3/mu_earth)

        return {'a': a_min, 'theta': theta_min, 'e': e_min, 'dt_opt': dt}

    def find_optimal_trajectory_time(self, r1, r2, v2, max_time):

        r2_t1 = r2
        v2_t1 = v2

        diff = 1e12
        t_opt = 0
        for t in xrange(1, max_time):
            r2_t2, v2_t2 = pk.propagate_lagrangian(r2_t1, v2_t1, t, mu_earth)

            t_ = self.find_time_optimal_trajectory(r1, r2_t2)['dt_opt']

            err = abs(t_ - t)
            if err < diff:
                diff = err
                t_opt = t
                if err < 1:
                    break

        return t_opt

    def find_optimal_trajectory_time_for_scenario(self, target, chaser_next, chaser):

        p_C_TEM_t0 = chaser.cartesian.R
        v_C_TEM_t0 = chaser.cartesian.V

        p_T_TEM_t0 = target.cartesian.R
        v_T_TEM_t0 = target.cartesian.V

        diff = 1e12
        t_opt = 0
        for t in xrange(0, chaser.execution_time):
            # Propagate target position at t = t0 + dt
            p_T_TEM_t1, v_T_TEM_t1 = pk.propagate_lagrangian(p_T_TEM_t0, v_T_TEM_t0, t, mu_earth)
            target.cartesian.R = p_T_TEM_t1
            target.cartesian.V = v_T_TEM_t1

            # Now that the target is propagated, we can calculate absolute position of the chaser from its relative
            # This is the position he will have at time t = t0 + dt
            chaser_next.cartesian.from_lvlh_frame(target.cartesian, chaser_next.lvlh)

            p_C_TEM_t1 = chaser_next.cartesian.R
            v_C_TEM_t1 = chaser_next.cartesian.V

            t_ = self.find_time_optimal_trajectory(p_C_TEM_t0, v_C_TEM_t0, p_C_TEM_t1)['dt_opt']

            err = abs(t_ - t)
            if err < diff:
                diff = err
                t_opt = t
                if err < 10:
                    break

        self.t_opt = t_opt



# a1 = 6000
# a2 = 9000
# e1 = 0.0
# e2 = 0.0
#
#
# opt = OptimalTime()
# sol = opt.find_optimal_position(a1, e1, a2, e2)
#
# print sol

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