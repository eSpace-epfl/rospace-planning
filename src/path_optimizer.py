from math import sqrt,pi,atan,tan,sin,cos
import space_tf
from PyKEP import *

class PathOptimizer:

    def __init__(self):
        # Initialize the optimizer
        self.stf_target = space_tf.OrbitalElements
        self.stf_chaser = space_tf.OrbitalElements
        print "Initialize"

    def find_optimal_path(self, msg):
        # Find optimal path
        print "Searching for optimal path"

    def callback(self, target_oe, chaser_oe):
        # Update orbital elements value
        orbital_elements = target_oe.position

        a = orbital_elements.semimajoraxis
        e = orbital_elements.eccentricity
        i = orbital_elements.inclination * 2*pi/360
        omega = orbital_elements.arg_perigee * 2*pi/360
        raan = orbital_elements.raan * 2*pi/360
        theta = orbital_elements.true_anomaly * 2*pi/360

        #self.stf_chaser.fromOE(a,e,i,omega,raan,theta)
        #self.stf_chaser.toCartesian()

        self.simple_pykep_solution()

        print "Inside callback with two messages"
        print target_oe
        #self.exact_lambert_solution(target_oe,chaser_oe, 2)

    def exact_lambert_solution(self, target_oe, chaser_oe, objective_time):
        t = target_oe.header.stamp.secs

        orb_el_target = space_tf.OrbitalElements
        orb_el_target.a



        [t_x, t_y, t_z] = self.keplerian_to_cartesian(target_oe.position)
        [c_x, c_y, c_z] = self.keplerian_to_cartesian(chaser_oe.position)


    def keplerian_to_cartesian(self, orbital_elements, t=0):

        # Extract OE
        a = orbital_elements.semimajoraxis
        e = orbital_elements.eccentricity
        i = orbital_elements.inclination * 2*pi/360
        omega = orbital_elements.arg_perigee * 2*pi/360
        raan = orbital_elements.raan * 2*pi/360
        theta = orbital_elements.true_anomaly * 2*pi/360

        # Mu km3/s
        mu_Earth = 3.986004418 * pow(10,5)

        # # Compute mean anomaly
        # n = sqrt(mu_Earth/pow(a/1000,3))
        # T = 2*pi/n
        #
        # # Compute eccentric anomaly
        # EA = 2*atan(tan(theta/2)*sqrt((1-e)/(1+e)))
        #
        # # Compute mean anomaly
        # MA = EA - e*sin(EA)
        #
        # # Compute radius
        # r = a*(1 - e*cos(EA))
        #
        # # Compute coordinates (in km)
        # x = r * (cos(raan)*cos(omega+theta) - sin(raan)*sin(omega+theta)*cos(i))
        # y = r * (sin(raan)*cos(omega+theta) - cos(raan)*sin(omega+theta)*cos(i))
        # z = r * (sin(i)*sin(omega+theta))

        # Compute velocity and position in perifocal frame
        p = a * (1 - pow(e,2))
        h = sqrt(mu_Earth * p)

        r = p/(1 + e*cos(theta))
        r_p = [r * cos(theta), r * sin(theta), 0]

        v = mu_Earth/h
        v_p = [-v * sin(theta), v * (e + cos(theta)), 0]

        # Transform to Earth Inertial Reference Frame


        return r_p


    def simple_pykep_solution(self):
        r_t1 = [1, 0, 0]
        r_c1 = [0.2, 0.2, 0]
        v_t1 = [0, 1, 0]
        t_rdv = 100
        mu = 1
        r_t2, v_t2 = propagate_lagrangian(r_t1,v_t1,t_rdv,mu)

        l = lambert_problem(r_c1, r_t2, t_rdv)


