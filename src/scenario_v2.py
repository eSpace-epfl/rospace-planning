import datetime as dt
import pykep as pk
import numpy as np
import pickle
import sys
import os
import xml.etree.ElementTree as ET
import rospy

from space_tf import Cartesian, CartesianLVLH, KepOrbElem, mu_earth, QNSRelOrbElements


class AimPoint:

    def __init__(self, id=0):
        self.id = id
        self.position = Position()


# class Lock:
#
#     def __init__(self):
#         self.cartesian = Cartesian()
#         self.lvlh = CartesianLVLH()
#         self.kep = KepOrbElem()
#         self.rel_kep = QNSRelOrbElements()


class Position:

    def __init__(self):
        # Position definition, made of every possible coordinate
        self.cartesian = Cartesian()
        self.lvlh = CartesianLVLH()
        self.kep = KepOrbElem()
        self.rel_kep = QNSRelOrbElements()

        # Contains a list of locked attributes
        #self._lock = []

        # TODO: Think about which keplerian elements are defined there. Osculating? Mean?

    def from_other_position(self, position):
        self.cartesian.R = position.cartesian.R
        self.cartesian.V = position.cartesian.V

        self.lvlh.R = position.lvlh.R
        self.lvlh.V = position.lvlh.V

        self.kep.a = position.kep.a
        self.kep.e = position.kep.e
        self.kep.i = position.kep.i
        self.kep.O = position.kep.O
        self.kep.w = position.kep.w
        self.kep.v = position.kep.v

        self.rel_kep.dA = position.rel_kep.dA
        self.rel_kep.dL = position.rel_kep.dL
        self.rel_kep.dEx = position.rel_kep.dEx
        self.rel_kep.dEy = position.rel_kep.dEy
        self.rel_kep.dIx = position.rel_kep.dIx
        self.rel_kep.dIy = position.rel_kep.dIy

    def update_target_from_cartesian(self, r, v):
        """
            Update target coordinates from cartesian.
            Note that in this case only keplerian elements and cartesian coordinates are update,
            as the other two make no sense.

        Args:
             r (array): Radius vector in cartesian coordinate of the target
             v (array): Velocity vector in cartesian coordinate of the target
        """

        # Update cartesian coordinates
        self.cartesian.R = np.array(r)
        self.cartesian.V = np.array(v)

        # Update keplerian coordinates
        self.kep.from_cartesian(self.cartesian)

    def update_from_cartesian(self, r, v, target):
        """
            Update a chaser Position given the target position to calculate relative elements.

        Args:
            r (array): Radius vector in cartesian coordinate of the chaser
            v (array): Velocity vector in cartesian coordinate of the chaser
            target (Position):
        """

        # Update cartesian coordinates
        self.cartesian.R = np.array(r)
        self.cartesian.V = np.array(v)

        # Update keplerian coordinates
        self.kep.from_cartesian(self.cartesian)

        # Update lvlh coordinates
        self.lvlh.from_cartesian_pair(self.cartesian, target.cartesian)

        # Update relative orbital elements
        self.rel_kep.from_keporb(target.kep, self.kep)

    def update_from_lvlh(self, target, dR):
        """
            Set a position coordinates depending on the relative distance we want to have from the target.

        Args:
            target (Position)
            dR (Vector3):   Vector in LHLV frame in km, defining relative position bewteen target and chaser.
        """

        self.lvlh.R = dR
        self.cartesian.from_lvlh_frame(target, self.lvlh)
        self.kep.from_cartesian(self.cartesian)
        self.rel_kep.from_keporb(self.kep)

    def update_from_lvc(self, target, dr, dv):
        """
            Assuming we are on the same plane, update position according to radial difference and true anomaly
            difference.
            Depending on where the target will be after a certain time, the position will be redefined
            accordingly.

        Args:
            target (Position)
            dr (float):     Radial difference (km) dr = r_target - r_chaser
            dv (float):     True anomaly difference (radiants) dv = v_target - v_chaser
        """

        v_target = target.kep.v
        v_chaser = dv + v_target

        # Set true anomaly in chaser, mean and eccentric anomaly should be calculated consequently
        self.kep.v = v_chaser

        # Calculate dA, assuming same plane
        self.rel_kep.dA = 2

        # Calculate dL, assuming same plane O_C - O_T = 0
        self.rel_kep.dL = (self.kep.m - self.kep.w) - (target.kep.m - target.kep.w)


        self.rel_kep.from_vector([])

    def update_from_true_anomaly(self, target, dv):
        v_target = target.kep.v
        v_chaser = dv + v_target

        # Set true anomaly in chaser, mean and eccentric anomaly should be calculated consequently
        self.kep.v = v_chaser

        # Calculate dA, assuming same plane
        self.rel_kep.dA = 2

        # Calculate dL, assuming same plane O_C - O_T = 0
        self.rel_kep.dL = (self.kep.m - self.kep.w) - (target.kep.m - target.kep.w)

        self.rel_kep.from_vector([])

class Scenario:

    def __init__(self):
        # Scenario information
        self.nr_positions = 0
        self.keep_out_zone = 0.05
        self.positions = []
        self.name = 'Standard'
        self.overview = ''
        self.mission_start = dt.datetime(2017, 9, 15, 13, 20, 0)

    def import_solved_scenario(self):
        """
            Import a solved scenario from pickle file 'scenario.p'
        """

        # Try to import the file
        try:
            with open('scenario.pickle', 'rb') as file:
                obj = pickle.load(file)
                if obj['scenario_name'] == self.name:
                    print "-----------------> Old manoeuvre elaborated <--------------------"
                    print "Loading old solution..."
                    return obj['command_line']
                else:
                    print "Old scenario does not correspond to actual one."
                    sys.exit(1)
        except IOError:
            print "Scenario file not found."
            sys.exit(1)

    def export_solved_scenario(self, command_line):
        """
            Export a solved scenario into pickle file 'scenario.p'
        """

        # TODO: Find a way to store the complete scenario and upload it afterwards

        # Export the "self" into "scenario.p"
        with open('scenario.pickle', 'wb') as file:

            obj = {'scenario_name': self.name, 'command_line': command_line}

            pickle.dump(obj, file)
            print "Command Line & Scenario saved..."

    def create_scenario(self, target, chaser):
        """
        Define a scenario w.r.t to quasi-relative orbital elements.
        Not suited for equatorial orbit.

        Args:
            target (KepOrbElem)
            chaser (KepOrbElem)
        """

        self.import_yaml_scenario()

        # Overview of a possible plan
        # 1. Reach and keep the same orbital plane
        # 2. Align to the same orbit, far by a certain mean anomaly
        # 3. Observe target from a convenient position behind or in front of him
        # 4. Approach the target up to a few hundred of meters
        # 5. Go on a circling orbit to increase the position estimation accuracy
        # 6. Stop in the front of the target
        # 7. Begin the approach staying always inside a cone arriving to approx 5m distance

        # self.nr_positions = 7
        # self.name = 'Scenario Euler'
        # self.overview = "1. Reach and keep the same orbital plane \n" \
        #                 "2. Align to the same orbit, far by a certain mean anomaly \n" \
        #                 "3. Observe target from a convenient position behind or in front of him \n" \
        #                 "4. Approach the target up to a few hundred of meters \n" \
        #                 "5. Go on a circling orbit to increase the position estimation accuracy \n" \
        #                 "6. Stop in the front of the target \n" \
        #                 "7. Begin the approach staying always inside a cone arriving to approx 5m distance \n"

        # n_t = np.sqrt(mu_earth/target.a**3)

        # Definition of #0: Actual position that is constantly updated

        # Definition of #1
        # P1 = Position()
        # P1.rel_kep.from_vector([(target.a - chaser.a)/chaser.a, target.m - chaser.m + target.w - chaser.w,
        #                         target.e * np.cos(target.w) - chaser.e * np.cos(chaser.w),
        #                         target.e * np.sin(target.w) - chaser.e * np.sin(chaser.w), 0, 0])

        # Definition of #2
        # P2 = Position()
        # P2.rel_kep.from_vector([0, target.m - chaser.m, 0, 0, 0, 0])

        # Definition of #3
        # TODO: Update the scenario at every timestep if the difference between mean anomalies is not constant!!
        # P3 = Position()
        # P3.rel_kep.from_vector([0, n_t, 0, 0, 0, 0])

        # Definition of #4
        # P4 = Position()
        # P4.rel_kep.from_vector([0, n_t * 0.1, 0, 0, 0, 0])

        # Definition of #5
        # P5 = Position()
        #
        # d1 = 120e-3
        # d2 = 80e-3
        # a = target.a
        # e_T = d1/(2*a)
        # e_C = d2/(2*a)
        # eta = -(e_T - e_C)/(1 + e_T*e_C)
        # E_T = 2*np.arctan(np.sqrt((1-e_T)/(1+e_T)) * np.tan(0.5*np.arccos(eta)))
        # E_C = 2*np.arctan(np.sqrt((1-e_C)/(1+e_C)) * np.tan(0.5*np.arccos(eta)))
        #
        # dL = abs(E_T - e_T*np.sin(E_T) - E_C + e_C*np.sin(E_C))
        #
        # P5.rel_kep.from_vector([0, dL,
        #                         100/target.a * np.cos(target.w), 100/target.a * np.sin(target.w),
        #                         0, 0])

        # Definition of #6
        # P6 = Position()
        # P6.rel_kep.from_vector([0, -n_t * 0.03, 0, 0, 0, 0])

        # Definition of #7
        # P7 = Position()
        # P7.rel_kep.from_vector([0, -n_t * 0.01, 0, 0, 0, 0])

        # Add all the position to the list
        # for i in xrange(0, self.nr_positions):
        #     self.positions.append(eval('P' + str(i+1)))

    def update_positions(self, target, chaser):
        """
            Scenario position are updated, as target and chaser are moving in time.
        """

        P1 = self.positions[0]
        P1.rel_kep.from_vector([(target.a - chaser.a)/chaser.a, target.m - chaser.m + target.w - chaser.w,
                                target.e * np.cos(target.w) - chaser.e * np.cos(chaser.w),
                                target.e * np.sin(target.w) - chaser.e * np.sin(chaser.w), 0, 0])

        P2 = self.positions[1]
        P2.rel_kep.from_vector([0, target.m - chaser.m, 0, 0, 0, 0])

        P5 = self.positions[4]

        d1 = 120e-3
        d2 = 80e-3
        a = target.a
        e_T = d1/(2*a)
        e_C = d2/(2*a)
        eta = -(e_T - e_C)/(1 + e_T*e_C)
        E_T = 2*np.arctan(np.sqrt((1-e_T)/(1+e_T)) * np.tan(0.5*np.arccos(eta)))
        E_C = 2*np.arctan(np.sqrt((1-e_C)/(1+e_C)) * np.tan(0.5*np.arccos(eta)))
        dL = abs(E_T - e_T*np.sin(E_T) - E_C + e_C*np.sin(E_C))
        P5.rel_kep.from_vector([0, dL,
                                100/target.a * np.cos(target.w), 100/target.a * np.sin(target.w),
                                0, 0])

    def import_yaml_scenario(self):
        scenario = rospy.get_param('scenario', 0)

        self.nr_positions = len(scenario['AimPoints'])
        self.name = scenario['name']
        self.overview = scenario['overview']

        AP = scenario['AimPoints']

        # Extract AimPoints
        for i in xrange(0, self.nr_positions):
            S = AimPoint()
            S.id = AP['S' + str(i)]['id']
            pos = AP['S' + str(i)]['position']
            for ref_frame in pos:
                for var in pos[ref_frame]:
                    exec('S.position.' + ref_frame + '.' + var +
                         '= np.array([' + pos[ref_frame][var] + '])')
                    #S.position._lock.append(ref_frame + '.' + var)
                    self.positions.append(S)

    def update_yaml_scenario(self, target):
        # Update scenario depending on future target position.
        # For now the locked reference frame is the LVLH, therefore all the other has to be updated according to that

        APs = self.positions

        for AP in APs:
            pos = AP.position
            pos.cartesian.from_lvlh_frame(target.cartesian, pos.lvlh)
            pos.kep.from_cartesian(pos.cartesian)
            pos.rel_kep.from_keporb(target.kep, pos.kep)

        # TODO: Right now, the only blocked value is R in LVLH frame. Would be interesting to implement
        # general conversions that depending on the locked update all the other parameters


