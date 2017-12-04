import datetime as dt
import pykep as pk
import numpy as np
import pickle

from space_tf import Cartesian, CartesianLVLH, KepOrbElem, mu_earth, QNSRelOrbElements


class Position:

    def __init__(self):
        # Position definition
        # Note that in theory the cartesian, keplerian and lvlh position are available
        # only for target and chaser actual position.
        # For all the others only the wanted relative keplerian are manually defined
        self.cartesian = Cartesian()
        self.lvlh = CartesianLVLH()
        self.kep = KepOrbElem()
        self.rel_kep = QNSRelOrbElements()
        # TODO: Think about which keplerian elements are defined there. Osculating? Mean?

        self.id = id
        self.next_position = []

    def set_neighbour(self, next_position):
        self.next_position.append(next_position)

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
        f = open('scenario.p', 'rb')

        # Write saved scenario information in self

    def export_solved_scenario(self):
        """
            Export a solved scenario into pickle file 'scenario.p'
        """

        f = open('scenario.p', 'wb')
        pickle.dump(self, f)
        f.close()

    def create_scenario(self, target, chaser):
        """
        Define a scenario w.r.t to quasi-relative orbital elements.
        Not suited for equatorial orbit.

        Args:
            target (KepOrbElem)
            chaser (KepOrbElem)
        """

        # Overview of a possible plan
        # 1. Reach and keep the same orbital plane
        # 2. Align to the same orbit, far by a certain mean anomaly
        # 3. Observe target from a convenient position behind or in front of him
        # 4. Approach the target up to a few hundred of meters
        # 5. Go on a circling orbit to increase the position estimation accuracy
        # 6. Stop in the front of the target
        # 7. Begin the approach staying always inside a cone arriving to approx 5m distance

        self.nr_positions = 7
        self.name = 'Scenario Euler'
        self.overview = "1. Reach and keep the same orbital plane \n" \
                        "2. Align to the same orbit, far by a certain mean anomaly \n" \
                        "3. Observe target from a convenient position behind or in front of him \n" \
                        "4. Approach the target up to a few hundred of meters \n" \
                        "5. Go on a circling orbit to increase the position estimation accuracy \n" \
                        "6. Stop in the front of the target \n" \
                        "7. Begin the approach staying always inside a cone arriving to approx 5m distance \n"

        n_t = np.sqrt(mu_earth/target.a**3)
        n_c = np.sqrt(mu_earth/chaser.a**3)

        # Definition of #0: Actual position that is constantly updated

        # Definition of #1
        P1 = Position()
        P1.rel_kep.from_vector([(target.a - chaser.a)/chaser.a, target.m - chaser.m + target.w - chaser.w,
                                target.e * np.cos(target.w) - chaser.e * np.cos(chaser.w),
                                target.e * np.sin(target.w) - chaser.e * np.sin(chaser.w), 0, 0])

        # Definition of #2
        P2 = Position()
        P2.rel_kep.from_vector([0, 1, 0, 0, 0, 0])

        # Definition of #3
        P3 = Position()
        P3.rel_kep.from_vector([0, target.m - chaser.m, 0, 0, 0, 0])

        # Definition of #4
        P4 = Position()
        P4.rel_kep.from_vector([0, n_t * 0.1, 0, 0, 0, 0])

        # Definition of #5
        P5 = Position()

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

        # Definition of #6
        P6 = Position()
        P6.rel_kep.from_vector([0, -n_t * 0.03, 0, 0, 0, 0])

        # Definition of #7
        P7 = Position()
        P7.rel_kep.from_vector([0, -n_t * 0.01, 0, 0, 0, 0])

        # Add all the position to the list
        for i in xrange(0, self.nr_positions):
            self.positions.append(eval('P' + str(i+1)))
