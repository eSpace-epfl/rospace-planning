import datetime as dt
import pykep as pk
import numpy as np
import pickle

from space_tf import Cartesian, CartesianLVLH, KepOrbElem, mu_earth, QNSRelOrbElements


class Command:

    def __init__(self):
        self.deltaV_TEM = [0, 0, 0]
        self.execution_time = dt.datetime(2017, 9, 15, 12, 20, 0)

    def set_deltaV(self, deltaV):
        self.deltaV_TEM = deltaV

    def set_execution_time(self, datetime):
        self.execution_time = datetime


class Position:

    def __init__(self):
        self.cartesian = Cartesian()
        self.lvlh = CartesianLVLH()
        self.kep = KepOrbElem()
        self.rel_kep = QNSRelOrbElements()

        self.id = id
        self.next_position = []
        self.command = Command()

    def set_neighbour(self, next_position):
        self.next_position.append(next_position)


class Scenario:

    def __init__(self):

        # Scenario information
        self.nr_positions = 0
        self.keep_out_zone = 0.05
        self.positions = []
        self.start_scenario = dt.datetime(2017, 9, 15, 12, 20, 0)

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
        # 2. Align to the same orbit
        # 3. Observe target from a convenient position behind or in front of him
        # 4. Approach the target up to a few hundred of meters
        # 5. Go on a circling orbit to increase the position estimation accuracy
        # 6. Stop in the front of the target
        # 7. Begin the approach staying always inside a cone arriving to approx 5m distance

        n_t = np.sqrt(mu_earth/target.a**3)
        n_c = np.sqrt(mu_earth/chaser.a**3)

        # Definition of #0: Actual position that is constantly updated
        P0 = Position()
        P0.rel_kep.from_keporb(target, chaser)
        P0.kep = chaser
        P0.cartesian.from_keporb(chaser)
        P0.lvlh.from_cartesian_pair(chaser, target)

        # Definition of #1
        P1 = Position()
        P1.rel_kep.from_vector([P0.dA, target.m - chaser.m + target.w - chaser.w,
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

