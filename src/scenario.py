import datetime as dt
import numpy as np
import pickle
import sys
import rospy
import scipy.io as sio
import warnings

from space_tf import Cartesian, CartesianLVLH, KepOrbElem, mu_earth, QNSRelOrbElements, CartesianLVC

class CheckPoint:

    def __init__(self, id=0):
        self.id = id
        self.position = Position()

        self.time_dependancy = False
        self.error_ellipsoid = [0, 0, 0]

    def generate_free_coordinates(self, var_list, chaser, target):
        """
            Check what is given as checkpoint and generate all the missing coordinate, assuming that:
             -> The next orbit has to be coelliptic with the actual one.
             -> If the missing values regards the plane tilt (RAAN, inclination), then we assume the plane is the same
             ->
        Args:
            list_of_defined_values: Contain a list of string of the names of the variables imposed in the .yaml file

        """
        # Set baseline orbital elements
        a = None
        i = None
        e = None
        O = None
        w = None
        v = None

        # Check time dependancy
        if 'kep.v' in var_list or 'rel_kep.dL' in var_list or 'lvlh.R' in var_list:
            self.time_dependancy = True

        # Check plane coordinates. Inclination/RAAN/Argument of perigee
        # Inclination
        if 'kep.i' in var_list:
            i = self.position.kep.i
        elif 'rel_kep.dIx' in var_list:
            i = target.kep.i - self.position.rel_kep.dIx
        elif 'rel_kep.dIy' in var_list:
            # Check RAAN
            if 'kep.O' in var_list:
                i = np.arcsin(self.position.rel_kep.dIy / (target.kep.O - self.position.kep.O))
            elif 'rel_kep.dL' in var_list:
                print "Still to think about it... TBD"
            else:
                # Think about this part...
                pass
        else:
            # Plane inclination not defined, keep the one of the chaser
            i = chaser.kep.i

        # RAAN
        if 'kep.O' in var_list:
            O = self.position.kep.O
        elif 'rel_kep.dIy' in var_list:
            if O != None:
                raise Warning("There may be some conflicts in the definitions of RAAN and inclination!")
            else:
                O = target.kep.O - self.position.rel_kep.dIy / np.sin(self.position.kep.i)
        else:
            # RAAN not defined, keep the one of the chaser
            O = chaser.kep.O

        # Argument of perigee
        if 'kep.w' in var_list:
            w = self.position.kep.w
        elif 'rel_kep.dEx' in var_list:
            # Think how to concatenate the eccentricity with perigee argument...
            pass
        elif 'rel_kep.dEy' in var_list:
            # Same as above
            pass
        else:
            # No definition found, keep using the one of the chaser
            w = chaser.kep.w

        # Check if semi-major axis and/or eccentricity are fixed.
        if 'kep.a' in var_list:
            a = self.position.kep.a
        elif 'rel_kep.dA' in var_list:
            a_T = target.kep.a
            a = a_T / (1.0 + self.position.rel_kep.dA)
        else:
            # Semi-Major axis is calculated according to the assumptions above
            a = -1.0

        if 'kep.e' in var_list:
            e = self.position.kep.e
        elif 'rel_kep.dEx' in var_list and 'rel_kep.dEy' in var_list and 'kep.w' in var_list:
            c = target.kep.e**2 - self.position.rel_kep.dEx**2 - self.position.rel_kep.dEy**2
            b = -2.0 * target.kep.e * np.cos(target.kep.w - self.position.kep.w)

            e1 = (-b + np.sqrt(b**2 - 4.0 * c)) / 2.0
            e2 = (-b - np.sqrt(b**2 - 4.0 * c)) / 2.0

            if e1 < 0:
                e = e2
            else:
                e = e1
        elif 'rel_kep.dEx' in var_list and 'rel_kep.dEy' in var_list:
            # kep.w not defined => assumed that it is the same as the target
            e = np.sqrt(self.position.rel_kep.dEx**2 + self.position.rel_kep.dEy**2 - target.kep.e**2)
        else:
            # Eccentricity is calculated according to the assumptions above
            e = -1.0

        # Check Relative position in LVLH frame
        if 'lvlh.R' in var_list and 'lvlh.V' in var_list:
            # Both relative position and velocity are set, therefore the position is fully defined.
            # Function:
            # -> update_from_lvlh
            # can be used then quit
            # Overwrite all the orbital elements previously defined
            self.position.update_from_lvlh(target)
            return
        elif 'lvlh.R' in var_list and 'lvlh.V' not in var_list:
            # Only radius is defined, therefore velocity has to be reconstructed according to the assumptions
            # above. Theoretically, the relative position and relative navigation starts when we have already reached
            # the same orbital plane

            # TODO: Think about when timing get taken into account... Target should be propagated

            R_rel_TEM_CP = np.linalg.inv(target.cartesian.get_lof()).dot(self.position.lvlh.R)
            R_TEM_CP = R_rel_TEM_CP + target.cartesian.R
            R_PERI_CP = target.kep.get_pof().dot(R_TEM_CP)
            R_PERI_CP_mag = np.linalg.norm(R_PERI_CP)

            e_R_PERI_CP = R_PERI_CP / R_PERI_CP_mag
            v = np.arccos(e_R_PERI_CP[0])

            if a == -1.0 and e == -1.0:
                # Evaluate possible eccentricity given above assumptions
                e1 = (-R_PERI_CP_mag + np.sqrt(R_PERI_CP_mag**2 + 4.0 * (R_PERI_CP_mag*np.cos(v) + chaser.kep.a*chaser.kep.e) * chaser.kep.a * chaser.kep.e)) \
                     / (2.0 * (R_PERI_CP_mag*np.cos(v) + chaser.kep.a * chaser.kep.e))
                e2 = (-R_PERI_CP_mag - np.sqrt(R_PERI_CP_mag**2 + 4.0 * (R_PERI_CP_mag*np.cos(v) + chaser.kep.a*chaser.kep.e) * chaser.kep.a * chaser.kep.e)) \
                     / (2.0 * (R_PERI_CP_mag*np.cos(v) + chaser.kep.a * chaser.kep.e))

                if e1 < 0:
                    e = e2
                else:
                    e = e1

                a = chaser.kep.a * chaser.kep.e / e
            elif a == -1.0 and e != -1.0:
                a = R_PERI_CP_mag * (1.0 + e * np.cos(v)) / (1.0 - e**2)
            elif a != -1.0 and e == -1.0:
                e1 = (-R_PERI_CP_mag * np.cos(v) + np.sqrt(R_PERI_CP_mag**2 * np.cos(v)**2 - 4.0 * a * (R_PERI_CP_mag - a)))/(2*a)
                e2 = (-R_PERI_CP_mag * np.cos(v) - np.sqrt(R_PERI_CP_mag**2 * np.cos(v)**2 - 4.0 * a * (R_PERI_CP_mag - a)))/(2*a)

                if e1 < 0:
                    e = e2
                else:
                    e = e1

            V_PERI_CP = np.sqrt(mu_earth / (a * (1.0 - e ** 2))) * np.array([-np.sin(v), e + np.cos(v), 0.0])
            V_TEM_CP = np.linalg.inv(target.kep.get_pof()).dot(V_PERI_CP)
            V_rel_TEM_CP = target.cartesian.V - V_TEM_CP

            self.position.lvlh.V = target.cartesian.get_lof().dot(V_rel_TEM_CP)

        elif 'lvlh.R' not in var_list and 'lvlh.V' in var_list:
            print "Case with only velocity in lvlh and no position TBD!"
            pass

        else:
            # No relative position defined!
            pass

        # Complete missing relative orbital elements

        # Complete missing orbital elements
        # Check if a is defined now
        if a == -1.0:
            # a TBD
            if e == -1.0:
                # Both semi-major axis and eccentricity are not defined
                warnings.warn("Eccentricity nor Semi-Major Axis defined for Checkpoint " + str(self.id) + " (using coelliptic assumption).")

                # a do not change
                a = chaser.kep.a
                e = target.kep.a * target.kep.e / a

            else:
                # Eccentricity is defined, calculate a w.r.t the position of the chaser
                # Calc specific angular momentum
                H = np.cross(chaser.cartesian.R.flat, chaser.cartesian.V.flat)
                h = np.linalg.norm(H, ord=2)
                rp = h ** 2 / mu_earth * 1 / (1 + e)
                ra = h ** 2 / mu_earth * 1 / (1 - e)
                a = 0.5 * (rp + ra)
        else:
            pass

        # Check true anomaly, if it has already been assigned it means that we are dealing with a time dependant
        # checkpoint.
        if v == None:
            # We do not really care yet about where we are with respect to the target, we just need to reach a certain
            # orbit, update to the actual value of the anomaly.
            v = chaser.kep.v

        # Update orbital elements
        self.position.kep.a = a
        self.position.kep.e = e
        self.position.kep.i = i
        self.position.kep.O = O
        self.position.kep.w = w
        self.position.kep.v = v

        print "\nCheckPoint " + str(self.id) + ":"
        print "      a :     " + str(a)
        print "      e :     " + str(e)
        print "      i :     " + str(i)
        print "      O :     " + str(O)
        print "      w :     " + str(w)
        print "      v :     " + str(v)
        print "      Time dependant? " + str(self.time_dependancy)


        # Update from keporb
        self.position.update_from_keporb()

class Position:

    def __init__(self):
        # Position definition, made of every possible coordinate
        self.cartesian = Cartesian()
        self.lvlh = CartesianLVLH()
        self.kep = KepOrbElem()
        self.rel_kep = QNSRelOrbElements()
        self.lvc = CartesianLVC()

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

    def update_target_from_cartesian(self, r_T, v_T):
        """
            Update target coordinates from cartesian.
            Note that in this case only keplerian elements and cartesian coordinates are update,
            as the other two make no sense.

        Args:
            r_T (3d-array or tuple): New target position in Earth-Inertial Frame
            v_T (3d-array or tuple): New target velocity in Earth-Inertial Frame
        """

        # Update cartesian coordinates
        self.cartesian.R = np.array(r_T)
        self.cartesian.V = np.array(v_T)

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

        # Update lvc coordinate frame
        self.lvc.from_keporb(self.kep, target.kep)

    def update_from_lvlh(self, target):
        """
            Set a position coordinates depending on the relative distance we want to have from the target.

        Args:
            target (Position)
        """

        self.cartesian.from_lvlh_frame(target.cartesian, self.lvlh)
        self.kep.from_cartesian(self.cartesian)
        self.rel_kep.from_keporb(target.kep, self.kep)

    def update_from_keporb(self):
        """
            Update all the other positions given self.kep.
        """

        pass


class Scenario:

    def __init__(self):
        # Scenario information
        self.nr_positions = 0
        self.keep_out_zone = 0.05
        self.checkpoints = []
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
                    print "\n -----------------> Old manoeuvre elaborated <--------------------"
                    print "Old solution loaded!"
                    return obj['command_line']
                else:
                    print "Old scenario does not correspond to actual one."
                    sys.exit(1)
        except IOError:
            print "\nScenario file not found."
            sys.exit(1)

    def export_solved_scenario(self, command_line):
        """
            Export a solved scenario into pickle file 'scenario.p'
        """

        # TODO: Find a way to store the complete scenario and upload it afterwards
        # TODO: Remove the first command, s.t the scenario can be applied regardless of the initial position

        # Export the "self" into "scenario.p"
        with open('scenario.pickle', 'wb') as file:

            obj = {'scenario_name': self.name, 'command_line': command_line}

            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
            print "Command Line & Scenario saved..."

    def create_scenario(self, target, chaser):
        """
        Define a scenario w.r.t to quasi-relative orbital elements.
        Not suited for equatorial orbit.

        Args:
            target (Position)
            chaser (Position)
        """

        self.import_yaml_scenario(chaser, target)

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

    def import_yaml_scenario(self, chaser, target):
        scenario = rospy.get_param('scenario', 0)

        self.nr_positions = len(scenario['CheckPoints'])
        self.name = scenario['name']
        self.overview = scenario['overview']

        CP = scenario['CheckPoints']
        chaser_next = chaser

        # Extract CheckPoints
        for i in xrange(0, self.nr_positions):
            S = CheckPoint()
            S.id = CP['S' + str(i)]['id']

            try:
                S.error_ellipsoid = CP['S' + str(i)]['error_ellipsoid']
            except:
                pass

            pos = CP['S' + str(i)]['position']
            var_list = []
            for ref_frame in pos:
                for var in pos[ref_frame]:
                    exec('S.position.' + ref_frame + '.' + var +
                         '= ' + str(pos[ref_frame][var]))
                    var_list.append(ref_frame + '.' + var)
                self.checkpoints.append(S)
            S.generate_free_coordinates(var_list, chaser_next, target)
            chaser_next = S.position

    def update_yaml_scenario(self, target):
        # Update scenario depending on future target position.
        # For now the locked reference frame is the LVLH, therefore all the other has to be updated according to that

        APs = self.checkpoints

        for AP in APs:
            pos = AP.position
            pos.update_from_lvlh(target, pos.lvlh.R)
            # pos.cartesian.from_lvlh_frame(target.cartesian, pos.lvlh)
            # pos.kep.from_cartesian(pos.cartesian)
            # pos.rel_kep.from_keporb(target.kep, pos.kep)

        # TODO: Right now, the only blocked value is R in LVLH frame. Would be interesting to implement
        # general conversions that depending on the locked update all the other parameters