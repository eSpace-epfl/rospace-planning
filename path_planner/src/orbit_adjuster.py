import numpy as np

from space_tf import mu_earth


class OrbitAdjuster(object):

    def __init__(self, satellite, checkpoint, prop_type='real-world'):
        self.satellite = satellite
        self.checkpoint = checkpoint
        self.prop_type = prop_type


class HohmannTransfer(OrbitAdjuster):

    def __init__(self, satellite, checkpoint, prop_type='real-world'):
        super(HohmannTransfer, self).__init__(satellite, checkpoint, prop_type)

    def is_necessary(self):

        mean_oe = self.satellite.get_mean_oe()

        da = self.checkpoint.state.a - mean_oe.a
        de = self.checkpoint.state.e - mean_oe.e

        tol_a = 0.2
        tol_e = 1.0 / mean_oe.a

        if abs(da) > tol_a or abs(de) > tol_e:
            return True
        else:
            return False

    def evaluate_manoeuvre(self):
        """
            Adjust eccentricity and semi-major axis at the same time with an Hohmann-Transfer like manoeuvre:
            1) Burn at perigee to match the needed intermediate orbit
            2) Burn at apogee to arrive at the final, wanted orbit

        References:
            Howard Curtis, Orbital Mechanics for Engineering Students, Chapter 6
            David A. Vallado, Fundamentals of Astrodynamics and Applications, Second Edition, Chapter 6
        """

        # Evaluating mean orbital elements
        mean_oe = self.satellite.get_mean_oe()

        # Extract initial and final semi-major axis and eccentricities
        a_i = mean_oe.a
        e_i = mean_oe.e
        a_f = self.checkpoint.state.a
        e_f = self.checkpoint.state.e

        r_p_i = a_i * (1.0 - e_i)
        r_p_f = a_f * (1.0 - e_f)

        r_a_i = a_i * (1.0 + e_i)
        r_a_f = a_f * (1.0 + e_f)

        if a_f > a_i:
            # Calculate intermediate orbital elements
            a_int = (r_a_f + r_p_i) / 2.0
            e_int = 1.0 - r_p_i / a_int

            # First burn at perigee, then apogee
            theta_1 = 0.0
            theta_2 = np.pi
        else:
            # Calculate intermediate orbital elements
            a_int = (r_a_i + r_p_f) / 2.0
            e_int = 1.0 - r_p_f / a_int

            # First burn at apogee, then perigee
            theta_1 = np.pi
            theta_2 = 0.0

        print "A_INT:" + str(a_int)
        print "E_INT:" + str(e_int)


        # Calculate delta-V's in perifocal frame of reference
        # First burn
        V_PERI_i_1 = np.sqrt(mu_earth / (a_i * (1.0 - e_i**2))) * np.array([-np.sin(theta_1), e_i + np.cos(theta_1), 0.0])
        V_PERI_f_1 = np.sqrt(mu_earth / (a_int * (1.0 - e_int**2))) * np.array([-np.sin(theta_1), e_int + np.cos(theta_1), 0.0])
        deltaV_C_1 = np.linalg.inv(mean_oe.get_pof()).dot(V_PERI_f_1 - V_PERI_i_1)

        # Second burn
        V_PERI_i_2 = np.sqrt(mu_earth / (a_int * (1.0 - e_int ** 2))) * np.array([-np.sin(theta_2), e_int + np.cos(theta_2), 0.0])
        V_PERI_f_2 = np.sqrt(mu_earth / (a_f * (1.0 - e_f ** 2))) * np.array([-np.sin(theta_2), e_f + np.cos(theta_2), 0.0])
        deltaV_C_2 = np.linalg.inv(mean_oe.get_pof()).dot(V_PERI_f_2 - V_PERI_i_2)

        return [(deltaV_C_1, theta_1), (deltaV_C_2, theta_2)]


class ArgumentOfPerigee(OrbitAdjuster):

    def __init__(self, satellite, checkpoint, prop_type='real-world'):
        super(ArgumentOfPerigee, self).__init__(satellite, checkpoint, prop_type)

    def is_necessary(self):

        mean_oe = self.satellite.get_mean_oe(self.prop_type)

        dw = self.checkpoint.state.w - mean_oe.w

        tol_w = 1.0 / mean_oe.a

        if abs(dw) > tol_w:
            return True
        else:
            return False

    def evaluate_manoeuvre(self):
        """
            Given the chaser relative orbital elements with respect to the target adjust the perigee argument.

        References:
            Howard Curtis, Orbital Mechanics for Engineering Students, Chapter 6
            David A. Vallado, Fundamentals of Astrodynamics and Applications, Second Edition, Chapter 6
        """
        # Mean orbital elements
        mean_oe = self.satellite.get_mean_oe(self.prop_type)

        # Extract constants
        a = mean_oe.a
        e = mean_oe.e

        # Evaluate perigee difference to correct
        dw = (self.checkpoint.state.w - mean_oe.w) % (2.0 * np.pi)

        # Positions where burn can occur
        theta_i_1 = dw / 2.0
        theta_i_2 = theta_i_1 + np.pi
        theta_f_1 = 2.0 * np.pi - theta_i_1
        theta_f_2 = theta_f_1 - np.pi

        # Check which one is the closest TODO: Check the least consuming instead of the closest
        if theta_i_1 < mean_oe.v:
            dv1 = 2.0 * np.pi + theta_i_1 - mean_oe.v
        else:
            dv1 = theta_i_1 - mean_oe.v

        if theta_i_2 < mean_oe.v:
            dv2 = 2.0 * np.pi + theta_i_2 - mean_oe.v
        else:
            dv2 = theta_i_2 - mean_oe.v

        if dv1 > dv2:
            theta_i = theta_i_2
            theta_f = theta_f_2
        else:
            theta_i = theta_i_1
            theta_f = theta_f_1

        # Initial velocity
        V_PERI_i = np.sqrt(mu_earth / (a * (1.0 - e**2))) * np.array([-np.sin(theta_i), e + np.cos(theta_i), 0.0])
        V_TEM_i = np.linalg.inv(mean_oe.get_pof()).dot(V_PERI_i)

        # Final velocity
        V_PERI_f = np.sqrt(mu_earth / (a * (1.0 - e**2))) * np.array([-np.sin(theta_f), e + np.cos(theta_f), 0.0])
        V_TEM_f = np.linalg.inv(self.checkpoint.state.get_pof()).dot(V_PERI_f)

        # Delta-V
        deltaV_C = V_TEM_f - V_TEM_i

        return [(deltaV_C, theta_i)]


class PlaneOrientation(OrbitAdjuster):

    def __init__(self, satellite, checkpoint, prop_type='real-world'):
        super(PlaneOrientation, self).__init__(satellite, checkpoint, prop_type)

    def is_necessary(self):

        mean_oe = self.satellite.get_mean_oe(self.prop_type)

        di = self.checkpoint.state.i - mean_oe.i
        dO = self.checkpoint.state.O - mean_oe.O

        tol_i = 1.0 / mean_oe.i
        tol_O = 1.0 / mean_oe.O

        if abs(di) > tol_i or abs(dO) > tol_O:
            return True
        else:
            return False

    def evaluate_manoeuvre(self):
        """
            Correct plane inclination and RAAN with a single manoeuvre at the node between the two orbital planes.

        References:
            Howard Curtis, Orbital Mechanics for Engineering Students, Chapter 6
            David A. Vallado, Fundamentals of Astrodynamics and Applications, Second Edition, Chapter 6
        """
        # Mean orbital elements
        mean_oe = self.satellite.get_mean_oe(self.prop_type)

        # Extract values
        a = mean_oe.a
        e = mean_oe.e
        i_i = mean_oe.i
        O_i = mean_oe.O

        # Final values
        O_f = self.checkpoint.state.O
        i_f = self.checkpoint.state.i

        # Difference between initial and final values
        dO = O_f - O_i
        di = i_f - i_i

        # Spherical trigonometry
        alpha = np.arccos(np.sin(i_i) * np.sin(i_f) * np.cos(dO) + np.cos(i_i) * np.cos(i_f))
        A_Li = np.arcsin(np.sin(i_f) * np.sin(dO) / np.sin(alpha))
        B_Lf = np.arcsin(np.sqrt(np.cos(i_f)**2 * np.sin(i_i)**2 * np.sin(dO)**2 /
                                 (np.sin(alpha)**2 - np.sin(i_i)**2 * np.sin(i_f)**2 * np.sin(dO)**2)))

        if (i_f > np.pi / 2.0 > i_i) or (i_i > np.pi / 2.0 > i_f):
            B_Lf *= -np.sign(dO)
        elif (i_f > i_i > np.pi / 2.0) or (i_i > i_f > np.pi / 2.0):
            B_Lf *= -np.sign(dO) * np.sign(di)
        else:
            B_Lf *= np.sign(dO) * np.sign(di)

        phi = O_f + B_Lf
        psi = np.sign(dO) * abs(np.arcsin(np.sin(i_i) * np.sin(i_f) * np.sin(dO) / np.sin(alpha)))

        if i_i > i_f:
            psi *= -1.0

        A_Li = -abs(A_Li) * np.sign(psi)

        # Two possible positions where the burn can occur
        theta_1 = (2.0 * np.pi - A_Li - mean_oe.w) % (2.0 * np.pi)
        theta_2 = (theta_1 + np.pi) % (2.0 * np.pi)

        # Choose which of the two position is the closest
        # They consume different dV, the decision has to be taken then depending on if you want to spent a bit more
        # and burn in a specific point, or if you can born anywhere regardless on how much it will cost.
        # Now it's just taking the closest point to do the burn, to decrease the total time of the mission.
        if theta_1 < mean_oe.v:
            dv1 = 2*np.pi + theta_1 - mean_oe.v
        else:
            dv1 = theta_1 - mean_oe.v

        if theta_2 < mean_oe.v:
            dv2 = 2*np.pi + theta_2 - mean_oe.v
        else:
            dv2 = theta_2 - mean_oe.v

        if dv1 > dv2:
            theta_i = theta_2
        else:
            theta_i = theta_1

        # Define vector c in Earth-Inertial frame of reference
        cx = np.cos(psi) * np.cos(phi)
        cy = np.cos(psi) * np.sin(phi)
        cz = np.sin(psi)

        if i_i > i_f:
            cx *= -1.0
            cy *= -1.0
            cz *= -1.0

        # Define rotation of alpha radiants around vector c following right-hand rule
        k1 = 1.0 - np.cos(alpha)
        k2 = np.cos(alpha)
        k3 = np.sin(alpha)
        p = np.array([k1 * cx**2 + k2, k1 * cx * cy + k3 * cz, k1 * cx * cz - k3 * cy])
        q = np.array([k1 * cx * cy - k3 * cz, k1 * cy**2 + k2, k1 * cy * cz + k3 * cx])
        w = np.array([k1 * cx * cz + k3 * cy, k1 * cy * cz - k3 * cx, k1 * cz**2 + k2])
        R_c = np.identity(3)
        R_c[0:3, 0] = p
        R_c[0:3, 1] = q
        R_c[0:3, 2] = w

        # Evaluate velocity vector in Earth-Inertial reference frame at theta_i
        V_PERI_i = np.sqrt(mu_earth / (a * (1.0 - e**2))) * np.array([-np.sin(theta_i), e + np.cos(theta_i), 0.0])
        V_TEM_i = np.linalg.inv(mean_oe.get_pof()).dot(V_PERI_i)

        # Rotate vector around c by alpha radiants
        V_TEM_f = R_c.dot(V_TEM_i)

        # Evaluate deltaV
        deltaV_C = V_TEM_f - V_TEM_i

        return [(deltaV_C, theta_i)]


class AnomalySynchronisation(OrbitAdjuster):

    def __init__(self):
        pass

    def evaluate_manoeuvre(self):
        pass


class Drift(OrbitAdjuster):

    def __init__(self):
        pass

    def evaluate_manoeuvre(self):
        pass


class MultiLambert(OrbitAdjuster):

    def __init__(self):
        pass

    def evaluate_manoeuvre(self):
        pass


class ClohessyWiltshire(OrbitAdjuster):

    def __init__(self):
        pass

    def evaluate_manoeuvre(self):
        pass


class TschaunerHempel(OrbitAdjuster):

    def __init__(self):
        pass

    def evaluate_manoeuvre(self):
        pass


class HamelDeLafontaine(OrbitAdjuster):

    def __init__(self):
        pass

    def evaluate_manoeuvre(self):
        pass


class GeneticAlgorithm(OrbitAdjuster):

    def __init__(self):
        pass

    def evaluate_manoeuvre(self):
        pass
