from propagator.OrekitPropagator import OrekitPropagator
from space_tf import KepOrbElem, CartesianLVLH, Cartesian, mu_earth, J_2, R_earth
import numpy as np
import yaml
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt

def linearized_including_J2(target, de0, v_f, N_orb):
    # Initial reference osculatin orbit
    a_0 = target.a
    e_0 = target.e
    i_0 = target.i
    w_0 = target.w
    M_0 = target.m
    v_0 = target.v

    eta_0 = np.sqrt(1.0 - e_0 ** 2)
    p_0 = a_0 * (1.0 - e_0 ** 2)
    r_0 = p_0 / (1.0 + e_0 * np.cos(v_0))

    # Initial reference mean orbit
    target_mean = KepOrbElem()
    target_mean.from_osc_elems(target)

    a_mean = target_mean.a
    i_mean = target_mean.i
    e_mean = target_mean.e

    eta_mean = np.sqrt(1.0 - e_mean ** 2)
    p_mean = a_mean * (1.0 - e_mean ** 2)
    n_mean = np.sqrt(mu_earth / a_mean ** 3)

    # Mean orbital element drift
    w_mean_dot = 0.75 * J_2 * n_mean * (R_earth / p_mean) ** 2 * (5.0 * np.cos(i_mean) ** 2 - 1.0)
    M_mean_dot = n_mean + 0.75 * J_2 * n_mean * (R_earth / p_mean) ** 2 * eta_mean * \
                 (3.0 * np.cos(i_mean) ** 2 - 1.0)

    # Epsilon_a partial derivatives: TODO: v_0 or v???
    gamma_2 = -0.5 * J_2 * (R_earth / a_0) ** 2
    depsde = a_0 * gamma_2 * ((2.0 - 3.0 * np.sin(i_0) ** 2) *
             (3.0 * np.cos(v_0) * (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 6 + 6.0 * e_0 * (1.0 + e_0 * np.cos(v_0)) ** 3 / eta_0 ** 8 - 3.0 * e_0 / eta_0 ** 5) +
             9.0 * np.sin(i_0) ** 2 * np.cos(2.0 * w_0 + 2.0 * v_0) * np.cos(v_0) * (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 6 +
             18.0 * np.sin(i_0) ** 2 * e_0 * np.cos(2.0 * w_0 + 2.0 * v_0) * (1.0 + e_0 * np.cos(v_0)) ** 3 / eta_0 ** 8)
    depsdi = -3.0 * a_0 * gamma_2 * np.sin(2.0 * i_0) * ((a_0 / r_0) ** 3 * (1.0 - np.cos(2.0 * w_0 + 2.0 * v_0)) - 1.0 / eta_0 ** 3)
    depsdw = -6.0 * a_0 * gamma_2 * (1.0 - np.cos(i_0) ** 2) * (a_0 / r_0) ** 3 * np.sin(2.0 * w_0 + 2.0 * v_0)
    depsdv = a_0 * gamma_2 * (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 6 * \
             ((-9.0 * np.cos(i_0) ** 2 + 3.0) * e_0 * np.sin(v_0) -
             (9.0 - 9.0 * np.cos(i_0) ** 2) * np.cos(2.0 * w_0 + 2.0 * v_0) * e_0 * np.sin(v_0) -
             (6.0 - 6.0 * np.cos(i_0) ** 2) * (1.0 + e_0 * np.cos(v_0)) * np.sin(2.0 * w_0 + 2.0 * v_0))

    # Mean elements partial derivatives
    C = J_2 * n_mean * R_earth ** 2 / (4.0 * p_mean ** 2)           # TODO: p or p_mean?
    dOda = 21.0 / a_mean * C * np.cos(i_mean)
    dOde = 24.0 * e_mean / eta_mean ** 2 * C * np.cos(i_mean)
    dOdi = 6.0 * C * np.sin(i_mean)
    dwda = -10.5 * C * (5.0 * np.cos(i_mean) ** 2 - 1.0) / a_mean
    dwde = 12.0 * e_mean * C * (5.0 * np.cos(i_mean) ** 2 - 1.0) / eta_mean ** 2
    dwdi = -15.0 * C * np.sin(2.0 * i_mean)
    dMda = -3.0 * n_mean / (2.0 * a_mean) - eta_mean / (2.0 * a_mean) * C * (63.0 * np.cos(i_mean)**2 - 21.0)
    dMde = 9.0 * e_mean * C * (3.0 * np.cos(i_mean) ** 2 - 1.0) / eta_mean
    dMdi = -9.0 * eta_mean * C * np.sin(2.0 * i_mean)

    # Estimate flight time
    # N_orb = ...
    E = lambda v: 2.0 * np.arctan(np.sqrt((1.0 - e_0) / (1.0 + e_0)) * np.tan(v / 2.0))
    M = lambda v: (E(v) - e_0 * np.sin(E(v))) % (2.0 * np.pi)

    tau = lambda v: (2.0 * np.pi * N_orb + M(v) - M_0) / M_mean_dot

    # Position
    r = lambda v: p_0 / (1.0 + e_0 * np.cos(v))

    # Position and true anomaly derivatives         # TODO: CHECK IF divided by eta_0 or eta?
    r_dot = lambda v: a_0 * e_0 * np.sin(v) / eta_0 * M_mean_dot
    v_dot = lambda v: (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3 * M_mean_dot

    # Phi_1
    k_x_dot = lambda v: a_0 * e_0 * v_dot(v) * np.cos(v) / eta_0
    phi_11 = lambda v: r_dot(v) / a_0 + (k_x_dot(v) * tau(v) + a_0 * e_0 * np.sin(v) / eta_0) * dMda
    phi_12 = lambda v: a_0 * v_dot(v) * np.sin(v) + (k_x_dot(v) * tau(v) + a_0 * e_0 * np.sin(v) / eta_0) * \
                       (dMde + dMda * depsde + dMda * depsdv * np.sin(v_0) / eta_0 ** 2 * (2.0 + e_0 * np.cos(e_0)))
    phi_13 = lambda v: (k_x_dot(v) * tau(v) + a_0 * e_0 * np.sin(v) / eta_0) * (dMda * depsdi + dMdi)
    phi_14 = 0.0
    phi_15 = lambda v: (k_x_dot(v) * tau(v) + a_0 * e_0 * np.sin(v) / eta_0) * dMda * depsdw
    phi_16 = lambda v: k_x_dot(v) + (k_x_dot(v) * tau(v) + a_0 * e_0 * np.sin(v) / eta_0) * dMda * depsdv * \
                       (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 3

    # Phi 2
    k_y_dot = lambda v: r_dot(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3 - 2.0 * e_0 * v_dot(v) * np.sin(v) * (1.0 + e_0 * np.cos(v)) / eta_0 ** 3
    phi_21 = lambda v: (r_dot(v) * np.cos(i_0) * tau(v) + r(v) * np.cos(i_0)) * dOda + (r_dot(v) * tau(v) + r(v)) * dwda + \
                       (k_y_dot(v) * tau(v) + r(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3) * dMda
    phi_22 = lambda v: 1.0 / eta_0 ** 2 * (r(v) * v_dot(v) * np.cos(v) * (2.0 + e_0 * np.cos(v)) - r(v) * e_0 * v_dot(v) * np.sin(v) ** 2 +
                       r_dot(v) * np.sin(v) * (2.0 + e_0 * np.cos(v))) + (r_dot(v) * np.cos(i_0) * tau(v) + r(v) * np.cos(i_0)) * \
                       (dOda * depsde + dOda * depsdv * np.sin(v_0) / eta_0 ** 2 * (2.0 + e_0 * np.cos(e_0)) + dOde) + \
                       (r_dot(v) * tau(v) + r(v)) * (dwda * depsde + dwda * depsdv * np.sin(v_0) / eta_0 ** 2 * (2.0 + e_0 * np.cos(e_0)) + dwde) + \
                       (k_y_dot(v) * tau(v) + r(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3) * \
                       (dMda * depsde + dMda * depsdv * np.sin(v_0) * (2.0 + e_0 * np.cos(e_0)) / eta_0 ** 2 + dMde)
    phi_23 = lambda v: (r_dot(v) * np.cos(i_0) * tau(v) + r(v) * np.cos(i_0)) * (dOda * depsdi + dOdi) + \
                       (r_dot(v) * tau(v) + r(v)) * (dwda * depsdi + dwdi) + \
                       (k_y_dot(v) * tau(v) + r(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3) * (dMda * depsdi + dMdi)
    phi_24 = lambda v: r_dot(v) * np.cos(i_0)
    phi_25 = lambda v: r_dot(v) + (r_dot(v) * np.cos(i_0) * tau(v) + r(v) * np.cos(i_0)) * dOda * depsdw + \
                       (r_dot(v) * tau(v) + r(v)) * dwda * depsdw + (k_y_dot(v) * tau(v) + r(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3) * dMda * depsdw
    phi_26 = lambda v: k_y_dot(v) + (r_dot(v) * np.cos(i_0) * tau(v) + r(v) * np.cos(i_0)) * dOda * depsdv * (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 3 + \
                       (r_dot(v) * tau(v) + r(v)) * dwda * depsdv * (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 3 + \
                       (k_y_dot(v) * tau(v) + r(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3) * dOda * depsdv * \
                       (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 3

    # Phi 3
    k_z_dot = lambda v: -r_dot(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0) + \
                        r(v) * np.sin(v + w_0 + w_mean_dot * tau(v)) * (v_dot(v) + w_mean_dot) * np.sin(i_0)
    phi_31 = lambda v: (k_z_dot(v) * tau(v) - r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0)) * dOda
    phi_32 = lambda v: (k_z_dot(v) * tau(v) - r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0)) * \
                       (dOda * depsde + dOda * depsdv * np.sin(v_0) / eta_0 ** 2 * (2.0 + e_0 * np.cos(e_0)) + dOde)
    phi_33 = lambda v: r_dot(v) * np.sin(v + w_0 + w_mean_dot * tau(v)) + r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * \
                       (v_dot(v) + w_mean_dot) + (k_z_dot(v) * tau(v) - r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0)) * dOda
    phi_34 = lambda v: k_z_dot(v)
    phi_35 = lambda v: (k_z_dot(v) * tau(v) - r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0)) * dOda * depsdw
    phi_36 = lambda v: (k_z_dot(v) * tau(v) - r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0)) * dOda * depsdv * \
                       (1.0 + e_0 * np.cos(e_0)) ** 2 / eta_0 ** 3

    # Phi 4
    phi_41 = lambda v: r(v) / a_0 + a_0 * e_0 * np.sin(v) / eta_0 * dMda * tau(v)
    phi_42 = lambda v: a_0 * e_0 * np.sin(v) / eta_0 * (dMda * depsde + dMda * depsdv * np.sin(v_0) / eta_0 ** 2 * (2.0 + e_0 * np.cos(e_0)) +
                       dMde) * tau(v) - a_0 * np.cos(v)
    phi_43 = lambda v: a_0 * e_0 * np.sin(v) / eta_0 * (dMda * depsdi + dMdi) * tau(v)
    phi_44 = 0.0
    phi_45 = lambda v: a_0 * e_0 * np.sin(v) / eta_0 * dMda * depsdw * tau(v)
    phi_46 = lambda v: a_0 * e_0 * np.sin(v) / eta_0 + a_0 * e_0 * np.sin(v) / eta_0 * dMda * depsdw * \
                       (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 3 * tau(v)

    # Phi 5
    phi_51 = lambda v: r(v) * np.cos(i_0) * dOda * tau(v) + r(v) * dwda * tau(v) + r(v) * (1.0 + e_0 * np.cos(v)) ** 2 \
                       / eta_0 ** 3 * dMda * tau(v)
    phi_52 = lambda v: r(v) * np.sin(v) / eta_0 ** 2 * (2.0 + e_0 * np.cos(v)) + r(v) * np.cos(i_0) * \
                       (dOda * depsde + dOda * depsdv * np.sin(v_0) / eta_0 ** 2 * (2.0 + e_0 * np.cos(e_0)) + dOde) * tau(v) + \
                       r(v) * (dwda * depsde + dwda * depsdv * np.sin(v_0) / eta_0 ** 2 * (2.0 + e_0 * np.cos(e_0)) + dwde) * tau(v) + \
                       r(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3 * \
                       (dMda * depsde + dMda * depsdv * np.sin(v_0) / eta_0 ** 2 * (2.0 + e_0 * np.cos(e_0)) + dMde) * tau(v)
    phi_53 = lambda v: r(v) * np.cos(i_0) * (dOda * depsdi + dOdi) * tau(v) + r(v) * (dwda * depsdi + dwdi) * tau(v) + \
                       r(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3 * (dMda * depsdi + dMdi) * tau(v)
    phi_54 = lambda v: r(v) * np.cos(i_0)
    phi_55 = lambda v: r(v) + r(v) * np.cos(i_0) * dOda * depsdw * tau(v) + r(v) * dwda * depsdw * tau(v) + \
                       r(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3 * dMda * depsdw * tau(v)
    phi_56 = lambda v: r(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3 + r(v) * np.cos(i_0) * dOda * depsdv * \
                       (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 3 * tau(v) + r(v) * dwda * depsdv * (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 3 * tau(v) + \
                       r(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3 * dMda * depsdv * (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 3 * tau(v)

    # Phi 6
    phi_61 = lambda v: -r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0) * dOda * tau(v)
    phi_62 = lambda v: -r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0) * (dOda * depsde + dOda * depsdv * np.sin(v_0) /
                        eta_0 ** 2 * (2.0 + e_0 * np.cos(e_0)) + dOde) * tau(v)
    phi_63 = lambda v: r(v) * np.sin(v + w_0 + w_mean_dot * tau(v)) - r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0) * (dOda * depsdi + dOdi) * tau(v)
    phi_64 = lambda v: -r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0)
    phi_65 = lambda v: -r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0) * dOda * depsdw * tau(v)
    phi_66 = lambda v: -r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0) * dOda * depsdv * (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 3 * tau(v)

    phi_ = np.array([
        [phi_11(v_f), phi_12(v_f), phi_13(v_f), phi_14, phi_15(v_f), phi_16(v_f)],
        [phi_21(v_f), phi_22(v_f), phi_23(v_f), phi_24(v_f), phi_25(v_f), phi_26(v_f)],
        [phi_31(v_f), phi_32(v_f), phi_33(v_f), phi_34(v_f), phi_35(v_f), phi_36(v_f)],
        [phi_41(v_f), phi_42(v_f), phi_43(v_f), phi_44, phi_45(v_f), phi_46(v_f)],
        [phi_51(v_f), phi_52(v_f), phi_53(v_f), phi_54(v_f), phi_55(v_f), phi_56(v_f)],
        [phi_61(v_f), phi_62(v_f), phi_63(v_f), phi_64(v_f), phi_65(v_f), phi_66(v_f)],
    ])

    state = phi_.dot(de0)

    print "TAU: " + str(tau(v_f))

    return phi_, tau(v_f)

def find_v(time, a_mean, e_mean, i_mean, M_0, e_0):

    n_mean = np.sqrt(mu_earth / a_mean**3)
    p_mean = a_mean * (1.0 - e_mean**2)
    eta_mean = np.sqrt(1.0 - e_mean ** 2)

    T = 2.0 * np.pi / n_mean

    M_mean_dot = n_mean + 0.75 * J_2 * n_mean * (R_earth / p_mean) ** 2 * eta_mean * \
                 (3.0 * np.cos(i_mean) ** 2 - 1.0)

    N_orb = np.floor(time / T)

    M_v = (time * M_mean_dot - 2.0 * np.pi * N_orb + M_0)
    E_v = calc_E_from_m(M_v, e_0)
    v = calc_v_from_E(E_v, e_0)

    return v, N_orb

def calc_E_from_m(m, e):
    if m < np.pi:
        E = m + e / 2.0
    else:
        E = m - e / 2.0

    max_int = 20  # maximum number of iterations

    while max_int > 1:
        fE = E - e * np.sin(E) - m
        fpE = 1.0 - e * np.cos(E)
        ratio = fE / fpE
        max_int = max_int - 1

        # check if ratio is small enough
        if abs(ratio) > 1e-15:
            E = E - ratio
        else:
            break

    if E < 0:
        E = E + np.pi * 2.0

    return E

def calc_v_from_E(E, e):
    v = 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(E / 2.0),
                               np.sqrt(1.0 - e) * np.cos(E / 2.0))

    if v < 0:
        v = v + np.pi * 2.0

    return v

def print_state(kep):
    print "      a :     " + str(kep.a)
    print "      e :     " + str(kep.e)
    print "      i :     " + str(kep.i)
    print "      O :     " + str(kep.O)
    print "      w :     " + str(kep.w)
    print "      v :     " + str(kep.v)

def travel_time(state, theta0, theta1):
    """
        Evaluate the travel time of a satellite from a starting true anomaly theta0 to an end anomaly theta1.

    Reference:
        Exercise of Nicollier's Lecture.
        David A. Vallado, Fundamentals of Astrodynamics and Applications, Second Edition, Algorithm 11 (p. 133)

    Args:
        state (KepOrbElem): Satellite state in keplerian orbital elements.
        theta0 (rad): Starting true anomaly.
        theta1 (rad): Ending true anomaly.

    Return:
        Travel time (seconds)
    """

    a = state.a
    e = state.e

    T = 2.0 * np.pi * np.sqrt(a**3 / mu_earth)

    theta0 = theta0 % (2.0 * np.pi)
    theta1 = theta1 % (2.0 * np.pi)

    t0 = np.sqrt(a**3/mu_earth) * (2.0 * np.arctan((np.sqrt((1.0 - e)/(1.0 + e)) * np.tan(theta0 / 2.0))) -
                                   (e * np.sqrt(1.0 - e**2) * np.sin(theta0))/(1.0 + e * np.cos(theta0)))
    t1 = np.sqrt(a**3/mu_earth) * (2.0 * np.arctan((np.sqrt((1.0 - e)/(1.0 + e)) * np.tan(theta1 / 2.0))) -
                                   (e * np.sqrt(1.0 - e**2) * np.sin(theta1))/(1.0 + e * np.cos(theta1)))

    dt = t1 - t0

    if dt < 0:
        dt += T

    return dt

start_date = datetime.utcnow()
init_state_ch = KepOrbElem()
init_state_ta = KepOrbElem()
target_mean = KepOrbElem()
chaser_cart = Cartesian()
target_cart = Cartesian()
chaser_kep = KepOrbElem()
target_kep = KepOrbElem()
chaser_LVLH = CartesianLVLH()

# OSCULATING OE
init_state_ta.a = 7067.94135375
init_state_ta.e = 0.000387549807193
init_state_ta.i = 1.72273328395
init_state_ta.O = 0.54965315202
init_state_ta.w = 1.08302214192
init_state_ta.v = 0.833317371615
target_cart.from_keporb(init_state_ta)
target_mean.from_osc_elems(init_state_ta)

print "----------------------------------------------------------"
print "Initial target cartesian position: "
print " >>> R: " + str(target_cart.R)
print " >>> V: " + str(target_cart.V)
print "----------------------------------------------------------\n"
print "----------------------------------------------------------"
print "Initial target osculating orbital elements: "
print_state(init_state_ta)
print "----------------------------------------------------------\n"

# Initial relative position
init_lvlh_ch = CartesianLVLH()
init_lvlh_ch.R = np.array([ -8.87017543e-4,   17.9770090,  -2.45741513e-3])
init_lvlh_ch.V = np.array([  6.89534714e-5,  -1.76776403e-3,  -1.42231073e-3])
chaser_cart.from_lvlh_frame(target_cart, init_lvlh_ch)
init_state_ch.from_cartesian(chaser_cart)

print "----------------------------------------------------------"
print "Initial chaser cartesian position: "
print " >>> R: " + str(chaser_cart.R)
print " >>> V: " + str(chaser_cart.V)
print "----------------------------------------------------------\n"

print "----------------------------------------------------------"
print "Initial chaser relative position: "
print " >>> R: " + str(init_lvlh_ch.R)
print " >>> V: " + str(init_lvlh_ch.V)
print "----------------------------------------------------------\n"

print "----------------------------------------------------------"
print "Initial chaser osculating orbital elements: "
print_state(init_state_ch)
print "----------------------------------------------------------\n"

de0_initial = np.array([
    init_state_ch.a - init_state_ta.a,
    init_state_ch.e - init_state_ta.e,
    init_state_ch.i - init_state_ta.i,
    init_state_ch.O - init_state_ta.O,
    init_state_ch.w - init_state_ta.w,
    init_state_ch.m - init_state_ta.m,
])

print "----------------------------------------------------------"
print "Initial osculating orbital elements difference: "
print " >>> da: " + str(de0_initial[0])
print " >>> de: " + str(de0_initial[1])
print " >>> di: " + str(de0_initial[2])
print " >>> dO: " + str(de0_initial[3])
print " >>> dw: " + str(de0_initial[4])
print " >>> dm: " + str(de0_initial[5])
print "----------------------------------------------------------\n"

# Final wanted position
state_final = np.array([0.0, 0.0, 0.0, 0.0, -2.0, 0.0])

tau = 29000.0
T_mean = 2.0*np.pi*np.sqrt(target_mean.a**3 / mu_earth)

# print T_mean
# print np.floor(tau / T_mean)
# print tau - np.floor(tau / T_mean) * T_mean
# print target_mean.v
# print travel_time(target_mean, target_mean.v, 0.0)
# print init_state_ta.v
# print travel_time(init_state_ta, init_state_ta.v, 0.0)

v = init_state_ta.v
N_orb = 0.0
st_0 = linearized_including_J2(init_state_ta, 0.0, v, N_orb)
phi_0 = st_0[0]

a_mean = target_mean.a
i_mean = target_mean.i
e_mean = target_mean.e
eta_mean = np.sqrt(1.0 - e_mean ** 2)
p_mean = a_mean * (1.0 - e_mean ** 2)
n_mean = np.sqrt(mu_earth / a_mean ** 3)
M_mean_dot = n_mean + 0.75 * J_2 * n_mean * (R_earth / p_mean) ** 2 * eta_mean * (3.0 * np.cos(i_mean) ** 2 - 1.0)

m_0 = init_state_ta.m
N_orb = 4.0
deltaM_4 = M_mean_dot * tau - (2.0 * np.pi * N_orb)
m_f_4 = deltaM_4 + m_0
v_4 = calc_v_from_E(calc_E_from_m(tau * M_mean_dot + init_state_ta.m - 2.0 * np.pi * N_orb, init_state_ta.e), init_state_ta.e)
N_orb = 5.0
deltaM_5 = M_mean_dot * tau - (2.0 * np.pi * N_orb)
m_f_5 = deltaM_5 + m_0
v_5 = calc_v_from_E(calc_E_from_m(tau * M_mean_dot + init_state_ta.m - 2.0 * np.pi * N_orb, init_state_ta.e), init_state_ta.e)

st = linearized_including_J2(init_state_ta, 0.0, v, N_orb)
phi = st[0]

phi_comb = np.array([
    phi_0[0:6][3],
    phi_0[0:6][4],
    phi_0[0:6][5],
    phi[0:6][3],
    phi[0:6][4],
    phi[0:6][5]
])

state_comb = np.array([init_lvlh_ch.R[0], init_lvlh_ch.R[1], init_lvlh_ch.R[2], 0.0, -2.0, 0.0])

# Wanted initial relative orbital elements
de0_wanted = np.linalg.inv(phi_comb).dot(state_comb)

print de0_wanted

de0_diff = de0_wanted - de0_initial
print "\n Difference in initial delta orbital elements"
print de0_diff

de_chaser_wanted = np.array([
    init_state_ch.a + de0_diff[0],
    init_state_ch.e + de0_diff[1],
    init_state_ch.i + de0_diff[2],
    init_state_ch.O + de0_diff[3],
    init_state_ch.w + de0_diff[4],
    init_state_ch.m + de0_diff[5],
])

print "\n Wanted chaser orbital elements"
print de_chaser_wanted

chaser_kep_wanted = KepOrbElem()
chaser_kep_wanted.a = de_chaser_wanted[0]
chaser_kep_wanted.e = de_chaser_wanted[1]
chaser_kep_wanted.i = de_chaser_wanted[2]
chaser_kep_wanted.O = de_chaser_wanted[3]
chaser_kep_wanted.w = de_chaser_wanted[4]
chaser_kep_wanted.m = de_chaser_wanted[5]

R_chaser_initial = chaser_cart.R
V_chaser_initial = chaser_cart.V

chaser_cart.from_keporb(chaser_kep_wanted)

R_chaser_initial_wanted = chaser_cart.R
V_chaser_initial_wanted = chaser_cart.V

print "R_chaser_wanted: "  + str(R_chaser_initial)
print "R_chaser:        " + str(R_chaser_initial_wanted)
print "\n"
print V_chaser_initial
print V_chaser_initial_wanted

chaser_LVLH.from_cartesian_pair(chaser_cart, target_cart)

print "\n "
print "LVLH initial:    " + str(init_lvlh_ch.R)
print "LVLH calculated: " + str(chaser_LVLH.R)

print "\n "
print "Delta-V: " + str(V_chaser_initial_wanted - V_chaser_initial)

chaser_cart.R = R_chaser_initial_wanted
chaser_cart.V = V_chaser_initial_wanted

chaser_kep.from_cartesian(chaser_cart)

# Propagate target by tau
settings_path = '/home/dfrey/cso_ws/src/rdv-cap-sim/simulator/cso_gnc_sim/cfg/chaser.yaml'
settings_file = file(settings_path, 'r')
propSettings = yaml.load(settings_file)
propSettings = propSettings['propagator_settings']

OrekitPropagator.init_jvm()
prop_chaser = OrekitPropagator()
# get settings from yaml file
prop_chaser.initialize(propSettings,
                       chaser_kep,
                       start_date)

settings_path = '/home/dfrey/cso_ws/src/rdv-cap-sim/simulator/cso_gnc_sim/cfg/target.yaml'
settings_file = file(settings_path, 'r')
propSettings = yaml.load(settings_file)
propSettings = propSettings['propagator_settings']
prop_target = OrekitPropagator()
# get settings from yaml file
prop_target.initialize(propSettings,
                       init_state_ta,
                       start_date)

target = prop_target.propagate(start_date + timedelta(seconds=tau))
chaser = prop_chaser.propagate(start_date + timedelta(seconds=tau))

target_cart.R = target[0].R
target_cart.V = target[0].V

chaser_cart.R = chaser[0].R
chaser_cart.V = chaser[0].V

chaser_LVLH.from_cartesian_pair(chaser_cart, target_cart)

print "Chaser R: " + str(chaser_LVLH.R)
print "Chaser V: " + str(chaser_LVLH.V)
