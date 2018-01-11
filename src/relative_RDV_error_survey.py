import numpy as np
import random
import space_tf
import pykep as pk
import matplotlib.pyplot as plt

def clohessy_wiltshire(n, T, dv_0, dr_0, dr_1, B):
    phi_rr = lambda t: np.array([
        [4.0 - 3.0 * np.cos(n * t), 0.0, 0.0],
        [6.0 * np.sin(n * t) - 6.0 * n * t, 1.0, 0.0],
        [0.0, 0.0, np.cos(n * t)]
    ])
    phi_rv = lambda t: np.array([
        [1.0 / n * np.sin(n * t), 2.0 / n * (1 - np.cos(n * t)), 0.0],
        [2.0 / n * (np.cos(n * t) - 1.0), 1.0 / n * (4.0 * np.sin(n * t) - 3.0 * n * t), 0.0],
        [0.0, 0.0, 1.0 / n * np.sin(n * t)]
    ])
    phi_vr = lambda t: np.array([
        [3.0 * n * np.sin(n * t), 0.0, 0.0],
        [6.0 * n * (np.cos(n * t) - 1), 0.0, 0.0],
        [0.0, 0.0, -n * np.sin(n * t)]
    ])
    phi_vv = lambda t: np.array([
        [np.cos(n * t), 2.0 * np.sin(n * t), 0.0],
        [-2.0 * np.sin(n * t), 4.0 * np.cos(n * t) - 3.0, 0.0],
        [0.0, 0.0, np.cos(n * t)]
    ])

    dv_0_to_1 = np.linalg.inv(phi_rv(T)).dot(dr_1 - phi_rr(T).dot(dr_0))

    DeltaV_LVLH = dv_0_to_1 - dv_0
    DeltaV_TEME = np.linalg.inv(B).dot(DeltaV_LVLH)

    return DeltaV_TEME

def O(R, V, B):
    H = np.cross(R, V)
    h = H / np.linalg.norm(H)
    n = h / np.linalg.norm(R) ** 2
    return n * B[2, 0:3]

def tschauner_hempel(a, e, R, dt, dr_0, dv_0, dr_1, dv_1, v_0, v_1, B_0):

    p = a * (1.0 - e**2)
    h = np.sqrt(p * space_tf.Constants.mu_earth)

    eta = np.sqrt(1.0 - e**2)
    n = h / np.linalg.norm(R)**2

    c1 = lambda v: 1.0 + e * np.cos(v)
    c2 = lambda v: e * np.sin(v)

    K = lambda t: n * t
    A = lambda v: c1(v) / p
    B = lambda v: -c2(v) / p
    C = lambda v: A(v) / n

    L = lambda v: np.array([
        [np.cos(v) * c1(v), np.sin(v) * c1(v), 2.0 / eta**2 * (1.0 - 1.5 * c2(v) * K(dt) * c1(v) / eta**3), 0.0, 0.0, 0.0],
        [-np.sin(v) * (1.0 + c1(v)), np.cos(v) * (1.0 + c1(v)), -3.0 / eta**5 * c1(v)**2 * K(dt), 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, np.cos(v), np.sin(v)],
        [-(np.sin(v) + e * np.sin(2.0 * v)), np.cos(v) + e * np.cos(2.0 * v), -3.0 / eta**2 * e * (np.sin(v) / c1(v) + 1.0 / eta**3 * K(dt) * (np.cos(v) + e * np.cos(2.0 * v))), 0.0, 0.0, 0.0],
        [-(2.0 * np.cos(v) + e * np.cos(2.0 * v)), -(2.0 * np.sin(v) + e * np.sin(2.0 * v)), -3.0 / eta**2 * (1.0 - e / eta**3 * (2.0 * np.sin(v) + e * np.sin(2.0 * v)) * K(dt)), 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -np.sin(v), np.cos(v)]
    ])

    M = lambda v: np.array([
        [-3.0 / eta**2 * (e + np.cos(v)), 0.0, 0.0, -1.0 / eta**2 * np.sin(v) * c1(v), -1.0 / eta**2 * (2.0 * np.cos(v) + e + e * np.cos(v)**2), 0.0],
        [-3.0 / eta**2 * np.sin(v) * (c1(v) + e**2) / c1(v), 0.0, 0.0, 1.0 / eta**2 * (np.cos(v) - 2.0 * e + e * np.cos(v)**2), -1.0 / eta**2 * np.sin(v) * (1.0 + c1(v)), 0.0],
        [2.0 + 3.0 * e * np.cos(v) + e**2, 0.0, 0.0, c2(v) * c1(v), c1(v)**2, 0.0],
        [-3.0 / eta**2 * (1.0 + c1(v)) * c2(v) / c1(v), 1.0, 0.0, -1.0 / eta**2 * (1.0 + c1(v)) * (1.0 - e * np.cos(v)), -1.0 / eta**2 * (1.0 + c1(v)) * c2(v), 0.0],
        [0.0, 0.0, np.cos(v), 0.0, 0.0, -np.sin(v)],
        [0.0, 0.0, np.sin(v), 0.0, 0.0, np.cos(v)]
    ])

    phi = L(v_1).dot(M(v_0))

    phi_rr = phi[0:3, 0:3]
    phi_rv = phi[0:3, 3:6]
    phi_vr = phi[3:6, 0:3]
    phi_vv = phi[3:6, 3:6]


    phi_rr_t = (A(v_0) * phi_rr + B(v_0) * phi_rv) / A(v_0)
    phi_rv_t = (A(v_0) * phi_rr + B(v_0) * phi_rv) / A(v_0)
    phi_vr_t = (A(v_0) * phi_vr + B(v_0) * phi_vv - B(v_1) * phi_rr_t) / C(v_1)
    phi_vv_t = (C(v_0) * phi_rv - B(v_1) * phi_rv_t) / C(v_1)

    dv_0_transfer = np.linalg.inv(phi_rv_t).dot(dr_1 - phi_rr_t.dot(dr_0))

    DeltaV_LVLH_TH = dv_0_transfer - dv_0

    DeltaV_TEME_TH = np.linalg.inv(B_0).dot(DeltaV_LVLH_TH)
    DeltaV_TEME_TH = np.linalg.inv(B_0).dot(np.linalg.inv(phi_rv).dot(dr_1 - phi_rr.dot(dr_0)) - dv_0)

    return DeltaV_TEME_TH * 1e-3

def chinese_solver(dr_0, dv_0, a, e, theta0, theta, T, t0):
    p = a * (1.0 - e**2)
    h = np.sqrt(space_tf.Constants.mu_earth * p)

    rho = lambda t: 1.0 + e * np.cos(t)
    s = lambda t: rho(t) * np.sin(t)
    c = lambda t: rho(t) * np.cos(t)
    J = h/p**2 * (T - t0)
    s_ = lambda t: np.cos(t) + e*np.cos(2.0*t)
    c_ = lambda t: -(np.sin(t) + e*np.sin(2.0*t))

    B = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0]
    ])

    dr_0 = B.dot(dr_0)
    dv_0 = B.dot(dv_0)

    # Pseudo-initial values
    phi_theta_0 = 1.0 / (1.0 - e**2) * np.array([
        [1.0 - e**2, 3.0 * e * s(theta0) / rho(theta0) * (1.0 + 1.0 / rho(theta0)), - e * s(theta0) * (1.0 + 1.0 / rho(theta0)), -e * c(theta0) + 2.0],
        [0.0, -3.0 * s(theta0) / rho(theta0) * (1.0 + e**2 / rho(theta0)), s(theta0) * (1.0 + 1.0 / rho(theta0)), c(theta0) - 2.0 * e],
        [0.0, -3.0 * (c(theta0) / rho(theta0) + e), c(theta0) * (1.0 + 1.0 / rho(theta0)) + e, -s(theta0)],
        [0.0, 3.0*rho(theta0) + e**2 - 1.0, -rho(theta0)**2, e * s(theta0)]
    ])

    phi_theta = np.array([
        [1.0, -c(theta) * (1.0 + 1.0 / rho(theta)), s(theta) * (1.0 + 1.0 / rho(theta)), 3.0 * rho(theta)**2 * J],
        [0.0, s(theta), c(theta), 2.0 - 3.0 * e * s(theta) * J],
        [0.0, 2.0 * s(theta), 2.0 * c(theta) - e, 3.0 * (1.0 - 2.0 * e * s(theta) * J)],
        [0.0, s_(theta), c_(theta), -3.0 * e * (s_(theta) * J + s(theta) / rho(theta)**2)]
    ])

    phi_oop = 1.0 / rho(theta - theta0) * np.array([
        [c(theta - theta0), s(theta - theta0)],
        [-s(theta - theta0), c(theta - theta0)]
    ])

    pseudo_drv = phi_theta_0.dot(np.array([dr_0[0], dr_0[2], dv_0[0], dv_0[2]]))
    drv_t = phi_theta.dot(pseudo_drv)
    drv_oop_t = phi_oop.dot(np.array([dr_0[1], dv_0[1]]))

    return drv_t, drv_oop_t


# Define max distance from the target (meters)
D_max = 100

# Possible travel times
T = np.arange(1500, 5000, 100)

# Define nominal semi-major axis of the target (km)
kep_T = space_tf.KepOrbElem()
kep_T.a = 7083.0
kep_T.i = 98.5*np.pi/180
kep_T.w = 343.9251*np.pi/180
kep_T.O = 157.5586*np.pi/180
kep_T.e = 0.0007137
kep_T.v = 0.0

cart_T = space_tf.Cartesian()
cart_T.from_keporb(kep_T)

n = np.sqrt(space_tf.Constants.mu_earth / kep_T.a**3)

R_T_0 = cart_T.R
V_T_0 = cart_T.V
B_0 = cart_T.get_lof()
O_T_0 = O(R_T_0, V_T_0, B_0)

seeds = 100.0
it = 0.0

while it < seeds:
    # Initial distance
    d0 = D_max * random.random()

    # Final distance
    d1 = D_max * random.random()

    # Assuming both are on the R-bar/V-bar plane...
    x0 = 2.0 * d0 * random.random() - d0
    y0 = np.sqrt(d0**2 - x0**2)

    x1 = 2.0 * d1 * random.random() - d1
    y1 = np.sqrt(d1**2 - x1**2)

    dr_0 = np.array([x0 * 0.001, y0 * 0.001, 0.0])
    dr_1 = np.array([x1 * 0.001, y1 * 0.001, 0.0])

    # print dr_0 , dr_1

    # Assuming starting relative velocity to be zero
    R_0 = R_T_0 + np.linalg.inv(B_0).dot(dr_0)
    V_0 = V_T_0 + np.cross(O_T_0, dr_0)
    dv_0 = np.array([0.0, 0.0, 0.0])
    dv_1 = np.array([0.0, 0.0, 0.0])

    for t in T:
        DeltaV_TEME_CW = clohessy_wiltshire(n, t, dv_0, dr_0, dr_1, B_0)

        R_1, V_1 = pk.propagate_lagrangian(R_0, V_0 + DeltaV_TEME_CW, t, space_tf.Constants.mu_earth)
        R_T_1, V_T_1 = pk.propagate_lagrangian(R_T_0, V_T_0, t, space_tf.Constants.mu_earth)

        cart_T_tmp = space_tf.Cartesian()
        cart_T_tmp.R = np.array(R_T_1)
        cart_T_tmp.V = np.array(V_T_1)

        kep_T_tmp = space_tf.KepOrbElem()
        kep_T_tmp.from_cartesian(cart_T_tmp)

        DeltaV_TEME_TH = tschauner_hempel(kep_T.a, kep_T.e, R_T_0, t, dr_0, dv_0, dr_1, dv_1, kep_T.v, kep_T_tmp.v, B_0)

        # DeltaV_TEME_CS = chinese_solver(dr_0, dv_0, kep_T.a, kep_T.e, kep_T.v, kep_T_tmp.v, t, 0)

        # print DeltaV_TEME_TH
        # print DeltaV_TEME_CW

        cart_T_1 = space_tf.Cartesian()
        cart_T_1.R = np.array(R_T_1)
        cart_T_1.V = np.array(V_T_1)
        B_1 = cart_T_1.get_lof()

        R_1_test = R_T_1 + np.linalg.inv(B_1).dot(dr_1)

        sol = pk.lambert_problem(R_0, R_1_test, t, space_tf.Constants.mu_earth, True)

        DeltaV_TEME_PK = sol.get_v1()[0] - V_0

        dr_1_LVLH_PK = B_1.dot(np.array(R_1) - np.array(R_T_1))

        R_1, V_1 = pk.propagate_lagrangian(R_0, V_0 + DeltaV_TEME_TH, t, space_tf.Constants.mu_earth)

        dr_1_LVLH_TH = B_1.dot(np.array(R_1) - np.array(R_T_1))

        err_DeltaV = np.linalg.norm(DeltaV_TEME_PK - DeltaV_TEME_CW)
        err_DeltaV_TH = np.linalg.norm(DeltaV_TEME_PK - DeltaV_TEME_TH)
        err_final_position_CW = np.linalg.norm(dr_1_LVLH_PK - dr_1)
        err_final_position_TH = np.linalg.norm(dr_1_LVLH_TH - dr_1)

        d = max(np.linalg.norm(dr_1), np.linalg.norm(dr_0))

        cart_C = space_tf.Cartesian()
        cart_C.R = R_0
        cart_C.V = V_0
        kep_C = space_tf.KepOrbElem()
        kep_C.from_cartesian(cart_C)
        e_C = kep_C.e
        plt.semilogy(d, err_DeltaV, 'k.', markersize=0.8)
        plt.semilogy(d, err_final_position_CW, 'r.', markersize=0.8)
        plt.semilogy(d, err_DeltaV_TH, 'g.', markersize=0.8)
        plt.semilogy(d, err_final_position_TH, 'b.', markersize=0.8)


    it += 1
    print str(it/seeds*100.0) + "%"

plt.show()