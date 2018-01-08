import matplotlib.pyplot as plt
import numpy as np
import pykep as pk

def B(R, V):
    i = R / np.linalg.norm(R)
    H = np.cross(R, V)
    k = H / np.linalg.norm(H)
    j = -np.cross(i, k)
    B_0 = np.identity(3)
    B_0[0, 0:3] = i
    B_0[1, 0:3] = j
    B_0[2, 0:3] = k

    return B_0

def chinese_solver(dr_0, dv_0, e, theta0, theta, h, p, T, t0):
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

def from_cartesian(R, V):
    K = np.array([0, 0, 1.0])  # 3rd basis vector

    # 1. Calc distance
    r = np.linalg.norm(R, ord=2)

    # 2. Calc  speed
    v = np.linalg.norm(V, ord=2)

    # 3. Calc radial velocity
    v_r = np.dot(R.flat, V.flat) / r

    # 4. Calc specific angular momentum
    H = np.cross(R.flat, V.flat)

    # 5. Calc magnitude of specific angular momentum
    h = np.linalg.norm(H, ord=2)

    # 6. Calc inclination
    i = np.arccos(H[2] / h)

    # 7. Calculate Node line
    N = np.cross(K, H)

    # 8. Calculate magnitude of N
    n = np.linalg.norm(N, ord=2)

    # 9. calculate RAAN
    O = np.arccos(N[0] / n)
    if N[1] < 0:
        O = 2 * np.pi - O

    # 10. calculate eccentricity vector  / 11. Calc eccentricity
    E = 1 / mu_earth * ((v ** 2 - mu_earth / r) * R.flat - r * v_r * V.flat)
    e = np.linalg.norm(E, ord=2)

    # direct form:
    # self.e = 1 / Constants.mu_earth * np.sqrt(
    #    (2 * Constants.mu_earth - r * v ** 2) * r * v_r ** 2 + (Constants.mu_earth - r * v ** 2) ** 2)

    # 11. Calculate arg. of perigee
    if e != 0.0:
        P = E / (n * e)
        w = np.arccos(np.dot(N, P))
        if E[2] < 0:
            w = 2 * np.pi - w
    else:
        w = 0

    # 12. Calculate the true anomaly
    # p2 = np.log(self.e)+np.log(r)
    if e != 0.0:
        v = np.arccos(np.dot(E, R.flat) / (e*r))
    else:
        v = np.arccos(np.dot(N, R.flat) / (n * r))

    if v_r < 0:
        v = 2 * np.pi - v

    # 13. Calculate semimajor axis
    rp = h ** 2 / mu_earth * 1 / (1 + e)
    ra = h ** 2 / mu_earth * 1 / (1 - e)
    a = 0.5 * (rp + ra)

    # 14. Calculate period (in [s])
    period = 2 * np.pi / np.sqrt(mu_earth) * (pow(a, 1.5))

    return a, e, i, O, w, v

mu_earth = 398600

R_0 = np.array([-749.0, 742.0, -7000.0])
V_0 = np.array([5.84, 4.70, -0.12])

n = np.sqrt(mu_earth / np.linalg.norm(R_0) ** 3.0)
T = 2600

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

B_0 = B(R_0, V_0)

dr_i = np.array([4.0, 4.0, 0.0])
dv_0 = np.linalg.inv(phi_rv(T)).dot(dr_i)

R_T_i, V_T_i = pk.propagate_lagrangian(R_0, V_0, T, mu_earth)

B_i = B(R_T_i, V_T_i)

R_i = np.array(R_T_i) + np.linalg.inv(B_i).dot(dr_i)
sol = pk.lambert_problem(R_0, R_i, T, mu_earth, True)
R_i, V_i = pk.propagate_lagrangian(R_0, np.array(sol.get_v1()[0]), T, mu_earth)

r = R_0 + np.array([0.04, 0.1, 0.01])
v = V_0 + np.array([0.002, 0.001, 0.0])
r_T = R_0
v_T = V_0
dr_0 = B_0.dot(r - r_T)
dv_0 = B_0.dot(v - v_T)

H = np.cross(r_T.flat, v_T.flat)
h_T = np.linalg.norm(H, ord=2)
p_T = h_T**2 / mu_earth

a, e_T, i, O, w, theta_T_0 = from_cartesian(r_T, v_T)

Tmax = 100

C = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

for t in xrange(1, Tmax):
    r, v = pk.propagate_lagrangian(r, v, 1, mu_earth)
    r_T, v_T = pk.propagate_lagrangian(r_T, v_T, 1, mu_earth)

    r = np.array(r)
    v = np.array(v)
    r_T = np.array(r_T)
    v_T = np.array(v_T)

    r_CW = phi_rr(t).dot(dr_0) + phi_rv(t).dot(dv_0)
    v_CW = phi_vr(t).dot(dr_0) + phi_vv(t).dot(dv_0)

    B_t = B(r_T, v_T)

    a, e_T, i, O, w, theta_T = from_cartesian(r_T, v_T)

    IP, OP = chinese_solver(dr_0, dv_0, e_T, theta_T_0, theta_T, h_T, p_T, t, 0)

    trafo = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0]
    ])

    err_pos = np.linalg.norm(r_CW - B_t.dot(r - r_T))
    err_vel = np.linalg.norm(v_CW - B_t.dot(v - v_T))
    err_chin_pos = np.linalg.norm(trafo.dot(np.array([IP[0], IP[1], OP[0]])) - B_t.dot(r - r_T))
    err_chin_vel = np.linalg.norm(trafo.dot(np.array([IP[2], IP[3], OP[1]])) - B_t.dot(v - v_T))

    plt.semilogy(t, err_pos, 'k.')
    plt.semilogy(t, err_vel, 'r.')
    plt.semilogy(t, err_chin_pos, 'ko')
    plt.semilogy(t, err_chin_vel, 'ro')

plt.show()