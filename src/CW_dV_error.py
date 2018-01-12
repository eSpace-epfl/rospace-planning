import matplotlib.pyplot as plt
import numpy as np
import pykep as pk

def B(R, V):
    i = R / np.linalg.norm(R)
    j_ = V / np.linalg.norm(V)
    H = np.cross(i, j_)
    k = H / np.linalg.norm(H)
    j = -np.cross(i, k)
    B_0 = np.identity(3)
    B_0[0, 0:3] = i
    B_0[1, 0:3] = j
    B_0[2, 0:3] = k

    return B_0



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

mu_earth = 398600.4418

R_T_0 = np.array([1622.39, 5305.10, 3717.44])
V_T_0 = np.array([-7.29977, 0.492357, 2.48318])
a_T, e_T, i_T, O_T, w_T, theta_T = from_cartesian(R_T_0, V_T_0)

R_0 = np.array([1612.75, 5310.19, 3750.33])
V_0 = np.array([-7.35321, 0.463856, 2.46920])

n = np.sqrt(mu_earth / np.linalg.norm(R_T_0) ** 3.0)
T = 2400.0

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

B_0 = B(R_T_0, V_T_0)
O_T = n * np.cross(R_T_0, V_T_0)/np.linalg.norm(np.cross(R_T_0, V_T_0))

dr_0 = R_0 - R_T_0
dv_0 = V_0 - V_T_0 - np.cross(O_T, dr_0)

dr_0_LVLH = B_0.dot(dr_0)
dv_0_LVLH = B_0.dot(dv_0)

DeltaV = -np.linalg.inv(phi_rv(T)).dot(phi_rr(T).dot(dr_0_LVLH)) - dv_0_LVLH

r, v = pk.propagate_lagrangian(R_0, V_0 + np.linalg.inv(B_0).dot(DeltaV), T, mu_earth)
r_T, v_T = pk.propagate_lagrangian(R_T_0, V_T_0, T, mu_earth)

# R_0 = R_T_0 + np.linalg.inv(B_0).dot(dr_0)
# V_0 = V_T_0 + np.linalg.inv(B_0).dot(dv_0)
# a_C, e_C, i_C, O_C, w_C, theta_C = from_cartesian(R_0, V_0)
#
# dv_0 = np.linalg.inv(phi_rv(T)).dot(dr_i - phi_rr(T).dot(dr_0))

R_T_i, V_T_i = pk.propagate_lagrangian(R_T_0, V_T_0, T, mu_earth)

B_i = B(R_T_i, V_T_i)

# R_i = np.array(R_T_i) + np.linalg.inv(B_i).dot(dr_i)

# print R_0, R_i

sol = pk.lambert_problem(R_0, R_T_i, 28800, mu_earth, False)

R_i, V_i = pk.propagate_lagrangian(R_0, np.array(sol.get_v1()[0]), T, mu_earth)

dr_i_CW = phi_rr(T).dot(dr_0) + phi_rv(T).dot(dv_0)
dr_i_PK = B_i.dot(np.array(R_i) - np.array(R_T_i))

print dv_0
print B_0.dot(sol.get_v1()[8] - V_T_0)

print "Error: " + str(np.linalg.norm(dv_0 - B_0.dot(sol.get_v1()[0] - V_T_0))) + "\n"

dv_i = phi_vr(T).dot(dr_0) + phi_vv(T).dot(dv_0)

print dv_i
print B_i.dot(sol.get_v2()[0] - np.array(V_T_i))

print "Error: " + str(np.linalg.norm(dv_i - B_i.dot(sol.get_v2()[0] - np.array(V_T_i)))) + "\n"

Tmax = T

r = R_0
v = sol.get_v1()[0]
r_T = R_T_0
v_T = V_T_0

for t in xrange(1, int(Tmax)):
    r, v = pk.propagate_lagrangian(r, v, 1, mu_earth)
    r_T, v_T = pk.propagate_lagrangian(r_T, v_T, 1, mu_earth)

    r = np.array(r)
    v = np.array(v)
    r_T = np.array(r_T)
    v_T = np.array(v_T)

    r_CW = phi_rr(t).dot(dr_0) + phi_rv(t).dot(dv_0)
    v_CW = phi_vr(t).dot(dr_0) + phi_vv(t).dot(dv_0)

    B_t = B(r_T, v_T)
    dr_t = B_t.dot(r - r_T)
    dv_t = B_t.dot(v - v_T)


    # a, e_T, i, O, w, theta_T = from_cartesian(r_T, v_T)
    #
    # IP, OP = chinese_solver(dr_0, dv_0, e_T, theta_T_0, theta_T, h_T, p_T, t, 0)
    #
    # trafo = np.array([
    #     [0.0, -1.0, 0.0],
    #     [1.0, 0.0, 0.0],
    #     [0.0, 0.0, -1.0]
    # ])

    err_pos = np.linalg.norm(r_CW - dr_0)
    err_vel = np.linalg.norm(v_CW - dv_0)
    # err_chin_pos = np.linalg.norm(trafo.dot(np.array([IP[0], IP[1], OP[0]])) - B_t.dot(r - r_T))
    # err_chin_vel = np.linalg.norm(trafo.dot(np.array([IP[2], IP[3], OP[1]])) - B_t.dot(v - v_T))

    plt.plot(r_CW[0], r_CW[1], 'b.')
    plt.plot(dr_t[0], dr_t[1], 'r.')

    # plt.semilogy(t, err_pos, 'k.')
    # plt.semilogy(t, err_vel, 'r.')
    # plt.semilogy(t, err_chin_pos, 'ko')
    # plt.semilogy(t, err_chin_vel, 'ro')

plt.show()