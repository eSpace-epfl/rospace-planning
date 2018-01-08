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

mu_earth = 398600

R_0 = np.array([-749.0, 742.0, -7000.0])
V_0 = np.array([5.84, 4.70, -0.12])

n = np.sqrt(mu_earth / np.linalg.norm(R_0) ** 3.0)
T = 1

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

print "Sol V1:    " + str(np.array(sol.get_v1()[0]))
print "Sol V2:    " + str(np.array(sol.get_v2()[0]))

print "\nDeltaV1_TEME = SolV1 - V_0                 ---> " + str(np.array(sol.get_v1()[0]) - V_0)
print "DeltaV1_LVLH = dv_0                        ---> " + str(dv_0)
print "DeltaV1_TEME = inv(B_0) * DeltaV1_LVLH     ---> " + str(np.linalg.inv(B_0).dot(dv_0))

dV_i_TEME = np.array(V_T_i) - np.array(V_i)
r_i_RSW = B_i.dot(np.array(R_i))

print "\ndV_i_LVLH = phi_vr(T) * dr_0 + phi_vv(T) * dv_0 ---> " + str(phi_vv(T).dot(dv_0))
print "dV_i_TEME = V_T_i - V_i                         ---> " + str(dV_i_TEME)
print "dV_i_LVLH = B_i * dV_i_TEME                     ---> " + str(np.linalg.inv(B_i).dot(np.array(V_T_i) - np.array(V_i)))
print "dV_i_TEME = inv(B_i) * phi_vv(T) * dv_0         ---> " + str(np.linalg.inv(B_i).dot(phi_vv(T).dot(dv_0)))

print "\n"
#
# dr_T2 = [0.0, 10.0, 0.0]
#
# dv_0 = np.linalg.inv(phi_rv(T)).dot(dr_T2 - phi_rr(T).dot(dr_T)) - phi_vv(T).dot(dv_0)
#
# print dv_0
#
# R_T_t3, V_T_t3 = pk.propagate_lagrangian(R_T_t2, V_T_t2, T, mu_earth)
#
# i = R_T_t3 / np.linalg.norm(R_T_t3)
# H = np.cross(R_T_t3, V_T_t3)
# k = H / np.linalg.norm(H)
# j = -np.cross(i, k)
#
# B = np.identity(3)
# B[0, 0:3] = i
# B[1, 0:3] = j
# B[2, 0:3] = k
#
# R_T_next = np.array(R_T_t3) + np.linalg.inv(B).dot(dr_T2)
#
# print R_T_t3
# print R_T_next
#
# sol = pk.lambert_problem(R_T_t2, R_T_next, T, mu_earth, True)
#
# i = R_T_t2 / np.linalg.norm(R_T_t2)
# H = np.cross(R_T_t2, V_T_t2)
# k = H / np.linalg.norm(H)
# j = -np.cross(i, k)
#
# B_i = np.identity(3)
# B_i[0, 0:3] = i
# B_i[1, 0:3] = j
# B_i[2, 0:3] = k
#
#
# print B_i.dot(sol.get_v1()[0] - np.array(V_T_t2))