# import OrekitPropagator

from propagator.OrekitPropagator import OrekitPropagator
from space_tf import KepOrbElem, CartesianLVLH, Cartesian
import numpy as np
import yaml
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import pykep as pk

start_date = datetime.utcnow()

init_state_ch = KepOrbElem()
init_state_ta = KepOrbElem()

init_state_ta.a = 7075.384
init_state_ta.e = 0.0003721779
init_state_ta.i = 1.727
init_state_ta.O = 0.74233
init_state_ta.w = 1.628
init_state_ta.v = 4.67845

init_state_ch.a = 7071.443
init_state_ch.e = (init_state_ta.a * init_state_ta.e+0.12) / init_state_ch.a
init_state_ch.i = 1.727
init_state_ch.O = 0.74234
init_state_ch.w = 1.628
init_state_ch.v = 4.67076

settings_path = '/home/dfrey/cso_ws/src/rdv-cap-sim/simulator/cso_gnc_sim/cfg/chaser.yaml'
settings_file = file(settings_path, 'r')
propSettings = yaml.load(settings_file)
propSettings = propSettings['propagator_settings']

OrekitPropagator.init_jvm()
prop_chaser = OrekitPropagator()
# get settings from yaml file
prop_chaser.initialize(propSettings,
                       init_state_ch,
                       datetime.utcnow())


settings_path = '/home/dfrey/cso_ws/src/rdv-cap-sim/simulator/cso_gnc_sim/cfg/target.yaml'
settings_file = file(settings_path, 'r')
propSettings = yaml.load(settings_file)
propSettings = propSettings['propagator_settings']


prop_target = OrekitPropagator()
# get settings from yaml file
prop_target.initialize(propSettings,
                       init_state_ta,
                       start_date)

chaser_kep = KepOrbElem()
target_kep = KepOrbElem()

chaser_LVLH = CartesianLVLH()

chaser_cart = Cartesian()
target_cart = Cartesian()
chaser_cart.from_keporb(init_state_ch)
target_cart.from_keporb(init_state_ta)

r_t_i = target_cart.R
v_t_i = target_cart.V
r_i = chaser_cart.R
v_i = chaser_cart.V

e = []
e_mean = []
sum_w_v = []
sum_w_v_mean = []

lvlh_r = []
lvlh_v = []
lvlh_h = []
lvlh_r_mean = []
lvlh_v_mean = []
lvlh_h_mean = []

lvlh_r_pk = []
lvlh_v_pk = []

diff_r = []
diff_r_pk = []

mu_earth = 398600.4

tmax = 10000
dt = 10
for i in xrange(1, tmax, dt):
    r_t, v_t = pk.propagate_lagrangian(r_t_i, v_t_i, i, mu_earth)
    r, v = pk.propagate_lagrangian(r_i, v_i, i, mu_earth)

    target_cart.R = np.array(r_t)
    target_cart.V = np.array(v_t)

    diff_r_pk.append(np.linalg.norm(np.array(r)) - np.linalg.norm(np.array(r_t)))

    dr_teme = np.array(r) - np.array(r_t)
    dr_lvlh = target_cart.get_lof().dot(dr_teme)

    lvlh_r_pk.append(dr_lvlh[0])
    lvlh_v_pk.append(dr_lvlh[1])

    chaser = prop_chaser.propagate(start_date + timedelta(seconds=i))
    target = prop_target.propagate(start_date + timedelta(seconds=i))

    chaser_cart = chaser[0]
    target_cart = target[0]

    diff_r.append(np.linalg.norm(chaser_cart.R) - np.linalg.norm(target_cart.R))

    chaser_kep.from_cartesian(chaser_cart)

    chaser_LVLH.from_cartesian_pair(chaser_cart, target_cart)

    lvlh_r.append(chaser_LVLH.R[0])
    lvlh_v.append(chaser_LVLH.R[1])
    lvlh_h.append(chaser_LVLH.R[2])

    e.append(chaser_kep.a)
    # plt.plot(i, chaser_kep.a, 'k.', markersize=5)
    sum_w_v.append(chaser_kep.v + chaser_kep.w)

    i_osc = chaser_kep.i

    chaser_kep.from_osc_elems(chaser_kep)
    # e_mean.append((i_osc - chaser_kep.i)* np.linalg.norm(chaser_cart.R))
    # plt.plot(i, chaser_kep.a, 'r.', markersize=5)
    sum_w_v_mean.append(chaser_kep.v + chaser_kep.w)



# plt.plot(np.arange(1, tmax, dt), diff_r, 'b-')
# plt.plot(np.arange(1, tmax, dt), diff_r_pk, 'r--')
plt.plot(lvlh_v, lvlh_r, 'b-')
plt.plot(lvlh_v_pk, lvlh_r_pk, 'r--')
# plt.plot(lvlh_v_mean, lvlh_r_mean, 'r--')
# plt.plot(np.arange(1, tmax, dt), e, 'r--')
# plt.plot(np.arange(1, tmax, dt), e_mean, 'b-')
plt.show()
