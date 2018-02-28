# import OrekitPropagator

from propagator.OrekitPropagator import OrekitPropagator
from space_tf import KepOrbElem
import numpy as np
import yaml
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt

start_date = datetime.utcnow()

init_state_ch = KepOrbElem()
init_state_ta = KepOrbElem()

init_state_ta.a = 6750
init_state_ta.e = 0.0004
init_state_ta.i = 30*np.pi/180
init_state_ta.O = 60*np.pi/180
init_state_ta.w = 30*np.pi/180
init_state_ta.v = 0.2

init_state_ch.a = 7084
init_state_ch.e = 0.0005
init_state_ch.i = 30.05*np.pi/180
init_state_ch.O = 59.96*np.pi/180
init_state_ch.w = 30.1*np.pi/180
init_state_ch.v = 0.189

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

e = []
e_mean = []
sum_w_v = []
sum_w_v_mean = []

tmax = 10000
dt = 500
for i in xrange(1, tmax, dt):
    chaser = prop_chaser.propagate(start_date + timedelta(seconds=i))

    chaser_cart = chaser[0]

    chaser_kep.from_cartesian(chaser_cart)
    e.append(chaser_kep.v)
    # plt.plot(i, chaser_kep.a, 'k.', markersize=5)
    sum_w_v.append(chaser_kep.v + chaser_kep.w)

    chaser_kep.from_osc_elems(chaser_kep)
    e_mean.append(chaser_kep.v)
    # plt.plot(i, chaser_kep.a, 'r.', markersize=5)
    sum_w_v_mean.append(chaser_kep.v + chaser_kep.w)


plt.plot(np.arange(1, tmax, dt), sum_w_v, 'r--')
plt.plot(np.arange(1, tmax, dt), sum_w_v_mean, 'b-')
plt.show()
