
import matplotlib.pyplot as plt
import numpy as np

from state import Satellite
from rospace_lib import KepOrbElem, OscKepOrbElem
from path_planning_propagator import Propagator
from datetime import timedelta, datetime
from solver import Solver
from orbit_adjuster import HohmannTransfer
from checkpoint import AbsoluteCP, CheckPoint
from scenario import Scenario
from mpl_toolkits.mplot3d import Axes3D

sat = Satellite()
sat.initialize_satellite('target', datetime.utcnow())
sat.prop.prop_type = 'real-world'

# Osculating orbital elements
osc = OscKepOrbElem()
osc.from_cartesian(sat.abs_state)

# Mean orbital elements
mean = KepOrbElem()
mean.from_osc_elems(osc)

print 'something shady happens there'

# # Propagate to v = 0
# solver = Solver()
# dt_mean = solver.travel_time(mean, mean.v, 0.0)
# dt_osc = solver.travel_time(osc, osc.v, 0.0)
#
# dt = dt_osc
#
# print mean.v
# print dt_mean
# print dt_osc
#
# target_prop = prop.propagator.propagate(prop.date + timedelta(seconds=0.0))
#
# osc.from_cartesian(target_prop[0])
# mean.from_osc_elems(osc)
#
# print mean.v
# print osc.v

# print target_prop[0].R
# print sat.abs_state.R
# print target_prop[0].V
# print sat.abs_state.V

print "TEST ORBIT ADJUSTER"
print "Starting semimajor axis: " + str(mean.a)
print "Starting eccentricity: " + str(mean.e)

state_chkp = KepOrbElem()
state_chkp.from_osc_elems(osc)
state_chkp.a += 500
state_chkp.e += 0.003

chkp = CheckPoint()
chkp.set_state(state_chkp)
ht = HohmannTransfer(sat, chkp)
dV = ht.evaluate_manoeuvre()


scenario = Scenario()
scenario.target_ic.set_from_satellite(sat)

solver = Solver()
solver.initialize_solver(scenario)
solver.create_manoeuvres(dV)

plt.figure(1)

# fig3d = plt.figure(1)
# ax = fig3d.add_subplot(111, projection='3d')

for t in xrange(0, 10000, 100):
    target_prop = sat.prop.orekit_prop.propagate(sat.prop.date + timedelta(seconds=t))
    sat.set_abs_state_from_cartesian(target_prop[0])

    # ax.scatter(sat.abs_state.R[0], sat.abs_state.R[1], sat.abs_state.R[2], 'ro')

    osc_kep = sat.get_osc_oe()
    mean_kep = sat.get_mean_oe('real-world')

    # plt.plot(t, osc_kep.v, 'ro')
    # plt.plot(t, osc_kep.w, 'go')
    plt.subplot(221)
    plt.plot(t, osc_kep.v, 'yo')
    plt.plot(t, mean_kep.v, 'ro')

    plt.subplot(222)
    plt.plot(t, osc_kep.a, 'bo')
    plt.plot(t, mean_kep.a, 'ro')

    plt.subplot(223)
    plt.plot(t, osc_kep.i, 'ko')
    plt.plot(t, mean_kep.i, 'ro')

    plt.subplot(224)
    plt.plot(t, osc_kep.O, 'go')
    plt.plot(t, mean_kep.O, 'ro')

plt.show()