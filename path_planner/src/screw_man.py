import numpy as np
import trajectory_plotter

from manoeuvre import Manoeuvre
from state import Satellite, Chaser
from checkpoint import RelativeCP
from datetime import datetime
from orbit_adjuster import *
from rospace_lib import mu_earth
from scenario import Scenario
from copy import deepcopy

# Initial conditions = -2, -10 ,0

# date = datetime(2014,1,1,12,0,0,0)

chaser = Chaser()
# chaser.abs_state.R = np.array([ 1352.15481415, -3080.1794726,   6240.56929186])
# chaser.abs_state.V = np.array([-4.63650391,  4.8198207,   3.38747188])
# chaser.mass = 30.0

target = Satellite()
# target.abs_state.R = np.array([ 1346.34426791, -3074.60459724,  6246.82351639])
# target.abs_state.V = np.array([-4.63787066,  4.82374008,  3.3776849 ])
# target.mass = 0.82
#
# print chaser.get_osc_oe()
# print target.get_osc_oe()

target.initialize_satellite('target', 'ic_davide', '2-body')
chaser.initialize_satellite('chaser', 'ic_davide', '2-body', target)


chaser.rel_state.from_cartesian_pair(chaser.abs_state, target.abs_state)

scenario = Scenario()
scenario.chaser_ic = deepcopy(chaser.get_osc_oe())
scenario.target_ic = deepcopy(target.get_osc_oe())
scenario.name = 'Screw manoeuvre'
scenario.ic_name = 'ic_davide'
scenario.prop_type = '2-body'
scenario.date = chaser.prop.date


checkpoint0 = RelativeCP()
checkpoint0.rel_state.R = np.array([-4.0, 12.0, 0.0])
checkpoint0.manoeuvre_type = 'drift'
checkpoint0.error_ellipsoid = np.array([0.1, 0.5, 0.1])

orbitadj = Drift()
man0 = orbitadj.evaluate_manoeuvre(chaser, checkpoint0, target, [])


checkpoint1 = RelativeCP()
checkpoint1.rel_state.R = np.array([0.0, 22.0,  0.0])
checkpoint1.rel_state.V = np.array([0.0,  0.0,  0.0])
checkpoint1.t_min = np.pi * np.sqrt(target.get_mean_oe().a ** 3 / mu_earth) - 450.0
checkpoint1.t_max = np.pi * np.sqrt(target.get_mean_oe().a ** 3 / mu_earth) - 150.0

orbitadj = MultiLambert()

print "\n Evaluating manoeuvre 1..."

man1 = orbitadj.evaluate_manoeuvre(chaser, checkpoint1, target, [0.0,  0.0,  0.0], True)

checkpoint2 = RelativeCP()
checkpoint2.rel_state.R = np.array([0.0, 7.0,  0.0])
checkpoint2.rel_state.V = np.array([0.0,  0.0,  0.0])
# checkpoint2.t_min = np.pi/2.0 * np.sqrt(target.get_mean_oe().a ** 3 / mu_earth)
checkpoint2.t_min = 2*np.pi * np.sqrt(target.get_mean_oe().a ** 3 / mu_earth) - 50.0
checkpoint2.t_max = checkpoint2.t_min + 100.0

print "\n Evaluating manoeuvre 2..."

man2 = orbitadj.evaluate_manoeuvre(chaser, checkpoint2, target, [0.0,  0.0,  0.0], True)

man3 = Manoeuvre()
man3.deltaV = np.linalg.inv(chaser.abs_state.get_lof()).dot([0.0,  0.00075,  -0.0025])
dt = (man2[1].execution_epoch - man2[0].execution_epoch).total_seconds()
man3.execution_epoch = man2[0].execution_epoch + timedelta(seconds=dt/2.0 - 150.0)

man = []
man += man0
man += man1
man += [man2[0]]
# man += [man3]

scenario.export_solved_scenario(man)

print "\n PLOTTING TRAJECTORY"

trajectory_plotter.main(man, scenario, extra_dt=100000)
