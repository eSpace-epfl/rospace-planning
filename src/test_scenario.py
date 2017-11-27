from solver import Solver
import numpy as np
from optimal_transfer_time import *
from space_tf.Constants import Constants as const


opt = OptimalTime()

r1 = np.array([7000, 0, 0])
v1 = np.array([0, 7.54605, 0])
r2 = np.array([-4499.9999999999982, -7794.2286340599485, 0])
v2 = np.array([7.50586*cos(pi/3), -6.8586*sin(pi/3), 0])

t_opt = opt.find_optimal_trajectory_time(r1, v1, r2, v2, 60000, 60000)
print t_opt