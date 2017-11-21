from solver import Solver
import numpy as np

s = Solver()

s.clohessy_wiltshire_solver(7063, np.array([2, -5, 0]), np.array([0,0,0]), 30000)

print s.cw_sol