from scenario import Scenario
from solver import Solver
# from yaml import load, dump
# try:
#     from yaml import CLoader as Loader, CDumper as Dumper
# except ImportError:
#     from yaml import Loader, Dumper
#--------------------

# Import scenario and initial conditions
scenario = Scenario()
scenario.import_yaml_scenario()

# Solve scenario
solver = Solver()
solver.solve_scenario(scenario)
