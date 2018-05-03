# @copyright Copyright (c) 2017, Davide Frey (frey.davide.ae@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

"""Main file to solve a scenario offline."""

from scenario import Scenario
from solver import Solver
from datetime import datetime

def main(filename, date=datetime.utcnow(), save=False):
    # Import scenario and initial conditions
    scenario = Scenario(date)
    scenario.import_yaml_scenario(filename)

    # Solve scenario
    solver = Solver(date)
    solver.initialize_solver(scenario)
    solver.solve_scenario()

    # Save manoeuvre plan
    if save:
        scenario.export_solved_scenario(solver.manoeuvre_plan)

    return solver.tot_dV

if __name__ == "__main__":
    main('scenario', save=True)

