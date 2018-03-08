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

def main():
    # Import scenario and initial conditions
    scenario = Scenario()
    # scenario.set_propagator_type('2-body')
    scenario.import_yaml_scenario()

    # Solve scenario
    solver = Solver()
    solver.set_ic(scenario)
    solver.solve_scenario()

    # Save manoeuvre plan
    scenario.export_solved_scenario(solver.manoeuvre_plan)

if __name__ == "__main__":
    main()

