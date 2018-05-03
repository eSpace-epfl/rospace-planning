# @copyright Copyright (c) 2017, Davide Frey (frey.davide.ae@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

"""Main file to solve a scenario offline."""

import argparse

from scenario import Scenario
from solver import Solver
from datetime import datetime


def main(date):
    """
        Run the offline path planner.
        This function import automatically the scenario stated in:
            cfg/scenario.yaml
        and import initial conditions in:
            cfg/initial_conditions.yaml

        After importing, it solves it and finally saves it in a .pickle file in the example/ folder.
    """

    # Import scenario and initial conditions
    # initial_conditions.yaml
    scenario = Scenario(date)
    scenario.import_yaml_scenario()

    # Solve scenario
    solver = Solver(date)
    solver.initialize_solver(scenario)
    solver.solve_scenario()

    # Save manoeuvre plan
    scenario.export_solved_scenario(solver.manoeuvre_plan)

if __name__ == "__main__":
    date = datetime.utcnow()
    main(date)
