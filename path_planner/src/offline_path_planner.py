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


def main(scenario_name, ic_name='std_ic', save=False):
    """Run the offline path planner.

    This function import the scenario named 'scenario_name' from the folder /cfg/ and takes the initial conditions
    named 'ic_name' from the yaml file /cfg/initial_conditions.yaml.

    After importing, it solves it and saves it (if save flag is true) in a .pickle file in the /example/ folder.

    Args:
        scenario_name (str): The name of the scenario which should be solved.
        ic_name (str): The initial condition name which should be used.
        save (bool): Boolean flag to save and store the result.

    Return:
        tot_dV (float): Total amount of deltaV consumed in km/s.
        chaser (Chaser): Chaser state after the mission (used for testing).

    """

    # Import scenario and initial conditions
    scenario = Scenario()
    scenario.import_yaml_scenario(scenario_name, ic_name)

    # Solve scenario
    solver = Solver()
    solver.initialize_solver(scenario)
    solver.solve_scenario()

    # Save manoeuvre plan
    if save:
        scenario.export_solved_scenario(solver.manoeuvre_plan)

    return solver.tot_dV, solver.chaser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario_name',
                        help='Name of the scenario which should be called. If not called use the one stated in main().')
    parser.add_argument('--ic_name',
                        help='Name of initial condition set. If not called, use "std_ic".')
    parser.add_argument('--save',
                        help='Save result in .pickle file. If not called do not save.')
    args = parser.parse_args()

    if args.ic_name:
        ic_name = args.ic_name
    else:
        ic_name = 'camille_homing'

    if args.scenario_name:
        scenario_name = args.scenario_name
    else:
        scenario_name = 'scenario_camille_homing'

    if args.save:
        save = True
    else:
        save = False

    main(scenario_name, ic_name, save)
