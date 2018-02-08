from scenario import Scenario
from solver import Solver

def main():
    # Import scenario and initial conditions
    scenario = Scenario()
    scenario.import_yaml_scenario()

    # Solve scenario
    solver = Solver()
    solver.solve_scenario(scenario)

    # Save manoeuvre plan
    scenario.export_solved_scenario(solver.manoeuvre_plan)

if __name__  == "__main__":
    main()