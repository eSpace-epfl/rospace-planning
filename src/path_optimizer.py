import epoch_clock

from scenario import Scenario
from solver import Solver
from manoeuvre import Manoeuvre
from state import State

epoch = epoch_clock.Epoch()


class TrajectoryController:

    def __init__(self):
        # Estimated positions of chaser and target
        self.chaser_est = State()
        self.target_est = State()

        self.tau_C = 0
        self.tau_T = 0
        self.m_C = 0
        self.m_T = 0

        # Scenario variables
        self.scenario = Scenario()
        self.scenario_flag = False

        # Solver variables
        self.solver = Solver()
        self.active_manoeuvre = Manoeuvre()
        self.previous_manoeuvre = True

        self.sleep_flag = False

    def run_command(self):

        print "Actual anomaly: " + str(self.chaser_est.kep.v)

        if self.previous_manoeuvre and len(self.solver.manoeuvre_plan) > 0:
            # Previous command has been executed, run the next
            # Extract first command
            self.active_manoeuvre = self.solver.manoeuvre_plan.pop(0)
            self.previous_manoeuvre = False
            print "Needed anomaly: " + str(self.active_manoeuvre.true_anomaly)

        elif len(self.solver.manoeuvre_plan) == 0:
            self.sleep_flag = False

        if abs(self.active_manoeuvre.true_anomaly - self.chaser_est.kep.v) < 1e-2 and len(self.solver.manoeuvre_plan) > 0 \
            and self.active_manoeuvre.theta_diff == None:
            # Command may be applied
            self.sleep_flag = True
            self.previous_manoeuvre = True
            print "MANOEUVRE EXECUTED"
        elif self.active_manoeuvre.theta_diff != None:
            pass


    def check_trajectory(self):
        """
            If the trajectory stays within a certain cone error (pre-calculated, for which the trajectory is
            always safe & well designed), then no correction needs to be done.
            Otherwise some correction should be made, and maybe also the scenario should be somehow rescheduled.
        """

    def callback(self, target_oe, chaser_oe):
        # Update target and chaser position from messages
        self.target_est.kep.from_message(target_oe.position)
        self.chaser_est.kep.from_message(chaser_oe.position)

        # Transform into mean orbital elements
        self.chaser_est.kep.osc_elems_transformation(self.chaser_est.kep, True)
        self.target_est.kep.osc_elems_transformation(self.target_est.kep, True)

        dt = epoch.now() - epoch.epoch_datetime
        dt = dt.seconds
        # n_C = np.sqrt(mu_earth / self.scenario.chaser.kep.a ** 3.0)
        # n_T = np.sqrt(mu_earth / self.scenario.target.kep.a ** 3.0)

        if not self.scenario_flag:
            # Import scenario
            self.scenario.import_yaml_scenario()

            # Import scenario offline solution
            self.solver.manoeuvre_plan = self.scenario.import_solved_scenario()

            self.scenario_flag = True
        else:
            self.run_command()

        # # If scenario has not been created yet or if there are no available, locally saved solution
        # if not self.scenario_flag:
        #     self.scenario.import_yaml_scenario()
        #     try:
        #         self.solver.manoeuvre_plan = self.scenario.import_solved_scenario()
        #     except:
        #         # Solve scenario
        #         self.solver.solve_scenario(self.scenario)
        #     self.scenario_flag = True
        #
        #     # Simulation has just been started, set tau_T and tau_C giving as t0 = 0 the start of the simulation
        #     # self.tau_C = dt - self.scenario.chaser.kep.m / n_C
        #     # self.tau_T = dt - self.scenario.target.kep.m / n_T
        # else:
        #     # Scenario already solved, only need to apply outputs through the trajectory controller
        #     self.run_command()

        # Update mean anomaly (so that it doesn't reset at v = 2*pi)
        # self.m_C = n_C * (dt - self.tau_C)
        # self.m_T = n_T * (dt - self.tau_T)