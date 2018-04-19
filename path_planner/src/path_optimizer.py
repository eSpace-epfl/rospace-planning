import epoch_clock
import numpy as np

from scenario import Scenario
from solver import Solver
from manoeuvre import Manoeuvre
from state import Satellite, Chaser


class TrajectoryController:

    def __init__(self):
        # Estimated positions of chaser and target
        self.chaser_est = Chaser()
        self.target_est = Satellite()

        # Scenario variables
        self.scenario = Scenario()
        self.scenario_flag = False

        # Solver variables
        self.solver = Solver()
        self.active_manoeuvre = Manoeuvre()
        self.previous_manoeuvre = True

        self.sleep_flag = False

    def run_command(self):

        print "Actual anomaly: " + str(self.chaser_est.abs_state.v)

        if self.previous_manoeuvre and len(self.solver.manoeuvre_plan) > 0:
            # Previous command has been executed, run the next
            # Extract first command
            self.active_manoeuvre = self.solver.manoeuvre_plan.pop(0)
            self.active_manoeuvre.add_lock()
            self.previous_manoeuvre = False
            print "Needed anomaly: " + str(self.active_manoeuvre.abs_state.v)

        elif len(self.solver.manoeuvre_plan) == 0:
            self.sleep_flag = False

        if abs(self.active_manoeuvre.abs_state.v - self.chaser_est.abs_state.v) < 1e-2 and len(self.solver.manoeuvre_plan) > 0:
            # Command may be applied
            self.sleep_flag = True
            self.previous_manoeuvre = True

            self.check_trajectory()

            print "MANOEUVRE EXECUTED"

    def check_trajectory(self):
        """
            If the trajectory stays within a certain cone error (pre-calculated, for which the trajectory is
            always safe & well designed), then no correction needs to be done.
            Otherwise some correction should be made, and maybe also the scenario should be somehow rescheduled.
        """

        # Actual estimated position: self.chaser_est / self.target_est

        check_a = self.active_manoeuvre.abs_state.a - self.chaser_est.abs_state.a
        check_e = self.active_manoeuvre.abs_state.e - self.chaser_est.abs_state.e
        check_i = self.active_manoeuvre.abs_state.i - self.chaser_est.abs_state.i
        check_w = self.active_manoeuvre.abs_state.w - self.chaser_est.abs_state.w
        check_O = self.active_manoeuvre.abs_state.O - self.chaser_est.abs_state.O

        print "a:" + str(check_a)
        print "e:" + str(check_e)
        print "i:" + str(check_i)
        print "w:" + str(check_w)
        print "O:" + str(check_O)

        # Set the interval of true anomaly around the point where we want to executed the burn. Inside this interval
        # the computer starts to check if the manoeuvre can be performed or not.
        # This interval can be maybe evaluated depending on the burn duration.
        dv_interval = 1.0 * np.pi / 180.0

        if self.chaser_est.abs_state.v - dv_interval > self.active_manoeuvre.abs_state.v:
            # Can we perform the manoeuvre?
            # Yes -> do stuff
            # No -> do other stuff
            pass

    def callback(self, target_oe, chaser_oe):
        # Update target and chaser position from messages
        self.target_est.abs_state.from_message(target_oe.position)
        self.chaser_est.abs_state.from_message(chaser_oe.position)

        # Transform into mean orbital elements
        self.chaser_est.abs_state.osc_elems_transformation(self.chaser_est.abs_state, True)
        self.target_est.abs_state.osc_elems_transformation(self.target_est.abs_state, True)

        print self.chaser_est.abs_state.a

        # dt = epoch.now() - epoch.epoch_datetime
        # dt = dt.seconds

        if not self.scenario_flag:
            # Import scenario
            self.scenario.import_yaml_scenario()

            # Import scenario offline solution
            self.solver.manoeuvre_plan = self.scenario.import_solved_scenario()

            self.scenario_flag = True
        else:
            self.run_command()
