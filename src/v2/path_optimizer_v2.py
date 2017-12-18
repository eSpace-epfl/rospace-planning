import pykep as pk
import numpy as np
import rospy
import datetime as dt
import scipy.io as sio
import epoch_clock

from scenario_v2 import Scenario, Position
from solver_v2 import Solver, Command
from space_tf import *

epoch = epoch_clock.Epoch()

class TrajectoryController:

    def __init__(self):
        self.chaser = Position()
        self.target = Position()

        self.tau_C = 0
        self.tau_T = 0
        self.m_C = 0
        self.m_T = 0

        # Scenario variables
        self.scenario = Scenario()
        self.scenario_flag = False

        # Solver variables
        self.solver = Solver()
        self.active_command = Command()
        self.previous_command = True

        self.sleep_flag = False

    def run_command(self):

        if self.previous_command and len(self.solver.command_line) > 0:
            # Previous command has been executed, run the next
            # Extract first command
            self.active_command = self.solver.command_line.pop(0)
            self.previous_command = False
        elif len(self.solver.command_line) == 0:
            self.sleep_flag = False

        if self.active_command.true_anomaly <= self.chaser.kep.v and epoch.now() >= self.scenario.mission_start \
                and len(self.solver.command_line) > 0 and epoch.now() >= self.active_command.epoch:
            # Command may be applied
            self.sleep_flag = True
            self.previous_command = True

    def check_trajectory(self):
        """
            If the trajectory stays within a certain cone error (pre-calculated, for which the trajectory is
            always safe & well designed), then no correction needs to be done.
            Otherwise some correction should be made, and maybe also the scenario should be somehow rescheduled.
        """

    def callback(self, target_oe, chaser_oe):
        # Update target and chaser position from messages
        self.target.kep.from_message(target_oe.position)
        self.chaser.kep.from_message(chaser_oe.position)

        # Transform into mean orbital elements
        self.chaser.kep.osc_elems_transformation(self.chaser.kep, True)
        self.target.kep.osc_elems_transformation(self.target.kep, True)

        # Update other coordinate frame from mean orbital elements
        self.target.cartesian.from_keporb(self.target.kep)
        self.chaser.cartesian.from_keporb(self.chaser.kep)
        self.chaser.rel_kep.from_keporb(self.target.kep, self.chaser.kep)
        self.chaser.lvlh.from_cartesian_pair(self.chaser.cartesian, self.target.cartesian)

        dt = epoch.now() - epoch.epoch_datetime
        dt = dt.seconds
        n_C = np.sqrt(mu_earth / self.chaser.kep.a ** 3.0)
        n_T = np.sqrt(mu_earth / self.target.kep.a ** 3.0)

        # If scenario has not been created yet or if there are no available, locally saved solution
        if not self.scenario_flag:
            self.scenario.create_scenario(self.target.kep, self.chaser.kep)
            try:
                self.solver.command_line = self.scenario.import_solved_scenario()
            except:
                # Solve scenario
                self.solver.solve_scenario(self.scenario, self.chaser, self.target)
            self.scenario_flag = True

            # Simulation has just been started, set tau_T and tau_C giving as t0 = 0 the start of the simulation
            self.tau_C = dt - self.chaser.kep.m / n_C
            self.tau_T = dt - self.target.kep.m / n_T
        else:
            # Scenario already solved, only need to apply outputs through the trajectory controller
            self.run_command()

        # Update mean anomaly (so that it doesn't reset at v = 2*pi)
        self.m_C = n_C * (dt - self.tau_C)
        self.m_T = n_T * (dt - self.tau_T)