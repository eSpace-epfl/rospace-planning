import pykep as pk
import numpy as np
import rospy
import datetime as dt
import scipy.io as sio

from scenario_v2 import Scenario, Position
from solver_v2 import Solver, Command
from space_tf import *


class TrajectoryController:

    def __init__(self):
        self.chaser = Position()
        self.target = Position()

        # Scenario variables
        self.scenario = Scenario()
        self.scenario_flag = False

        # Solver variables
        self.solver = Solver()
        self.active_command = Command()

        self.sleep_flag = False

    def callback(self, target_oe, chaser_oe):
        # Update target and chaser position from messages
        self.target.kep.from_message(target_oe.position)
        self.target.cartesian.from_keporb(self.target.kep)

        self.chaser.kep.from_message(chaser_oe.position)
        self.chaser.cartesian.from_keporb(self.chaser.kep)
        self.chaser.rel_kep.from_keporb(self.target.kep, self.chaser.kep)
        self.chaser.lvlh.from_cartesian_pair(self.chaser.cartesian, self.target.cartesian)

        # If scenario has not been created yet or if there are no available, locally saved solution
        if not self.scenario_flag:
            self.scenario.create_scenario(self.target.kep, self.chaser.kep)
            self.scenario_flag = True

            # Solve scenario
            self.solver.solve_scenario(self.scenario, self.chaser, self.target)

        elif self.scenario.import_solved_scenario():
            # Scenario already solver, only need to apply outputs through the trajectory controller
            # Import scenario
            pass

        else:
            # Define the active command and wait until it's time to output it
            # TODO: Orientation time!

            self.active_command = self.solver.command_line.pop(0)
