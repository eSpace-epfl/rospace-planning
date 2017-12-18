from math import sqrt,pi,atan,tan,sin,cos
from space_tf import *
from PyKEP import *
import numpy as np
import rospy

class OptimizationProblem:

    def __init__(self):
        self.objective = 0;
        self.constraints = ();
        self.jacobian_objective = 0;

        print "Initialize Problem"

    def init_objective_function(self, objective):
        self.objective = objective

    def init_objective_function_derivative(self, jacobian):
        self.jacobian_objective = jacobian

    def add_constraint(self, type, function, jacobian):
        self.constraints += {'type': type, 'fun': function, 'jac':jacobian}

