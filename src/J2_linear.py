import numpy as np
import yaml

from datetime import datetime, timedelta
from propagator.OrekitPropagator import OrekitPropagator
from space_tf import KepOrbElem, CartesianLVLH, Cartesian, mu_earth, J_2, R_earth

from org.orekit.propagation import SpacecraftState
from org.orekit.frames import FramesFactory
from org.orekit.orbits import CartesianOrbit
from org.orekit.utils import PVCoordinates
from org.orekit.utils import Constants as Cst
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.time import AbsoluteDate, TimeScalesFactory


class Individual(object):

    def __init__(self):
        self.delta_v = np.array([0.0, 0.0, 0.0])
        self.fitness = 1e12
        self.arrival = None

    def create_individual(self, seed, mu=0.0):
        # Evaluate a possible delta-v based on the seed given
        self.delta_v[0] = seed[0]
        self.delta_v[1] = seed[1]
        self.delta_v[2] = seed[2]

        # Add some disturbance according to a gaussian distribution
        x_d = np.random.normal(mu, 10**(np.floor(np.log10(abs(self.delta_v[0]))) - 2.0))
        y_d = np.random.normal(mu, 10**(np.floor(np.log10(abs(self.delta_v[1]))) - 2.0))
        z_d = np.random.normal(mu, 10**(np.floor(np.log10(abs(self.delta_v[2]))) - 2.0))

        # Add disturbances according to linear distribution
        # x_d = (np.random.random() - 0.5) * 10**(np.floor(np.log10(abs(self.delta_v[0]))) - 1.0) * (1 - np.exp(-(self.fitness/1e4)**0.8))
        # y_d = (np.random.random() - 0.5) * 10**(np.floor(np.log10(abs(self.delta_v[1]))) - 1.0) * (1 - np.exp(-(self.fitness/1e4)**0.8))
        # z_d = (np.random.random() - 0.5) * 10**(np.floor(np.log10(abs(self.delta_v[2]))) - 1.0) * (1 - np.exp(-(self.fitness/1e4)**0.8))

        self.delta_v[0] += x_d
        self.delta_v[1] += y_d
        self.delta_v[2] += z_d

    def evaluate_fitness(self, prop, mass, init_state, target_cart, wanted_position, start_date, time):
        # Correct initial state with delta-v
        new_init_state = Cartesian()
        new_init_state.R = init_state.R
        new_init_state.V = init_state.V + self.delta_v

        p = Vector3D(float(new_init_state.R[0])*1e3, float(new_init_state.R[1])*1e3, float(new_init_state.R[2])*1e3)
        v = Vector3D(float(new_init_state.V[0])*1e3, float(new_init_state.V[1])*1e3, float(new_init_state.V[2])*1e3)

        # Initialize propagators
        seconds = float(start_date.second) + float(start_date.microsecond) / 1e6
        orekit_date = AbsoluteDate(start_date.year,
                                   start_date.month,
                                   start_date.day,
                                   start_date.hour,
                                   start_date.minute,
                                   seconds,
                                   TimeScalesFactory.getUTC())

        inertialFrame = FramesFactory.getEME2000()
        initialOrbit = CartesianOrbit(PVCoordinates(p, v), inertialFrame, orekit_date, Cst.WGS84_EARTH_MU)

        newSpacecraftState = SpacecraftState(initialOrbit, mass)

        prop._propagator_num.setInitialState(newSpacecraftState)

        new_state = prop.propagate(start_date + timedelta(seconds=time))
        state_cart = new_state[0]

        state_lvlh = CartesianLVLH()
        state_lvlh.from_cartesian_pair(state_cart, target_cart)

        self.arrival = state_lvlh

        # Evaluate fitness depending on how far we are from the wanted position
        self.fitness = (state_lvlh.R[0]*1e3 - wanted_position[0]*1e3)**2 + \
                       (state_lvlh.R[1]*1e3 - wanted_position[1]*1e3)**2 + \
                       (state_lvlh.R[2]*1e3 - wanted_position[2]*1e3)**2


class Population(object):

    def __init__(self):
        self.pop = []
        self.size = 0
        self.avg_fitness = 0
        self.prop = None
        self.mass = 0

    def set_propagator(self, init_state, start_date=datetime.utcnow()):
        settings_path = '/home/dfrey/cso_ws/src/rdv-cap-sim/simulator/cso_gnc_sim/cfg/chaser.yaml'
        settings_file = file(settings_path, 'r')
        propSettings = yaml.load(settings_file)

        propSettings = propSettings['propagator_settings']
        self.mass = propSettings['orbitProp']['State']['settings']['mass']

        self.prop = OrekitPropagator()
        self.prop.initialize(propSettings, init_state, start_date)

    def generate_population(self, seed, init_state, target_cart, wanted_position, start_date, time, N=1000):
        self.size = N

        for i in xrange(0, self.size):
            ind = Individual()
            ind.create_individual(seed)
            ind.evaluate_fitness(self.prop, self.mass, init_state, target_cart, wanted_position, start_date, time)
            self.pop.append(ind)

        self.avg_fitness = 1.0 / self.size * sum([ind.fitness for ind in self.pop])

    def evolve(self, init_state, target_cart, wanted_position, start_date, time, retain=0.2, random_select=0.05, mutate=0.01):
        # Order population by fitness
        graded = [(ind.fitness, ind) for ind in self.pop]
        graded = [x[1] for x in sorted(graded)]

        # Keep only a percentage of the population to breed
        retain_length = int(len(graded) * retain)
        parents = graded[:retain_length]

        # Randomly add other individuals to promote genetic diversity
        for ind in graded[retain_length:]:
            if random_select > np.random.random():
                parents.append(ind)

        # Mutate some individuals
        for ind in parents:
            if mutate > np.random.random():
                x_d = np.random.normal(0.0, 10**(np.floor(np.log10(abs(ind.delta_v[0]))) - 2.0))
                y_d = np.random.normal(0.0, 10**(np.floor(np.log10(abs(ind.delta_v[1]))) - 2.0))
                z_d = np.random.normal(0.0, 10**(np.floor(np.log10(abs(ind.delta_v[2]))) - 2.0))

                ind.delta_v[0] += x_d
                ind.delta_v[1] += y_d
                ind.delta_v[2] += z_d

        # Cross-over of parents to create children
        parents_length = len(parents)
        children_legth = len(self.pop) - parents_length
        childrens = []
        while len(childrens) < children_legth:
            male = np.random.randint(0, parents_length-1)
            female = np.random.randint(0, parents_length-1)
            if male != female:
                male = parents[male]
                female = parents[female]

                child = Individual()
                delta_v_seed = 0.5 * (male.delta_v + female.delta_v)

                # if male.fitness > female.fitness:
                #     child.delta_v = male.delta_v
                # else:
                #     child.delta_v = female.delta_v

                child.delta_v[0] = delta_v_seed[0]
                child.delta_v[1] = delta_v_seed[1]
                child.delta_v[2] = delta_v_seed[2]

                # child.create_individual(delta_v_seed)
                child.evaluate_fitness(self.prop, self.mass, init_state, target_cart, wanted_position, start_date, time)

                childrens.append(child)

            # parent = parents[0]
            # child = Individual()
            # delta_v_seed = parent.delta_v
            # child.fitness = parent.fitness
            # child.create_individual(delta_v_seed)
            # child.evaluate_fitness(self.prop, self.mass, init_state, target_cart, wanted_position, start_date, time)
            # childrens.append(child)

        parents.extend(childrens)
        self.pop = parents

        self.avg_fitness = 1.0 / self.size * sum([ind.fitness for ind in self.pop])

def linearized_including_J2(target, v_f, N_orb):
    # Initial reference osculatin orbit
    a_0 = target.a
    e_0 = target.e
    i_0 = target.i
    w_0 = target.w
    M_0 = target.m
    v_0 = target.v

    eta_0 = np.sqrt(1.0 - e_0 ** 2)
    p_0 = a_0 * (1.0 - e_0 ** 2)
    r_0 = p_0 / (1.0 + e_0 * np.cos(v_0))

    # Initial reference mean orbit
    target_mean = KepOrbElem()
    target_mean.from_osc_elems(target, 'real-world')

    a_mean = target_mean.a
    i_mean = target_mean.i
    e_mean = target_mean.e

    eta_mean = np.sqrt(1.0 - e_mean ** 2)
    p_mean = a_mean * (1.0 - e_mean ** 2)
    n_mean = np.sqrt(mu_earth / a_mean ** 3)

    # Mean orbital element drift
    a_mean_dot = 0.0
    e_mean_dot = 0.0
    i_mean_dot = 0.0
    O_mean_dot = -1.5 * J_2 * n_mean * (R_earth / p_mean) ** 2 * np.cos(i_mean)
    w_mean_dot = 0.75 * J_2 * n_mean * (R_earth / p_mean) ** 2 * (5.0 * np.cos(i_mean) ** 2 - 1.0)
    M_mean_dot = n_mean + 0.75 * J_2 * n_mean * (R_earth / p_mean) ** 2 * eta_mean * \
                 (3.0 * np.cos(i_mean) ** 2 - 1.0)

    # Epsilon_a partial derivatives: TODO: v_0 or v???
    gamma_2 = -0.5 * J_2 * (R_earth / a_0) ** 2

    depsda = 1.0 - gamma_2 * ((3.0 * np.cos(i_0) ** 2 - 1.0) * ((a_0 / r_0) ** 3 - 1.0 / eta_0 ** 3) +
                              3.0 * (1.0 - np.cos(i_0) ** 2) * (a_0 / r_0) ** 3 * np.cos(2.0 * w_0 + 2.0 * v_0))
    depsde = a_0 * gamma_2 * ((2.0 - 3.0 * np.sin(i_0) ** 2) *
             (3.0 * np.cos(v_0) * (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 6 + 6.0 * e_0 * (1.0 + e_0 * np.cos(v_0)) ** 3 / eta_0 ** 8 - 3.0 * e_0 / eta_0 ** 5) +
             9.0 * np.sin(i_0) ** 2 * np.cos(2.0 * w_0 + 2.0 * v_0) * np.cos(v_0) * (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 6 +
             18.0 * np.sin(i_0) ** 2 * e_0 * np.cos(2.0 * w_0 + 2.0 * v_0) * (1.0 + e_0 * np.cos(v_0)) ** 3 / eta_0 ** 8)
    depsdi = -3.0 * a_0 * gamma_2 * np.sin(2.0 * i_0) * ((a_0 / r_0) ** 3 * (1.0 - np.cos(2.0 * w_0 + 2.0 * v_0)) - 1.0 / eta_0 ** 3)
    depsdw = -6.0 * a_0 * gamma_2 * (1.0 - np.cos(i_0) ** 2) * (a_0 / r_0) ** 3 * np.sin(2.0 * w_0 + 2.0 * v_0)
    depsdv = a_0 * gamma_2 * (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 6 * \
             ((-9.0 * np.cos(i_0) ** 2 + 3.0) * e_0 * np.sin(v_0) -
             (9.0 - 9.0 * np.cos(i_0) ** 2) * np.cos(2.0 * w_0 + 2.0 * v_0) * e_0 * np.sin(v_0) -
             (6.0 - 6.0 * np.cos(i_0) ** 2) * (1.0 + e_0 * np.cos(v_0)) * np.sin(2.0 * w_0 + 2.0 * v_0))

    # Mean elements partial derivatives
    C = J_2 * n_mean * R_earth ** 2 / (4.0 * p_mean ** 2)           # TODO: p or p_mean?
    dOda = 21.0 / a_mean * C * np.cos(i_mean)
    dOde = 24.0 * e_mean / eta_mean ** 2 * C * np.cos(i_mean)
    dOdi = 6.0 * C * np.sin(i_mean)
    dwda = -10.5 * C * (5.0 * np.cos(i_mean) ** 2 - 1.0) / a_mean
    dwde = 12.0 * e_mean * C * (5.0 * np.cos(i_mean) ** 2 - 1.0) / eta_mean ** 2
    dwdi = -15.0 * C * np.sin(2.0 * i_mean)
    dMda = -3.0 * n_mean / (2.0 * a_mean) - eta_mean / (2.0 * a_mean) * C * (63.0 * np.cos(i_mean)**2 - 21.0)
    dMde = 9.0 * e_mean * C * (3.0 * np.cos(i_mean) ** 2 - 1.0) / eta_mean
    dMdi = -9.0 * eta_mean * C * np.sin(2.0 * i_mean)

    # Estimate flight time
    # N_orb = ...
    E = lambda v: 2.0 * np.arctan(np.sqrt((1.0 - e_0) / (1.0 + e_0)) * np.tan(v / 2.0))
    M = lambda v: (E(v) - e_0 * np.sin(E(v))) % (2.0 * np.pi)

    tau = lambda v: (2.0 * np.pi * N_orb + M(v) - M_0) / M_mean_dot

    # Position
    r = lambda v: p_0 / (1.0 + e_0 * np.cos(v))

    # Position and true anomaly derivatives         # TODO: CHECK IF divided by eta_0 or eta?
    r_dot = lambda v: a_0 * e_0 * np.sin(v) / eta_0 * M_mean_dot
    v_dot = lambda v: (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3 * M_mean_dot

    # Phi_1
    k_x_dot = lambda v: a_0 * e_0 * v_dot(v) * np.cos(v) / eta_0
    phi_11 = lambda v: r_dot(v) / a_0 + (k_x_dot(v) * tau(v) + a_0 * e_0 * np.sin(v) / eta_0) * dMda
    phi_12 = lambda v: a_0 * v_dot(v) * np.sin(v) + (k_x_dot(v) * tau(v) + a_0 * e_0 * np.sin(v) / eta_0) * \
                       (dMde + dMda * depsde + dMda * depsdv * np.sin(v_0) / eta_0 ** 2 * (2.0 + e_0 * np.cos(e_0)))
    phi_13 = lambda v: (k_x_dot(v) * tau(v) + a_0 * e_0 * np.sin(v) / eta_0) * (dMda * depsdi + dMdi)
    phi_14 = 0.0
    phi_15 = lambda v: (k_x_dot(v) * tau(v) + a_0 * e_0 * np.sin(v) / eta_0) * dMda * depsdw
    phi_16 = lambda v: k_x_dot(v) + (k_x_dot(v) * tau(v) + a_0 * e_0 * np.sin(v) / eta_0) * dMda * depsdv * \
                       (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 3

    # Phi 2
    k_y_dot = lambda v: r_dot(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3 - 2.0 * e_0 * v_dot(v) * np.sin(v) * (1.0 + e_0 * np.cos(v)) / eta_0 ** 3
    phi_21 = lambda v: (r_dot(v) * np.cos(i_0) * tau(v) + r(v) * np.cos(i_0)) * dOda + (r_dot(v) * tau(v) + r(v)) * dwda + \
                       (k_y_dot(v) * tau(v) + r(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3) * dMda
    phi_22 = lambda v: 1.0 / eta_0 ** 2 * (r(v) * v_dot(v) * np.cos(v) * (2.0 + e_0 * np.cos(v)) - r(v) * e_0 * v_dot(v) * np.sin(v) ** 2 +
                       r_dot(v) * np.sin(v) * (2.0 + e_0 * np.cos(v))) + (r_dot(v) * np.cos(i_0) * tau(v) + r(v) * np.cos(i_0)) * \
                       (dOda * depsde + dOda * depsdv * np.sin(v_0) / eta_0 ** 2 * (2.0 + e_0 * np.cos(e_0)) + dOde) + \
                       (r_dot(v) * tau(v) + r(v)) * (dwda * depsde + dwda * depsdv * np.sin(v_0) / eta_0 ** 2 * (2.0 + e_0 * np.cos(e_0)) + dwde) + \
                       (k_y_dot(v) * tau(v) + r(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3) * \
                       (dMda * depsde + dMda * depsdv * np.sin(v_0) * (2.0 + e_0 * np.cos(e_0)) / eta_0 ** 2 + dMde)
    phi_23 = lambda v: (r_dot(v) * np.cos(i_0) * tau(v) + r(v) * np.cos(i_0)) * (dOda * depsdi + dOdi) + \
                       (r_dot(v) * tau(v) + r(v)) * (dwda * depsdi + dwdi) + \
                       (k_y_dot(v) * tau(v) + r(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3) * (dMda * depsdi + dMdi)
    phi_24 = lambda v: r_dot(v) * np.cos(i_0)
    phi_25 = lambda v: r_dot(v) + (r_dot(v) * np.cos(i_0) * tau(v) + r(v) * np.cos(i_0)) * dOda * depsdw + \
                       (r_dot(v) * tau(v) + r(v)) * dwda * depsdw + (k_y_dot(v) * tau(v) + r(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3) * dMda * depsdw
    phi_26 = lambda v: k_y_dot(v) + (r_dot(v) * np.cos(i_0) * tau(v) + r(v) * np.cos(i_0)) * dOda * depsdv * (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 3 + \
                       (r_dot(v) * tau(v) + r(v)) * dwda * depsdv * (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 3 + \
                       (k_y_dot(v) * tau(v) + r(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3) * dOda * depsdv * \
                       (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 3

    # Phi 3
    k_z_dot = lambda v: -r_dot(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0) + \
                        r(v) * np.sin(v + w_0 + w_mean_dot * tau(v)) * (v_dot(v) + w_mean_dot) * np.sin(i_0)
    phi_31 = lambda v: (k_z_dot(v) * tau(v) - r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0)) * dOda
    phi_32 = lambda v: (k_z_dot(v) * tau(v) - r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0)) * \
                       (dOda * depsde + dOda * depsdv * np.sin(v_0) / eta_0 ** 2 * (2.0 + e_0 * np.cos(e_0)) + dOde)
    phi_33 = lambda v: r_dot(v) * np.sin(v + w_0 + w_mean_dot * tau(v)) + r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * \
                       (v_dot(v) + w_mean_dot) + (k_z_dot(v) * tau(v) - r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0)) * dOda
    phi_34 = lambda v: k_z_dot(v)
    phi_35 = lambda v: (k_z_dot(v) * tau(v) - r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0)) * dOda * depsdw
    phi_36 = lambda v: (k_z_dot(v) * tau(v) - r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0)) * dOda * depsdv * \
                       (1.0 + e_0 * np.cos(e_0)) ** 2 / eta_0 ** 3

    # Phi 4
    phi_41 = lambda v: r(v) / a_0 + a_0 * e_0 * np.sin(v) / eta_0 * dMda * tau(v)
    phi_42 = lambda v: a_0 * e_0 * np.sin(v) / eta_0 * (dMda * depsde + dMda * depsdv * np.sin(v_0) / eta_0 ** 2 * (2.0 + e_0 * np.cos(e_0)) +
                       dMde) * tau(v) - a_0 * np.cos(v)
    phi_43 = lambda v: a_0 * e_0 * np.sin(v) / eta_0 * (dMda * depsdi + dMdi) * tau(v)
    phi_44 = 0.0
    phi_45 = lambda v: a_0 * e_0 * np.sin(v) / eta_0 * dMda * depsdw * tau(v)
    phi_46 = lambda v: a_0 * e_0 * np.sin(v) / eta_0 + a_0 * e_0 * np.sin(v) / eta_0 * dMda * depsdw * \
                       (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 3 * tau(v)

    # Phi 5
    phi_51 = lambda v: r(v) * np.cos(i_0) * dOda * tau(v) + r(v) * dwda * tau(v) + r(v) * (1.0 + e_0 * np.cos(v)) ** 2 \
                       / eta_0 ** 3 * dMda * tau(v)
    phi_52 = lambda v: r(v) * np.sin(v) / eta_0 ** 2 * (2.0 + e_0 * np.cos(v)) + r(v) * np.cos(i_0) * \
                       (dOda * depsde + dOda * depsdv * np.sin(v_0) / eta_0 ** 2 * (2.0 + e_0 * np.cos(e_0)) + dOde) * tau(v) + \
                       r(v) * (dwda * depsde + dwda * depsdv * np.sin(v_0) / eta_0 ** 2 * (2.0 + e_0 * np.cos(e_0)) + dwde) * tau(v) + \
                       r(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3 * \
                       (dMda * depsde + dMda * depsdv * np.sin(v_0) / eta_0 ** 2 * (2.0 + e_0 * np.cos(e_0)) + dMde) * tau(v)
    phi_53 = lambda v: r(v) * np.cos(i_0) * (dOda * depsdi + dOdi) * tau(v) + r(v) * (dwda * depsdi + dwdi) * tau(v) + \
                       r(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3 * (dMda * depsdi + dMdi) * tau(v)
    phi_54 = lambda v: r(v) * np.cos(i_0)
    phi_55 = lambda v: r(v) + r(v) * np.cos(i_0) * dOda * depsdw * tau(v) + r(v) * dwda * depsdw * tau(v) + \
                       r(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3 * dMda * depsdw * tau(v)
    phi_56 = lambda v: r(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3 + r(v) * np.cos(i_0) * dOda * depsdv * \
                       (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 3 * tau(v) + r(v) * dwda * depsdv * (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 3 * tau(v) + \
                       r(v) * (1.0 + e_0 * np.cos(v)) ** 2 / eta_0 ** 3 * dMda * depsdv * (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 3 * tau(v)

    # Phi 6
    phi_61 = lambda v: -r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0) * dOda * tau(v)
    phi_62 = lambda v: -r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0) * (dOda * depsde + dOda * depsdv * np.sin(v_0) /
                        eta_0 ** 2 * (2.0 + e_0 * np.cos(e_0)) + dOde) * tau(v)
    phi_63 = lambda v: r(v) * np.sin(v + w_0 + w_mean_dot * tau(v)) - r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0) * (dOda * depsdi + dOdi) * tau(v)
    phi_64 = lambda v: -r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0)
    phi_65 = lambda v: -r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0) * dOda * depsdw * tau(v)
    phi_66 = lambda v: -r(v) * np.cos(v + w_0 + w_mean_dot * tau(v)) * np.sin(i_0) * dOda * depsdv * (1.0 + e_0 * np.cos(v_0)) ** 2 / eta_0 ** 3 * tau(v)

    phi_ = np.array([
        [phi_11(v_f), phi_12(v_f), phi_13(v_f), phi_14, phi_15(v_f), phi_16(v_f)],
        [phi_21(v_f), phi_22(v_f), phi_23(v_f), phi_24(v_f), phi_25(v_f), phi_26(v_f)],
        [phi_31(v_f), phi_32(v_f), phi_33(v_f), phi_34(v_f), phi_35(v_f), phi_36(v_f)],
        [phi_41(v_f), phi_42(v_f), phi_43(v_f), phi_44, phi_45(v_f), phi_46(v_f)],
        [phi_51(v_f), phi_52(v_f), phi_53(v_f), phi_54(v_f), phi_55(v_f), phi_56(v_f)],
        [phi_61(v_f), phi_62(v_f), phi_63(v_f), phi_64(v_f), phi_65(v_f), phi_66(v_f)],
    ])

    return phi_, tau(v_f)

def find_v(time, a_mean, e_mean, i_mean, M_0, e_0):

    n_mean = np.sqrt(mu_earth / a_mean**3)
    p_mean = a_mean * (1.0 - e_mean**2)
    eta_mean = np.sqrt(1.0 - e_mean ** 2)

    T = 2.0 * np.pi / n_mean

    M_mean_dot = n_mean + 0.75 * J_2 * n_mean * (R_earth / p_mean) ** 2 * eta_mean * \
                 (3.0 * np.cos(i_mean) ** 2 - 1.0)

    N_orb = np.floor(time / T)

    M_v = (time * M_mean_dot - 2.0 * np.pi * N_orb + M_0)
    E_v = calc_E_from_m(M_v, e_0)
    v = calc_v_from_E(E_v, e_0)

    return v, N_orb

def calc_E_from_m(m, e):
    if m < np.pi:
        E = m + e / 2.0
    else:
        E = m - e / 2.0

    max_int = 20  # maximum number of iterations

    while max_int > 1:
        fE = E - e * np.sin(E) - m
        fpE = 1.0 - e * np.cos(E)
        ratio = fE / fpE
        max_int = max_int - 1

        # check if ratio is small enough
        if abs(ratio) > 1e-15:
            E = E - ratio
        else:
            break

    if E < 0:
        E = E + np.pi * 2.0

    return E

def calc_v_from_E(E, e):
    v = 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(E / 2.0),
                               np.sqrt(1.0 - e) * np.cos(E / 2.0))

    if v < 0:
        v = v + np.pi * 2.0

    return v

def main():
    # Define start date
    start_date = datetime.utcnow()

    chaser_osc = KepOrbElem()
    target_osc = KepOrbElem()
    target_mean = KepOrbElem()

    chaser_cart = Cartesian()
    target_cart = Cartesian()

    # Define target and chaser initial positions
    target_osc.a = 7075.384
    target_osc.e = 0.0003721779
    target_osc.i = 1.727
    target_osc.O = 0.74233
    target_osc.w = 1.628
    target_osc.v = 4.67845

    target_cart.from_keporb(target_osc)
    target_mean.from_osc_elems(target_osc, 'real-world')

    chaser_lvlh = CartesianLVLH()
    chaser_lvlh.R = np.array([-3.84647216, 7.99996761, 0.03330542])
    chaser_lvlh.V = np.array([1.67308709e-4, 6.11002933e-3, 4.20827334e-5])
    chaser_cart.from_lvlh_frame(target_cart, chaser_lvlh)
    chaser_osc.from_cartesian(chaser_cart)

    # Choose a time for the transfer
    t = 3000.0

    # Evaluate anomaly after t seconds
    vn = find_v(t, target_mean.a, target_mean.e, target_mean.i, target_osc.m, target_osc.e)

    st_0 = linearized_including_J2(target_osc, target_osc.v, 0.0)
    phi_0 = st_0[0]

    st = linearized_including_J2(target_osc, vn[0], 1.0)
    phi = st[0]

    phi_comb = np.array([
        phi_0[0:6][3],
        phi_0[0:6][4],
        phi_0[0:6][5],
        phi[0:6][3],
        phi[0:6][4],
        phi[0:6][5]
    ])

    state_comb = np.array([chaser_lvlh.R[0], chaser_lvlh.R[1], chaser_lvlh.R[2], 0.0, 18.0, 0.0])

    # Wanted initial relative orbital elements
    de0_wanted = np.linalg.inv(phi_comb).dot(state_comb)

    # Initial difference in osculating orbital elements
    de0_initial = np.array([
        chaser_osc.a - target_osc.a,
        chaser_osc.e - target_osc.e,
        chaser_osc.i - target_osc.i,
        chaser_osc.O - target_osc.O,
        chaser_osc.w - target_osc.w,
        chaser_osc.m - target_osc.m,
    ])

    de0_diff = de0_wanted - de0_initial

    de_chaser_wanted = np.array([
        chaser_osc.a + de0_diff[0],
        chaser_osc.e + de0_diff[1],
        chaser_osc.i + de0_diff[2],
        chaser_osc.O + de0_diff[3],
        chaser_osc.w + de0_diff[4],
        chaser_osc.m + de0_diff[5],
    ])

    chaser_kep_wanted = KepOrbElem()
    chaser_kep_wanted.a = de_chaser_wanted[0]
    chaser_kep_wanted.e = de_chaser_wanted[1]
    chaser_kep_wanted.i = de_chaser_wanted[2]
    chaser_kep_wanted.O = de_chaser_wanted[3]
    chaser_kep_wanted.w = de_chaser_wanted[4]
    chaser_kep_wanted.m = de_chaser_wanted[5]

    R_chaser_initial = chaser_cart.R
    V_chaser_initial = chaser_cart.V

    chaser_cart.from_keporb(chaser_kep_wanted)

    R_chaser_initial_wanted = chaser_cart.R
    V_chaser_initial_wanted = chaser_cart.V

    # Evaluate delta-v seed
    delta_v_seed = V_chaser_initial_wanted - V_chaser_initial

    # Re-set chaser_cart to initial condition
    chaser_cart.R = R_chaser_initial
    chaser_cart.V = V_chaser_initial

    # Target propagator
    settings_path = '/home/dfrey/cso_ws/src/rdv-cap-sim/simulator/cso_gnc_sim/cfg/target.yaml'
    settings_file = file(settings_path, 'r')
    propSettings = yaml.load(settings_file)
    propSettings = propSettings['propagator_settings']

    OrekitPropagator.init_jvm()
    prop_target = OrekitPropagator()
    prop_target.initialize(propSettings, target_osc, start_date)

    target_prop = prop_target.propagate(start_date + timedelta(seconds=t))

    delta_v_seed = np.array([-5.62461607e-04, -9.63933081e-05, 1.07564374e-03])

    # Create population
    pop = Population()
    pop.set_propagator(chaser_kep_wanted, start_date)
    pop.generate_population(delta_v_seed, chaser_cart, target_prop[0], state_comb[3:6], start_date, t)

    gen = 200
    i = 0
    while i < gen:

        pop.evolve(chaser_cart, target_prop[0], state_comb[3:6], start_date, t)
        print "\nGen nr. " + str(i) + ":  " + str(pop.avg_fitness)
        print " >>> Best:    " + str(pop.pop[0].fitness)
        print " >>> Deltav:  " + str(pop.pop[0].delta_v)
        print " >>> Arrival: " + str(pop.pop[0].arrival.R)
        i += 1

    print pop.pop[0].delta_v

if __name__ == "__main__":
    main()

