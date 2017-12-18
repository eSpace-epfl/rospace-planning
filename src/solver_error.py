import pykep as pk
import random as rd
from numpy import sin, cos, pi, sqrt, array, linalg
from space_tf import Cartesian, CartesianLVLH, KepOrbElem
from solver_v2 import Solver
from scenario_v2 import Position
from Constants import mu_earth
import os

# Generate some random possible relative positions with a certain increasing distance from the target

chaser = Position()
target = Position()

solver = Solver()

distances = [1e-2, 1e-1, 5e-1, 1, 2, 4, 10, 50]

for i in xrange(0, len(distances)):
    for j in xrange(0, 100):
        target.kep.generate_random()
        chaser.cartesian.from_keporb(target.kep)
        alpha = 2.0*pi*rd.random()
        beta = 2.0*pi*rd.random()
        rel_pos = [distances[i] * cos(alpha) * cos(beta), distances[i] * cos(alpha) * sin(beta), distances[i] * sin(alpha)]
        chaser.lvlh.R = rel_pos

        chaser.cartesian.from_lvlh_frame(target.cartesian, chaser.lvlh)

        # Solve CW
        solver.clohessy_wiltshire_solver(chaser, target, target)

        dt = solver.test_error['CW-sol']['dt']

        # Solve lambert
        r_T_t1, v_T_t1 = pk.propagate_lagrangian(target.cartesian.R, target.cartesian.V, dt, mu_earth)
        T = 2*pi*sqrt(target.kep.a**3/mu_earth)
        Nmax = int(dt / T) + 1
        sol = pk.lambert_problem(chaser.cartesian.R, array(r_T_t1), dt, mu_earth, multi_revs=Nmax)

        best_deltaV = 1e12
        best_deltaV_1 = 0
        best_deltaV_2 = 0
        for k in xrange(0, len(sol.get_v1())):
            deltaV_1 = array(sol.get_v1()[k]) - chaser.cartesian.V
            deltaV_2 = array(sol.get_v2()[k]) - target.cartesian.V

            deltaV_tot = linalg.norm(deltaV_1) + linalg.norm(deltaV_2)

            if deltaV_tot < best_deltaV:
                best_deltaV = deltaV_tot
                best_deltaV_1 = deltaV_1
                best_deltaV_2 = deltaV_2

        if os.path.isdir('/home/dfrey/polybox/manoeuvre'):

            # Simulating the whole manoeuvre and store the result
            chaser_tmp = Position()
            target_tmp = Position()

            chaser_tmp.from_other_position(chaser)
            target_tmp.from_other_position(target)

            # Creating list of radius of target and chaser
            R_target = [target_tmp.cartesian.R]
            R_chaser = [chaser_tmp.cartesian.R]
            R_chaser_lvlh = [chaser_tmp.lvlh.R]
            R_chaser_lvc =  [array([chaser_tmp.lvc.dR, chaser_tmp.lvc.dV, chaser_tmp.lvc.dH])]


            for k in xrange(0, dt):
                solver._propagator(chaser_tmp, target_tmp, 1)
                R_chaser.append(chaser_tmp.cartesian.R)
                R_target.append(target_tmp.cartesian.R)
                R_chaser_lvlh.append(chaser_tmp.lvlh.R)
                R_chaser_lvc.append(array([chaser_tmp.lvc.dR, chaser_tmp.lvc.dV, chaser_tmp.lvc.dH]))


            for i in xrange(0, len(solver.command_line)):
                cmd = solver.command_line[i]

                # Apply dV
                chaser_tmp.cartesian.V += cmd.deltaV_C

                for j in xrange(0, cmd.duration):
                    solver._propagator(chaser_tmp, target_tmp, 1)
                    R_chaser.append(chaser_tmp.cartesian.R)
                    R_target.append(target_tmp.cartesian.R)
                    R_chaser_lvlh.append(chaser_tmp.lvlh.R)
                    R_chaser_lvc.append(solver.array([chaser_tmp.lvc.dR, chaser_tmp.lvc.dV, chaser_tmp.lvc.dH]))

                print "Relative Position after command " + str(i) + ":    " + str(chaser_tmp.lvlh.R)

            # Saving in .mat file
            sio.savemat('/home/dfrey/polybox/manoeuvre/full_manoeuvre.mat',
                        mdict={'abs_pos_c': R_chaser, 'rel_pos_c': R_chaser_lvlh, 'abs_pos_t': R_target,
                               'lvc_c': R_chaser_lvc})

            print "Manoeuvre saved."


