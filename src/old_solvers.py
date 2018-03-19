def clohessy_wiltshire(self, checkpoint):
    """
        Solve Hill's Equation to get the amount of DeltaV needed to go to the next checkpoint.

    References:
        David A. Vallado, Fundamentals of Astrodynamics and Applications, Second Edition, Algorithm 47 (p. 382)

    Args:
        chaser (Chaser): Chaser state.
        checkpoint (RelativeCP): Next checkpoint.
        target (Satellite): Target state.
    """

    # TODO: Apply correction according to the new definition of objects
    # TODO: Consider thruster accuracy

    print ">>>> Solving CW-equations\n"

    a = target.abs_state.a
    max_time = int(2 * np.pi * np.sqrt(a ** 3 / mu_earth))

    r_rel_c_0 = chaser.rel_state.R
    v_rel_c_0 = chaser.rel_state.V

    r_rel_c_n = checkpoint.rel_state.R
    v_rel_c_n = [0, 0, 0]

    n = np.sqrt(mu_earth / a ** 3.0)

    phi_rr = lambda t: np.array([
        [4.0 - 3.0 * np.cos(n * t), 0.0, 0.0],
        [6.0 * (np.sin(n * t) - n * t), 1.0, 0.0],
        [0.0, 0.0, np.cos(n * t)]
    ])

    phi_rv = lambda t: np.array([
        [1.0 / n * np.sin(n * t), 2.0 / n * (1 - np.cos(n * t)), 0.0],
        [2.0 / n * (np.cos(n * t) - 1.0), 1.0 / n * (4.0 * np.sin(n * t) - 3.0 * n * t), 0.0],
        [0.0, 0.0, 1.0 / n * np.sin(n * t)]
    ])

    phi_vr = lambda t: np.array([
        [3.0 * n * np.sin(n * t), 0.0, 0.0],
        [6.0 * n * (np.cos(n * t) - 1), 0.0, 0.0],
        [0.0, 0.0, -n * np.sin(n * t)]
    ])

    phi_vv = lambda t: np.array([
        [np.cos(n * t), 2.0 * np.sin(n * t), 0.0],
        [-2.0 * np.sin(n * t), 4.0 * np.cos(n * t) - 3.0, 0.0],
        [0.0, 0.0, np.cos(n * t)]
    ])

    best_deltaV = 1e12
    delta_T = 0

    for t_ in xrange(1, max_time):
        rv_t = phi_rv(t_)
        deltaV_1 = np.linalg.inv(rv_t).dot(r_rel_c_n - np.dot(phi_rr(t_), r_rel_c_0)) - v_rel_c_0
        deltaV_2 = np.dot(phi_vr(t_), r_rel_c_0) + np.dot(phi_vv(t_), v_rel_c_0 + deltaV_1) - v_rel_c_n

        deltaV_tot = np.linalg.norm(deltaV_1) + np.linalg.norm(deltaV_2)

        if best_deltaV > deltaV_tot:
            # Check if the keep out zone is invaded and if we are not approaching it
            # if id != 1:
            #     for t_test in xrange(0, t_ + 1):
            #         r_test = np.dot(phi_rr(t_test), r_rel_c_0) + np.dot(phi_rv(t_test), v_rel_c_0 + deltaV_1)
            #         if all(abs(r_test[i]) >= ko_zone for i in range(0, 3)):
            #             best_deltaV = deltaV_tot
            #             best_deltaV_1 = deltaV_1
            #             best_deltaV_2 = deltaV_2
            #             delta_T = t_
            best_deltaV = deltaV_tot
            best_deltaV_1 = deltaV_1
            best_deltaV_2 = deltaV_2
            delta_T = t_

    target_cart = Cartesian()
    target_cart.from_keporb(target.abs_state)

    # Change frame of reference of deltaV. From LVLH to Earth-Inertial
    B = target_cart.get_lof()
    deltaV_C_1 = np.linalg.inv(B).dot(best_deltaV_1)

    # Create command
    c1 = RelativeMan()
    c1.dV = deltaV_C_1
    c1.set_abs_state(chaser.abs_state)
    c1.set_rel_state(chaser.rel_state)
    c1.duration = 0
    c1.description = 'CW approach'

    # Propagate chaser and target
    self._propagator(chaser, target, 1e-5, deltaV_C_1)

    self._propagator(chaser, target, delta_T)

    self.print_state(chaser, target)

    self.manoeuvre_plan.append(c1)

    target_cart.from_keporb(target.abs_state)

    R = target_cart.get_lof()
    deltaV_C_2 = np.linalg.inv(R).dot(best_deltaV_2)

    # Create command
    c2 = RelativeMan()
    c2.dV = deltaV_C_2
    c2.set_abs_state(chaser.abs_state)
    c2.set_rel_state(chaser.rel_state)
    c2.duration = delta_T
    c2.description = 'CW approach'

    # Propagate chaser and target to evaluate all the future commands properly
    self._propagator(chaser, target, 1e-5, deltaV_C_2)

    self.print_state(chaser, target)

    self.manoeuvre_plan.append(c2)

