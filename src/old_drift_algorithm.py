def drift_to_new(self, checkpoint):
    """
        Algorithm that tries to drift to the next checkpoint, staying within a certain error ellipsoid.

    Args:
        checkpoint (AbsoluteCP or RelativeCP): Next checkpoint.

    Return:
        t_est (float64): Drifting time to reach the next checkpoint (in seconds). If not reachable, return None.
    """
    # Creating mean orbital elements
    chaser_mean = KepOrbElem()
    target_mean = KepOrbElem()

    # Creating cartesian coordinates
    chaser_cart = Cartesian()
    target_cart = Cartesian()

    # Creating old chaser and target objects to store their temporary value
    chaser_old = Chaser()
    target_old = Satellite()

    # Define a function F for the angle calculation
    F = lambda dv_req, dv, n: int((dv - dv_req) / n > 0.0) * np.sign(n)

    # Correct altitude at every loop until drifting is possible
    while 1:
        # Assign mean values from osculating
        chaser_mean.from_osc_elems(self.chaser.abs_state)
        target_mean.from_osc_elems(self.target.abs_state)

        # Assign cartesian coordinates from mean-orbital (mean orbital radius needed)
        chaser_cart.from_keporb(chaser_mean)
        target_cart.from_keporb(target_mean)

        # Assign information to the new chaser and target objects
        chaser_old.set_from_other_satellite(self.chaser)
        target_old.set_from_other_satellite(self.target)

        # Evaluate relative mean angular velocity. If it's below zero chaser moves slower than target,
        # otherwise faster
        n_c = np.sqrt(mu_earth / chaser_mean.a ** 3)
        n_t = np.sqrt(mu_earth / target_mean.a ** 3)
        n_rel = n_c - n_t

        # Required true anomaly difference at the end of the manoeuvre, estimation assuming circular
        # orbit
        r_C = np.linalg.norm(chaser_cart.R)
        dv_req = checkpoint.rel_state.R[1] / r_C

        # Evaluate the actual true anomaly difference
        actual_dv = (chaser_mean.v + chaser_mean.w) % (2.0 * np.pi) - (target_mean.v + target_mean.w) % (
                2.0 * np.pi)

        # Millisecond tolerance to exit the loop
        tol = 1e-3

        t_est = 10 ** np.floor(np.log10((2.0 * np.pi * F(dv_req, actual_dv, n_rel) + dv_req - actual_dv) / n_rel))
        t_est_old = 0.0
        t_old = 0.0
        ellipsoid_flag = False
        dt = t_est
        dr_next_old = 0.0
        while abs(t_est - t_old) > tol:
            chaser_prop = self.scenario.prop_chaser.propagate(self.epoch + timedelta(seconds=t_est))
            target_prop = self.scenario.prop_target.propagate(self.epoch + timedelta(seconds=t_est))

            chaser_cart = chaser_prop[0]
            target_cart = target_prop[0]

            self.chaser.abs_state.from_cartesian(chaser_cart)
            self.target.abs_state.from_cartesian(target_cart)
            self.chaser.rel_state.from_cartesian_pair(chaser_cart, target_cart)

            # Correct plane in the middle of the drifting
            tol_i = 1.0 / self.chaser.abs_state.a
            tol_O = 1.0 / self.chaser.abs_state.a

            # At this point, inclination and raan should match the one of the target
            di = target_mean.i - chaser_mean.i
            dO = target_mean.O - chaser_mean.O
            if abs(di) > tol_i or abs(dO) > tol_O:
                checkpoint_abs = AbsoluteCP()
                checkpoint_abs.abs_state.i = target_mean.i
                checkpoint_abs.abs_state.O = target_mean.O
                self.plane_correction(checkpoint_abs)

            dr_next = self.chaser.rel_state.R[1] - checkpoint.rel_state.R[1]

            t_old = t_est

            if dr_next <= 0.0 and dr_next_old <= 0.0:
                t_est_old = t_est
                t_est += dt
            elif dr_next >= 0.0 and dr_next_old >= 0.0:
                t_est_old = t_est
                t_est -= dt
            elif (dr_next <= 0.0 and dr_next_old >= 0.0) or (dr_next >= 0.0 and dr_next_old <= 0.0):
                t_est = (t_est_old + t_est) / 2.0
                t_est_old = t_old
                dt /= 10.0

            dr_next_old = dr_next

            if abs(checkpoint.rel_state.R[1] - self.chaser.rel_state.R[1]) <= checkpoint.error_ellipsoid[1]:
                # Almost in line with the checkpoint
                if abs(checkpoint.rel_state.R[0] - self.chaser.rel_state.R[0]) <= checkpoint.error_ellipsoid[0]:
                    # Inside the tolerance, the point may be reached by drifting
                    ellipsoid_flag = True
                else:
                    # Outside tolerance, point may not be reached!
                    break

            self.chaser.set_from_other_satellite(chaser_old)
            self.target.set_from_other_satellite(target_old)

        if ellipsoid_flag:
            # It is possible to drift in t_est
            return t_est
        else:
            # Drift is not possible, drop a warning and correct altitude!
            print "\n[WARNING]: Drifting to checkpoint nr. " + str(checkpoint.id) + " not possible!"
            print "           Correcting altitude automatically...\n"

            # Create new checkpoint
            checkpoint_new_abs = AbsoluteCP()
            checkpoint_new_abs.set_abs_state(chaser_mean)
            checkpoint_new_abs.abs_state.a = target_mean.a + checkpoint.rel_state.R[0]
            checkpoint_new_abs.abs_state.e = target_mean.a * target_mean.e / checkpoint_new_abs.abs_state.a

            self.adjust_eccentricity_semimajoraxis(checkpoint_new_abs)

def drift_to(self, checkpoint):
    """
        Algorithm that tries to drift to the next checkpoint, staying within a certain error ellipsoid.

    Args:
        chaser (Chaser): Chaser state.
        checkpoint (AbsoluteCP or RelativeCP): Next checkpoint.
        target (Satellite): Target state.

    Return:
        t_est (float64): Drifting time to reach the next checkpoint (in seconds). If not reachable, return None.
    """
    # Calculate the mean altitude difference
    chaser_mean = KepOrbElem()
    chaser_mean.from_osc_elems(self.chaser.abs_state)

    target_mean = KepOrbElem()
    target_mean.from_osc_elems(self.target.abs_state)

    chaser_cart = Cartesian()
    target_cart = Cartesian()

    chaser_cart.from_keporb(chaser_mean)
    target_cart.from_keporb(target_mean)

    r_C = np.linalg.norm(chaser_cart.R)
    r_T = np.linalg.norm(target_cart.R)

    # Assuming we are on a coelliptic orbit, check if the distance allows us to drift or if we are not really
    # close to the target's orbit
    # if abs(checkpoint.rel_state.R[0] - r_C + r_T) >= checkpoint.error_ellipsoid[0] or \
    #                 abs(self.chaser.abs_state.a - self.target.abs_state.a) < 2e-1:
    #     return None

    chaser_old = Chaser()
    chaser_old.set_from_other_satellite(self.chaser)
    target_old = Satellite()
    target_old.set_from_other_satellite(self.target)

    n_c = np.sqrt(mu_earth / chaser_mean.a ** 3)
    n_t = np.sqrt(mu_earth / target_mean.a ** 3)

    # If n_rel is below zero, we are moving slower than target. Otherwise faster.
    n_rel = n_c - n_t

    # Required dv at the end of the manoeuvre, estimation based on the relative position
    dv_req = checkpoint.rel_state.R[1] / r_C

    # Check if a drift to the wanted position is possible, if yes check if it can be done under a certain time,
    # if not try to resync
    actual_dv = (chaser_mean.v + chaser_mean.w) % (2.0*np.pi) - (target_mean.v + target_mean.w) % (2.0*np.pi)

    # Define a function F for the angle calculation
    F = lambda dv_req, dv, n: int((dv - dv_req) / n > 0.0) * np.sign(n)

    t_est = (2.0 * np.pi * F(dv_req, actual_dv, n_rel) + dv_req - actual_dv) / n_rel
    t_est_old = 0.0
    t_old = 0.0
    ellipsoid_flag = False
    tol = 1e-3         # Millisecond tolerance
    dt = 10000.0
    dr_next_old = 0.0
    while abs(t_est - t_old) > tol:
        chaser_prop = self.scenario.prop_chaser.propagate(self.epoch + timedelta(seconds=t_est))
        target_prop = self.scenario.prop_target.propagate(self.epoch + timedelta(seconds=t_est))

        chaser_cart = chaser_prop[0]
        target_cart = target_prop[0]

        chaser_old.abs_state.from_cartesian(chaser_cart)
        target_old.abs_state.from_cartesian(target_cart)
        chaser_old.rel_state.from_cartesian_pair(chaser_cart, target_cart)

        dr_next = chaser_old.rel_state.R[1] - checkpoint.rel_state.R[1]

        t_old = t_est

        if dr_next <= 0.0 and dr_next_old <= 0.0:
            t_est_old = t_est
            t_est += dt
        elif dr_next >= 0.0 and dr_next_old >= 0.0:
            t_est_old = t_est
            t_est -= dt
        elif (dr_next <= 0.0 and dr_next_old >= 0.0) or (dr_next >= 0.0 and dr_next_old <= 0.0):
            t_est = (t_est_old + t_est) / 2.0
            t_est_old = t_old
            dt /= 10.0

        dr_next_old = dr_next

        # Assuming to stay on the same plane
        if abs(checkpoint.rel_state.R[0] - chaser_old.rel_state.R[0]) <= checkpoint.error_ellipsoid[0] and \
                abs(checkpoint.rel_state.R[1] - chaser_old.rel_state.R[1]) <= checkpoint.error_ellipsoid[1]:
            # We have reached the error ellipsoid, can break
            ellipsoid_flag = True

        chaser_old.set_from_other_satellite(self.chaser)
        target_old.set_from_other_satellite(self.target)

    if ellipsoid_flag:
        # With the estimated time, we are in the error-ellipsoid
        return t_est
    else:
        return None
