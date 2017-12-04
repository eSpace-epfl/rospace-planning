import numpy as np


class TrajController:

    def __init__(self):
        self.command_line = []
        # TODO: Define a command, independant from epoch
        # A command should contain:
        # --> When in terms of true anomaly the command has to be executed
        # --> Thrust vector in different bases
        # --> Thrust intensity

    def adjust_inclination(self, chaser, chaser_next):
        """
            Given the chaser relative orbital elements w.r.t target,
            ad its next wanted status correct the inclination.

        Args:
              chaser (Position):
              chaser_next (Position):
        """

        # Position at which the burn should occur
        theta = [2*np.pi - chaser.kep.w, (3*np.pi - chaser.kep.w) % (2*np.pi)]

        # Evaluate the inclination difference to correct
        di = chaser.rel_kep.dIx - chaser_next.rel_kep.dIx

        # Calculate burn intensity
        # TODO: The velocity has to be propagated to the point where the burn occurs
        deltav = np.norm(chaser.cartesian.v) * np.sqrt(2*(1 - np.cos(di)))

        # Evaluate burn direction in chaser frame of reference
        deltav_C = deltav * np.array([np.cos(np.pi/2 + di/2), 0, np.sin(np.pi/2 + di/2)])

        # Add to the command line the burn needed
        self.command_line.append(deltav_C)

    def adjust_raan(self, chaser, chaser_next):
        """
            Given the chaser relative orbital elements w.r.t target,
            ad its next wanted status correct the RAAN.

        Args:
              chaser (Position):
              chaser_next (Position):
        """

        # Evaluate RAAN difference to correct
        # TODO: Think about what happen when i = pi/2...
        i_C_next = chaser.rel_kep.dIx - chaser_next.rel_kep.dIy + chaser.kep.i
        draan = chaser.rel_kep.dIy / np.sin(chaser.kep.i) - chaser_next.rel_kep.dIy / np.sin(i_C_next)

        # Rotational matrix between the two planes
        R = np.identity(3)
        R[0, 0:3] = np.array([np.cos(draan), -np.sin(draan), 0])
        R[1, 0:3] = np.array([np.sin(draan), np.cos(draan), 0])
        R[2, 0:3] = np.array([0, 0, 1])

        # Position at which the burn should occur
        # TODO: Review this calculation...
        h = np.cross(chaser.cartesian.R, chaser.cartesian.V)
        h_next = R.dot(h)
        n = np.cross(h, h_next)
        theta = [np.arccos(n[0]), np.arccos(n[0]) + np.pi]

        # Calculate burn intensity
        # TODO: The velocity has to be propagated to the point where the burn occurs
        deltav = np.norm(chaser.cartesian.v) * np.sqrt(2 * (1 - np.cos(draan)))

        # Evaluate burn direction in chaser frame of reference
        deltav_C = deltav * np.array([np.cos(np.pi / 2 + draan / 2), 0, np.sin(np.pi / 2 + draan / 2)])

        # Add to the command line the burn needed
        self.command_line.append(deltav_C)

    def adjust_perigee(self, chaser, chaser_next):
        """
            Given the chaser relative orbital elements w.r.t target,
            ad its next wanted status correct the perigee argument.

        Args:
              chaser (Position):
              chaser_next (Position):
        """

        # Evaluate perigee difference to correct
        # TODO: Think about what happen when sin(0.5*(w_n - w)) = 0...
        ddEx = chaser.rel_kep.dEx - chaser_next.rel_kep.dEx
        ddEy = chaser.rel_kep.dEy - chaser_next.rel_kep.dEy
        dw = 2 * (np.arctan(-ddEx/ddEy) - chaser.kep.w)

        # Position at which the burn should occur
        # TODO: Review this calculation, think in which of the two the deltav will be less
        theta = [dw/2.0, np.pi + dw/2.0]

        # Calculate burn intensity
        # TODO: The velocity has to be propagated to the point where the burn occurs as well as the radius
        alpha = np.arccos((1 + 2*chaser.kep.e*np.cos(dw/2.0) + np.cos(dw) * chaser.kep.e**2) *
                          np.linalg.norm(chaser.cartesian.R)/((1 - chaser.kep.e**2)*(2*chaser.kep.a - np.linalg.norm(chaser.cartesian.R))))
        deltav = np.linalg.norm(chaser.cartesian.v) * np.sqrt(2 * (1 - np.cos(alpha)))

        # Evaluate burn direction in chaser frame of reference
        deltav_C = deltav * np.array([np.cos(np.pi / 2 + alpha / 2), 0, np.sin(np.pi / 2 + alpha / 2)])

        # Add to the command line the burn needed
        self.command_line.append(deltav_C)

