import sys
import yaml

from datetime import datetime
from propagator.OrekitPropagator import OrekitPropagator
from org.orekit.propagation import SpacecraftState
from org.orekit.frames import FramesFactory
from org.orekit.orbits import CartesianOrbit
from org.orekit.utils import PVCoordinates
from org.orekit.utils import Constants as Cst
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.time import AbsoluteDate, TimeScalesFactory


class Propagator(object):
    """
        Class that holds the definition of the orekit propagator.

    Attributes:
        propagator (OrekitPropagator): The propagator created from orekit.
        prop_type (str): Propagator type.
        name (str): Name of the satellite of this propagator.
        date (datetime): Propagator actual date.

    """

    def __init__(self):
        OrekitPropagator.init_jvm()
        self.propagator = OrekitPropagator()
        self.prop_type = ''
        self.name = ''
        self.date = None

    def initialize_propagator(self, name, initial_state, start_date=datetime.utcnow()):
        """
            Initialize the propagator.

        Args:
            name (str): Name of the satellite referring to this propagator, should correspond to the name of the
                propagator configuration file!
            initial_state (KepOrbElem): Initial osculating orbital elements of the satellite.
            start_date (datetime): Initial date of the propagator.
        """

        # Set name
        self.name = name

        # Set date
        self.date = start_date

        # Open the configuration file
        abs_path = sys.argv[0]
        path_idx = abs_path.find('nodes')
        abs_path = abs_path[0:path_idx]
        settings_path = abs_path + 'simulator/cso_gnc_sim/cfg/' + name + '.yaml'
        settings = file(settings_path, 'r')
        propSettings = yaml.load(settings)

        # Initialize propagator
        self.propagator.initialize(propSettings['propagator_settings'], initial_state, self.date)

    def change_initial_conditions(self, initial_state, date, mass):
        """
            Allows to change the initial conditions given to the propagator without initializing it again.

        Args:
            initial_state (Cartesian): New initial state of the satellite in cartesian coordinates.
            date (datetime): New starting date of the propagator.
            mass (float64): New satellite mass.
        """
        # Redefine the start date
        self.date = date

        # Create position and velocity vectors as Vector3D
        p = Vector3D(float(initial_state.R[0]) * 1e3, float(initial_state.R[1]) * 1e3,
                     float(initial_state.R[2]) * 1e3)
        v = Vector3D(float(initial_state.V[0]) * 1e3, float(initial_state.V[1]) * 1e3,
                     float(initial_state.V[2]) * 1e3)

        # Initialize orekit date
        seconds = float(date.second) + float(date.microsecond) / 1e6
        orekit_date = AbsoluteDate(date.year,
                                   date.month,
                                   date.day,
                                   date.hour,
                                   date.minute,
                                   seconds,
                                   TimeScalesFactory.getUTC())

        # Extract frame
        inertialFrame = FramesFactory.getEME2000()

        # Evaluate new initial orbit
        initialOrbit = CartesianOrbit(PVCoordinates(p, v), inertialFrame, orekit_date, Cst.WGS84_EARTH_MU)

        # Create new spacecraft state
        newSpacecraftState = SpacecraftState(initialOrbit, mass)

        # Rewrite propagator initial conditions
        self.propagator._propagator_num.setInitialState(newSpacecraftState)
