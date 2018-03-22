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
    """Class that holds the definition of the orekit propagator."""

    def __init__(self):
        OrekitPropagator.init_jvm()
        self.propagator = OrekitPropagator()
        self.prop_type = ''
        self.spacecraft = 'Satellite'
        self.date = datetime.utcnow()

    def set_spacecraft(self, spacecraft):
        self.spacecraft = spacecraft

    def set_prop_type(self, prop_type):
        self.prop_type = prop_type
        print "Propagator type for spacecraft " + self.spacecraft + " set to: " + prop_type
        print "[WARNING]: Check if cfg file match the type and if all propagators share the same type!"

    def initialize_propagator(self, cfg_filename, satellite_osc_oe, date=datetime.utcnow()):
        # Set date
        self.date = date

        # Search for the configuration file
        abs_path = sys.argv[0]
        path_idx = abs_path.find('nodes')
        abs_path = abs_path[0:path_idx]
        settings_path = abs_path + 'simulator/cso_gnc_sim/cfg/' + cfg_filename
        settings = file(settings_path, 'r')
        propSettings = yaml.load(settings)

        # Initialize propagator
        self.propagator.initialize(propSettings['propagator_settings'], satellite_osc_oe, self.date)

    def change_initial_conditions(self, initial_state, date, mass):
        """
            Allows to change the initial conditions given to the propagator without initializing it again.

        Args:
            propagator (OrekitPropagator): The propagator that has to be changed.
            initial_state (Cartesian): New cartesian coordinates of the initial state.
            date (datetime): New starting epoch.
            mass (float64): Satellite mass.
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
