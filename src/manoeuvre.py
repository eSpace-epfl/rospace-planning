
class Manoeuvre(object):

    def __init__(self):
        self.deltaV_C = [0, 0, 0]
        self.ideal_transfer_orbit = []
        self.duration = 0

        self.true_anomaly = 0
        self.mean_anomaly = 0
        self.epoch = 0

        self.theta_diff = None
        self.description = None