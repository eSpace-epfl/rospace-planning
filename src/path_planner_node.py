#!/usr/bin/env python

import rospy
import rosbag
import message_filters
import space_msgs
from geometry_msgs.msg import Vector3
from path_optimizer import PathOptimizer

from space_msgs.msg import SatelitePose
import numpy as np
import epoch_clock
if __name__=='__main__':
    rospy.init_node('path_planner', anonymous=True)

    optimizer = PathOptimizer()
    optimizer.maneouvre_start = 2

    pub_deltaV = rospy.Publisher('deltaV', Vector3, queue_size=10)

    # Subscribe to target orbital elements
    target_oe_sub = message_filters.Subscriber('target_oe', SatelitePose)

    # Subscribe to chaser orbital elements
    chaser_oe_sub = message_filters.Subscriber('chaser_oe', SatelitePose)

    # optimizer.find_optimal_path(target_oe,chaser_oe)
    ts = message_filters.TimeSynchronizer([target_oe_sub, chaser_oe_sub], 10)
    ts.registerCallback(optimizer.callback)

    #rospy.spin()

    rate = rospy.Rate(optimizer.rate)
    while not rospy.is_shutdown():
        oe = optimizer.kep_chaser
        rospy_now = rospy.Time.now()

        deltaV = Vector3()
        deltaV.x = optimizer.deltaV[0]
        deltaV.y = optimizer.deltaV[1]
        deltaV.z = optimizer.deltaV[2]
        msg = SatelitePose()

        msg.header.stamp = rospy_now
        msg.header.frame_id = "teme"

        msg.position.semimajoraxis = oe.a
        msg.position.eccentricity = oe.e
        msg.position.inclination = np.rad2deg(oe.i)
        msg.position.arg_perigee = np.rad2deg(oe.w)
        msg.position.raan = np.rad2deg(oe.O)
        msg.position.true_anomaly = np.rad2deg(0.2)

        msg.orientation.x = 0
        msg.orientation.y = 0
        msg.orientation.z = 0
        msg.orientation.w = 0

        if optimizer.sleep_flag:
            pub_deltaV.publish(deltaV)
            rospy.sleep(200)
            optimizer.sleep_flag = False