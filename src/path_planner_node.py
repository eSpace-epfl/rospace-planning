#!/usr/bin/env python

import rospy
import rosbag
import message_filters
import space_msgs


from geometry_msgs.msg import Vector3, PointStamped
from std_msgs.msg import Float64
from space_msgs.msg import SatelitePose


from path_optimizer import PathOptimizer

import numpy as np
import epoch_clock

if __name__=='__main__':
    rospy.init_node('path_planner', anonymous=True)

    optimizer = PathOptimizer()

    # Set when the manoeuvre can start. Afterwards the optimization is evaluated considering the state after 60 seconds.
    # In those 60 seconds (theoretically) from ground you can decide which of the available path you want to follow.
    optimizer.set_manoeuvre_start(2017, 9, 15, 16, 40, 00)

    pub_deltaV = rospy.Publisher('deltaV', Vector3, queue_size=10)

    # Subscribe to target orbital elements
    target_oe_sub = message_filters.Subscriber('target_oe', SatelitePose)

    # Subscribe to chaser orbital elements
    chaser_oe_sub = message_filters.Subscriber('chaser_oe', SatelitePose)

    # optimizer.find_optimal_path(target_oe,chaser_oe)
    ts = message_filters.TimeSynchronizer([target_oe_sub, chaser_oe_sub], 10)
    ts.registerCallback(optimizer.callback)

    rate = rospy.Rate(optimizer.rate)
    deltaV = Vector3()
    while not rospy.is_shutdown():
        oe = optimizer.kep_chaser

        deltaV.x = optimizer.deltaV[0]
        deltaV.y = optimizer.deltaV[1]
        deltaV.z = optimizer.deltaV[2]

        if optimizer.sleep_flag:
            pub_deltaV.publish(deltaV)
            rospy.sleep(200)
            optimizer.sleep_flag = False
            print deltaV