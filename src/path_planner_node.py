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
    optimizer.set_manoeuvre_start(2017, 9, 15, 12, 40, 00)

    pub_deltaV = rospy.Publisher('deltaV', Vector3, queue_size=10)
    pub_distance = rospy.Publisher('distance_t_c', PointStamped, queue_size=10)

    # Subscribe to target orbital elements
    target_oe_sub = message_filters.Subscriber('target_oe', SatelitePose)

    # Subscribe to chaser orbital elements
    chaser_oe_sub = message_filters.Subscriber('chaser_oe', SatelitePose)

    # optimizer.find_optimal_path(target_oe,chaser_oe)
    ts = message_filters.TimeSynchronizer([target_oe_sub, chaser_oe_sub], 10)
    ts.registerCallback(optimizer.callback)

    rate = rospy.Rate(optimizer.rate)
    while not rospy.is_shutdown():
        oe = optimizer.kep_chaser
        rospy_now = rospy.Time.now()

        deltaV = Vector3()
        deltaV.x = optimizer.deltaV[0]
        deltaV.y = optimizer.deltaV[1]
        deltaV.z = optimizer.deltaV[2]

        distance = PointStamped()
        distance.point.x = optimizer.estimated_distance
        distance.point.y = 0
        distance.point.z = 0
        distance.header.stamp = rospy.get_rostime()

        pub_distance.publish(distance)

        if optimizer.sleep_flag:
            pub_deltaV.publish(deltaV)
            rospy.sleep(200)
            optimizer.sleep_flag = False