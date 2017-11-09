#!/usr/bin/env python

import rospy
import message_filters
import space_msgs
from geometry_msgs.msg import Vector3
from path_optimizer import PathOptimizer

from space_msgs.msg import SatelitePose

optimizer = PathOptimizer()

if __name__=='__main__':
    rospy.init_node('path_planner', anonymous=True)

    pub_deltaV = rospy.Publisher('deltaV', Vector3, queue_size=10)

    # Subscribe to target orbital elements
    target_oe_sub = message_filters.Subscriber('target_oe', SatelitePose)

    # Subscribe to chaser orbital elements
    chaser_oe_sub = message_filters.Subscriber('chaser_oe', SatelitePose)

    # optimizer.find_optimal_path(target_oe,chaser_oe)
    ts = message_filters.TimeSynchronizer([target_oe_sub, chaser_oe_sub], 10)
    ts.registerCallback(optimizer.callback)
    rospy.spin()