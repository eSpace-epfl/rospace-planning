#!/usr/bin/env python

import rospy
import message_filters
import space_msgs

from geometry_msgs.msg import Vector3Stamped
from space_msgs.msg import SatelitePose
from path_optimizer_v2 import TrajectoryController


if __name__=='__main__':
    rospy.init_node('path_planner', anonymous=True)

    controller = TrajectoryController()

    pub_deltaV = rospy.Publisher('deltaV', Vector3Stamped, queue_size=10)

    # Subscribe to target orbital elements
    target_oe_sub = message_filters.Subscriber('target_oe', SatelitePose)

    # Subscribe to chaser orbital elements
    chaser_oe_sub = message_filters.Subscriber('chaser_oe', SatelitePose)

    # optimizer.find_optimal_path(target_oe,chaser_oe)
    ts = message_filters.TimeSynchronizer([target_oe_sub, chaser_oe_sub], 10)
    ts.registerCallback(controller.callback)

    # TODO: Think about simulation rate...
    rate = rospy.Rate(0.1)
    deltaV = Vector3Stamped()
    while not rospy.is_shutdown():
        deltaV.vector.x = controller.active_command.deltaV_C[0]
        deltaV.vector.y = controller.active_command.deltaV_C[1]
        deltaV.vector.z = controller.active_command.deltaV_C[2]

        if controller.sleep_flag:
            pub_deltaV.publish(deltaV)
            controller.sleep_flag = False
            print deltaV