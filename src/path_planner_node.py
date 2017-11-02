#!/usr/bin/env python

import rospy
from path_optimizer import PathOptimizer

from space_msgs.msg import SatelitePose


#pub = rospy.Publisher('new_topic', new_topic_type, queue_size=10)

optimizer = PathOptimizer()

def callback(satelitepose):
    # Implement something
    bla = 0


if __name__=='__main__':
    rospy.init_node('path_planner', anonymous=True)

    # Subscribe to target orbital elements
    rospy.Subscriber('target_oe', SatelitePose, callback)

    # Subscribe to chaser orbital elements
    rospy.Subscriber('chaser_oe', SatelitePose, callback)

    while not rospy.is_shutdown():
        optimizer.find_optimal_path()