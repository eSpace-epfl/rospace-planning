#!/usr/bin/env python

import rospy
import message_filters
from path_optimizer import PathOptimizer

from space_msgs.msg import SatelitePose


#pub = rospy.Publisher('new_topic', new_topic_type, queue_size=10)

optimizer = PathOptimizer()

def callback(satelitepose):
    # Implement something
    bla = 0
    print bla
    print "I'm in the callback"


if __name__=='__main__':
    rospy.init_node('path_planner', anonymous=True)

    # Subscribe to target orbital elements
    target_oe_sub = message_filters.Subscriber('target_oe', SatelitePose)

    # Subscribe to chaser orbital elements
    chaser_oe_sub = message_filters.Subscriber('chaser_oe', SatelitePose)

    # optimizer.find_optimal_path(target_oe,chaser_oe)
    r = rospy.Rate(1)

    #while not rospy.is_shutdown():
    #    print target_oe.callback