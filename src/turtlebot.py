#!/usr/bin/env python
import time
import numpy as np
import math
import sys
import os

import rospy
import std_msgs.msg, geometry_msgs.msg, sensor_msgs.msg, kobuki_msgs.msg
from rospy.numpy_msg import numpy_msg

default = object()


class TurtlebotController(object):
    """Controller for a Turtlebot"""

    def __init__(self):
        super(TurtlebotController, self).__init__()
        rospy.on_shutdown(self.shutdown_hook)
        # rospy.init_node('turtlebot_controller')

        self.frequency = 10
        self.rate = rospy.Rate(self.frequency)

        self.cmd_vel_pub_ = rospy.Publisher('/mobile_base/commands/velocity', geometry_msgs.msg.Twist, queue_size=1)

        self.laser_scan_sub_ = rospy.Subscriber('/laserscan', sensor_msgs.msg.LaserScan, self.scan_cb)
        self.bumper_sub_ = rospy.Subscriber('/mobile_base/events/bumper', kobuki_msgs.msg.BumperEvent, self.bumper_cb)

        self.laser_msg = sensor_msgs.msg.LaserScan()
        self.scan_angle_min = 0
        self.scan_angle_max = 0
        self.scan_angle_increment = 0
        self.scan_range_min = 0
        self.scan_range_max = 0

        self.bumper_state = 0

        self.cmd_msg = self.create_zeroed_twist()
        self.segmented_laser_data = np.zeros(13)
        self.scale_params = [0.10, 10, -1, 1]
        rospy.loginfo("Turtlebot: Completed init")

    def shutdown_hook(self):
        """
            Callback function invoked when initiating a node shutdown. Called
            before actual shutdown occurs.
        """

        rospy.logwarn("Turtlebot: Shutting down all the things fam")
        # sys.exit()
        # os.system('kill %d' % os.getpid())

    def create_zeroed_twist(self):
        msg = geometry_msgs.msg.Twist()
        msg.linear.x = 0
        msg.linear.y = 0
        msg.linear.z = 0

        msg.angular.x = 0
        msg.angular.y = 0
        msg.angular.z = 0
        return msg

    def scan_cb(self, msg):
        self.laser_msg = msg
        laser_ranges = np.asarray(msg.ranges)
        laser_ranges = np.nan_to_num(laser_ranges)
        laser_ranges = np.where(laser_ranges == 0, msg.range_max + 10, laser_ranges)
        laser_ranges = np.where(laser_ranges > 30, msg.range_max + 10, laser_ranges)
        # self.segmented_laser_data = laser_ranges[0:266:int(round(len(laser_ranges)/7))]
        self.segmented_laser_data = np.round(laser_ranges, decimals=2)
        # print(self.segmented_laser_data)

    def bumper_cb(self, msg):
        self.bumper_state = msg.state

    def extract_scan_params(self):
        self.scan_angle_min = self.laser_msg.angle_min
        self.scan_angle_max = self.laser_msg.angle_max
        self.scan_angle_increment = self.laser_msg.angle_increment
        self.scan_range_min = self.laser_msg.range_min
        self.scan_range_max = self.laser_msg.range_max

    def publish_vel(self, twist_msg):
        self.cmd_vel_pub_.publish(twist_msg)

    def return_laser_scan_params(self):
        return self.scan_angle_min, self.scan_angle_max, self.scan_angle_increment

    def return_laser_msg(self):
        return self.laser_msg

    def scale_value(self, value, parameters=default):
        if parameters is default:
            parameters = self.scale_params

        out = (((parameters[3] - parameters[2]) * (value - parameters[0])) /
            (parameters[1] - parameters[0])) + parameters[2]

        # if out > parameters[-1]:
        #     out = parameters[-1]
        # elif out < -1*parameters[-1]:
        #     out = -1*parameters[-1]

        return out

    def return_segmented_laser_data(self):
        return self.segmented_laser_data

    def return_rate(self):
        return self.rate

    def return_bumper_state(self):
        return self.bumper_state

    def test(self):
        print("testing")

def main():
    try:
        instance = TurtlebotController()
        node_rate = instance.return_rate()

        while rospy.is_shutdown() is not True:
            # This node should stay alive and purely act as a controller
            node_rate.sleep()

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
