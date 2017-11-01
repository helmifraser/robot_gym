#!/usr/bin/env python

# import gym
# import gym_gazebo
import time
import numpy as np
import math

import rospy
import std_msgs.msg, geometry_msgs.msg, sensor_msgs.msg
from rospy.numpy_msg import numpy_msg
import mlp

default = object()


class TurtlebotController(object):
    """Reinforcement learning of a turtlebot using Gazebo and OpenAI Gym"""

    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        super(TurtlebotController, self).__init__()
        rospy.on_shutdown(self.shutdown_hook)
        rospy.init_node('turtlebot_learn')

        self.frequency = 10
        self.rate = rospy.Rate(self.frequency)

        self.cmd_vel_pub_ = rospy.Publisher('/mobile_base/commands/velocity', geometry_msgs.msg.Twist, queue_size=1)

        self.laser_scan_sub_ = rospy.Subscriber('/laserscan', sensor_msgs.msg.LaserScan, self.scan_cb)

        self.laser_msg = sensor_msgs.msg.LaserScan()
        self.scan_angle_min = 0
        self.scan_angle_max = 0
        self.scan_angle_increment = 0
        self.scan_range_min = 0
        self.scan_range_max = 0

        self.cmd_msg = self.create_zeroed_twist()
        self.segmented_laser_data = [0, 0, 0, 0, 0, 0, 0]
        self.scale_params = [0.10, 10, -1, 1]

        self.controller = mlp.MLP_NeuralNetwork(input=input_nodes,hidden=hidden_nodes,output=output_nodes)
        rospy.loginfo("Created NN")

    def shutdown_hook(self):
        """
            Callback function invoked when initiating a node shutdown. Called
            before actual shutdown occurs.
        """

        rospy.logwarn("Shutting down all the things fam")
        # os.kill(self.pid, signal.SIGKILL)

    def drive(self, segmented_laser_data):
        # if math.isnan(segmented_laser_data)
        self.cmd_msg.linear.x = (segmented_laser_data[0] + segmented_laser_data[3] + segmented_laser_data[6])/(self.laser_msg.range_max + 10)
        self.cmd_msg.angular.z = (segmented_laser_data[6] - segmented_laser_data[0])/(self.laser_msg.range_max + 10)
        rospy.loginfo("linear x: {} angular z: {}".format(self.cmd_msg.linear.x, self.cmd_msg.angular.z))

        if math.isnan(self.cmd_msg.linear.x) or math.isnan(self.cmd_msg.angular.z):
            return

        self.publish_vel(self.cmd_msg)
        # if segmented_laser_data[0] < 0.5:

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
        self.segmented_laser_data = laser_ranges[0:266:round(len(laser_ranges)/7)]
        # print(self.segmented_laser_data)


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

        if out > parameters[-1]:
            out = parameters[-1]
        elif out < -1*parameters[-1]:
            out = -1*parameters[-1]

        return out

    def return_segmented_laser_data(self):
        return self.segmented_laser_data

    def return_rate(self):
        return self.rate

    def return_mlp(self):
        return self.controller

    def test(self):
        print("testing")

def main():
    try:
        input_size = 8
        hidden_size = 16
        output_size = 2
        instance = TurtlebotController(input_size, hidden_size, output_size)
        node_rate = instance.return_rate()
        mlp_controller = instance.return_mlp()

        nn_inputs = np.ones(input_size)

        wi = np.random.randn(input_size, hidden_size)
        wo = np.random.randn(hidden_size, output_size)

        twist_msg = instance.create_zeroed_twist()

        commands = mlp_controller.feedForward(nn_inputs)
        # print(commands)
        twist_msg.linear.x = commands[0]
        twist_msg.angular.z = commands[1]

        # hello = "yuo"
        # start = input('Enter anything to start: ')
        # print("all good fam", start)

        while rospy.is_shutdown() is not True:

            # instance.test()
            # instance.publish_vel(twist_msg)
            # instance.extract_scan_params()
            # angle_min, angle_max, angle_increment = instance.return_laser_scan_params()
            # angle_min = int(angle_min*180/math.pi)
            # angle_max = int(angle_max*180/math.pi)
            # angle_increment = angle_increment*180/math.pi
            # print("min: {}, max: {}, inc: {}".format(angle_min, angle_max, angle_increment))

            # laser_msg = instance.return_segmented_laser_data()
            # print("vals: {}".format(laser_msg))
            # instance.drive(laser_msg)
            node_rate.sleep()

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
