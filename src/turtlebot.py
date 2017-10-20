#!/usr/bin/env python

# import gym
# import gym_gazebo
import time
import numpy as np#
import math

import rospy
import std_msgs.msg, geometry_msgs.msg, sensor_msgs.msg
import mlp

class TurtlebotLearning(object):
    """Reinforcement learning of a turtlebot using Gazebo and OpenAI Gym"""

    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        super(TurtlebotLearning, self).__init__()
        rospy.on_shutdown(self.shutdown_hook)
        rospy.init_node('turtlebot_learn')

        self.frequency = 10
        self.rate = rospy.Rate(self.frequency)

        self.cmd_vel_pub_ = rospy.Publisher('/mobile_base/commands/velocity', geometry_msgs.msg.Twist, queue_size=10)

        self.laser_scan_sub_ = rospy.Subscriber('/scan', sensor_msgs.msg.LaserScan, self.scan_cb)

        self.laser_msg = sensor_msgs.msg.LaserScan()
        self.scan_angle_min = 0
        self.scan_angle_max = 0
        self.scan_angle_increment = 0
        self.scan_range_min = 0
        self.scan_range_max = 0

        self.segmented_laser_data = [0, 0, 0, 0, 0, 0, 0]

        self.controller = mlp.MLP_NeuralNetwork(input=input_nodes,hidden=hidden_nodes,output=output_nodes)
        rospy.loginfo("Created NN")

    def shutdown_hook(self):
        """
            Callback function invoked when initiating a node shutdown. Called
            before actual shutdown occurs.
        """

        rospy.logwarn("Shutting down all the things fam")
        # os.kill(self.pid, signal.SIGKILL)

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
        if msg.ranges[0] is 'nan':
            self.segmented_laser_data[0] = msg.range_max + 10
        else:
            self.segmented_laser_data[0] = msg.ranges[0]

        for i in range(91, len(msg.ranges) - 91, 91):
            try:
                if msg.ranges[i] is 'nan':
                    rospy.logerr("shit is nan {}".format(i))
                    self.segmented_laser_data[int(i/91)] = msg.range_max + 10
                else:
                    rospy.logerr("shit is not nan {}".format(i))
                    self.segmented_laser_data[int(i/91)] = msg.ranges[i]
            except Exception as e:
                rospy.logerr(i)

        if msg.ranges[6] is 'nan':
            self.segmented_laser_data[6] = msg.range_max + 10
        else:
            self.segmented_laser_data[6] = msg.ranges[637]

        rospy.logwarn("msg: {}".format(msg.ranges[637]))

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
        instance = TurtlebotLearning(input_size, hidden_size, output_size)
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


        while rospy.is_shutdown() is not True:
            # instance.test()
            # instance.publish_vel(twist_msg)
            # instance.extract_scan_params()
            # angle_min, angle_max, angle_increment = instance.return_laser_scan_params()
            # angle_min = int(angle_min*180/math.pi)
            # angle_max = int(angle_max*180/math.pi)
            # angle_increment = angle_increment*180/math.pi
            # print("min: {}, max: {}, inc: {}".format(angle_min, angle_max, angle_increment))
            laser_msg = instance.return_laser_msg()
            print("vals: {}".format(laser_msg))
            node_rate.sleep()

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
