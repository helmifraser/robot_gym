#!/usr/bin/env python

import gym
import gym_gazebo
import time
import numpy
import random
import time
import rospy
from rosgraph_msgs.msg import Log

import matplotlib
import matplotlib.pyplot as plt

class TurtlebotLearning(object):
    """Reinforcement learning of a turtlebot using Gazebo and OpenAI Gym"""

    def __init__(self):
        super(TurtlebotLearning, self).__init__()
        self.env = gym.make('GazeboCircuitTurtlebotLidar-v0')
        rospy.on_shutdown(self.shutdown_hook)
        # rospy.init_node('turtlebot_learn')
        self.frequency = 10
        self.rate = rospy.Rate(self.frequency)
        self.validation = "empty"
        self.rosout_sub = rospy.Subscriber('/rosout', Log, self.rosout_cb)

        self.wait_for_gym()

    def shutdown_hook(self):
        """
            Callback function invoked when initiating a node shutdown. Called
            before actual shutdown occurs.
        """

        rospy.logwarn("Shutting down all the things fam")

    def wait_for_gym(self):
        completed = False
        while completed is False:
            if "GazeboRosKobuki" in self.validation:
                completed = True
                rospy.logwarn("Successfully initialised, ready to learn! Run 'gzclient' in another terminal")

    def rosout_cb(self, data):
        self.validation = data.msg

    def return_rate(self):
        return self.rate

    def test(self):
        print("testing")

def main():
    try:
        instance = TurtlebotLearning()
        node_rate = instance.return_rate()

        while rospy.is_shutdown() is not True:
            # instance.test()
            node_rate.sleep()

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
